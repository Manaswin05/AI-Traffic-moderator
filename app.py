import os

os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')

# =============================================================================
# AI Traffic Moderator — Parallel Pipeline Architecture
# =============================================================================
# Thread 1   (FrameCaptureThread):   cv2.VideoCapture.read() → frame_queue
#                                     Paced to source FPS for natural playback
# Thread 2a  (InferenceThread GPU):  frame_queue → YOLO on Arc GPU → result_queue
# Thread 2b  (InferenceThread NPU):  frame_queue → YOLO on NPU     → result_queue
#                                     Both pull from the same queue; whichever
#                                     finishes first pushes to result_queue.
#                                     This drives BOTH chips simultaneously.
# Thread 3   (KMeansTrainThread):    wakes on retrain_event, trains off hot path
# Thread 4   (Flask request thread): result_queue → draw → imencode → MJPEG yield
# =============================================================================
# OMP/MKL threads capped at 4 to prevent PyTorch from fighting hybrid CPU cores.
# =============================================================================

from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from flask_cors import CORS
import cv2
import time
import torch
import numpy as np
import pickle
import threading
import queue
from sklearn.cluster import KMeans
from collections import deque

video_lock = threading.Lock()

# Monkey patch torch.load to use weights_only=False
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO

# Serve React build in production
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'dist')
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)

# Module-level queues for the parallel pipeline
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# retrain_event signals KMeansTrainThread to wake and retrain
retrain_event = threading.Event()

# Lazy-loaded YOLO model — each InferenceThread owns its own instance
_device = None  # last successfully loaded device name (for status reporting)


def _load_model_for_device(device_name):
    """
    Load a YOLO model. Tries PyTorch CPU to ensure it works reliably across environments.
    """
    PT_MODEL = "models/yolov8n.pt"
    m = YOLO(PT_MODEL)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    m.predict(dummy, device="cpu", imgsz=640, verbose=False)
    print(f"[+] YOLO worker ready on CPU OK")
    return m

def _load_best_model():
    """
    Fallback loader if the main one fails.
    """
    return _load_model_for_device("cpu"), "CPU"


VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


# ---------------------------
# Camera / Video Initialization
# ---------------------------
def init_camera():
    # Only use video file as requested
    video_path = "demo_traffic.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"SUCCESS: Using video file: {video_path}")
            return cap, True
    print("WARNING: No video file found. Will stream placeholder frames.")
    return None, False


# ============================================
# K-Means Traffic Classification System
# Memory-optimized for Render free tier (512MB)
# ============================================

class TrafficKMeansOptimized:
    """Memory-efficient K-means for traffic classification"""

    def __init__(self):
        self.MIN_SAMPLES = 20
        self.RETRAIN_INTERVAL = 150
        self.MAX_DATA_SIZE = 200

        self.training_data = deque(maxlen=self.MAX_DATA_SIZE)

        self.model = None
        self.cluster_centers = None
        self.samples_since_last_train = 0
        self.last_train_time = time.time()
        self.is_trained = False

        # Pre-seed with realistic traffic patterns for immediate operation
        initial_patterns = [
            1, 2, 2, 3, 3, 4, 5,       # Low traffic
            7, 8, 9, 10, 11, 12,        # Medium traffic
            15, 17, 18, 20, 22, 25      # High traffic
        ]
        for count in initial_patterns:
            self.training_data.append(count)

        self.load_model()

        if not self.is_trained:
            self.train()

    def add_sample(self, vehicle_count):
        """Add new sample and signal for retraining when threshold is reached"""
        self.training_data.append(vehicle_count)
        self.samples_since_last_train += 1

        should_retrain = False
        if self.samples_since_last_train >= self.RETRAIN_INTERVAL:
            should_retrain = True
        elif time.time() - self.last_train_time > 10800:
            should_retrain = True

        if should_retrain:
            retrain_event.set()

    def train(self):
        """Train K-means model — runs in KMeansTrainThread, never on the hot path"""
        if len(self.training_data) < self.MIN_SAMPLES:
            return

        try:
            X = np.array(list(self.training_data)).reshape(-1, 1)

            self.model = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10,
                max_iter=100
            )
            self.model.fit(X)

            centers = self.model.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            self.label_mapping = {old: new for new, old in enumerate(sorted_indices)}
            self.cluster_centers = np.sort(centers)

            self.samples_since_last_train = 0
            self.last_train_time = time.time()
            self.is_trained = True

            print(f"[+] K-means trained | Centers: {self.cluster_centers.round(1)}")

        except Exception as e:
            print(f"[-] K-means training error: {e}")

    def classify(self, vehicle_count):
        """Classify traffic density — returns (cluster, density_label)"""
        if not self.is_trained or self.model is None:
            return self._fallback_classification(vehicle_count)

        try:
            cluster = self.model.predict([[vehicle_count]])[0]
            cluster = self.label_mapping[cluster]
            density_labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
            density = density_labels[cluster]
            return cluster, density
        except Exception as e:
            print(f"[-] Classification error: {e}")
            return self._fallback_classification(vehicle_count)

    def _fallback_classification(self, vehicle_count):
        """Simple rule-based fallback when model isn't ready"""
        if vehicle_count <= 5:
            return 0, "LOW"
        elif vehicle_count <= 12:
            return 1, "MEDIUM"
        else:
            return 2, "HIGH"

    def save_model(self):
        """Save model to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            model_data = {
                'model': self.model,
                'cluster_centers': self.cluster_centers,
                'label_mapping': self.label_mapping,
                'training_data': list(self.training_data)
            }
            with open('models/traffic_kmeans.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            print("[+] Model saved successfully")
        except Exception as e:
            print(f"[-] Model save error: {e}")

    def load_model(self):
        """Load model from disk if exists"""
        try:
            with open('models/traffic_kmeans.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.cluster_centers = model_data['cluster_centers']
                self.label_mapping = model_data['label_mapping']
                self.is_trained = True
                print(f"[+] Model loaded | Centers: {self.cluster_centers.round(1)}")
        except FileNotFoundError:
            print("[i] No saved model found - will train from seed data")
        except Exception as e:
            print(f"[-] Model load error: {e}")

    def get_stats(self):
        """Get model statistics"""
        return {
            "trained": self.is_trained,
            "samples_collected": len(self.training_data),
            "cluster_centers": self.cluster_centers.tolist() if self.cluster_centers is not None else None,
            "samples_since_retrain": self.samples_since_last_train,
            "next_retrain_in": self.RETRAIN_INTERVAL - self.samples_since_last_train
        }


# ---------------------------
# Background Threads
# ---------------------------

class FrameCaptureThread(threading.Thread):
    """
    Thread 1: Reads frames from cv2.VideoCapture → frame_queue.
    Paces reads to the source FPS so video plays at natural speed.
    Webcam: no sleep — OS driver already paces frame delivery.
    """

    def __init__(self):
        super().__init__(daemon=True, name="FrameCapture")

    @staticmethod
    def _get_frame_delay(capture, is_file):
        """Return sleep time (seconds) between reads to match source FPS."""
        if not is_file:
            return 0.0
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps and 10 <= fps <= 120:
            return 1.0 / fps
        return 1.0 / 25.0  # safe default for unknown FPS metadata

    def run(self):
        while True:
            if cap is None:
                time.sleep(0.1)
                continue

            frame_delay = self._get_frame_delay(cap, is_video_file)
            loop_start = time.monotonic()

            try:
                with video_lock:
                    ret, frame = cap.read()
            except Exception:
                ret, frame = False, None

            # Loop video file when it ends
            if not ret and is_video_file:
                try:
                    with video_lock:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                except Exception:
                    ret, frame = False, None

            if not ret or frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            frame_idx = getattr(self, 'frame_idx', 0) + 1
            self.frame_idx = frame_idx

            # Non-blocking put: drop oldest frame if full, keep freshest
            try:
                frame_queue.put_nowait((frame_idx, frame))
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait((frame_idx, frame))
                except queue.Full:
                    pass

            # Pace to source FPS — subtract time already spent on read
            if frame_delay > 0:
                elapsed = time.monotonic() - loop_start
                sleep_for = frame_delay - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)


class InferenceThread(threading.Thread):
    """
    Thread 2: Pulls frames from frame_queue, runs YOLO on a specific device,
    pushes (frame, vehicles, vehicle_count) to result_queue.

    Two instances run in parallel — one for GPU, one for NPU — so both
    Intel chips are active simultaneously. Frames are distributed between
    workers via the shared frame_queue (thread-safe fair round-robin).
    """

    def __init__(self, device_name):
        super().__init__(daemon=True, name=f"Inference-{device_name}")
        self.device_name = device_name
        self._model = None  # each thread owns its own YOLO/OpenVINO instance

    def _detect_vehicles(self, frame):
        """Run YOLO on this thread's assigned device."""
        results = self._model(
            frame,
            verbose=False,
            imgsz=320,  # Reduced from 640 for significantly faster inference
            conf=0.4,   # ignore detections below 40% confidence
            iou=0.45,   # NMS threshold — removes duplicate boxes on same vehicle
            device="cpu",
        )[0]
        vehicles = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicles.append((class_id, (x1, y1, x2, y2)))
        return vehicles

    def run(self):
        global _device

        # Lazy-load this thread's model on its assigned device
        try:
            self._model = _load_model_for_device(self.device_name)
            _device = self.device_name
        except Exception as e:
            print(f"[-] InferenceThread {self.device_name} failed to load: {e}")
            # Fall back so we always have at least one working inference worker
            try:
                self._model, self.device_name = _load_best_model()
                _device = self.device_name
            except Exception as e2:
                print(f"[-] Fallback also failed: {e2}")
                return  # thread exits; the other worker keeps running

        while True:
            frame_data = frame_queue.get()  # blocking — fair between both workers
            if isinstance(frame_data, tuple):
                frame_idx, frame = frame_data
            else:
                frame_idx, frame = 0, frame_data

            vehicles = self._detect_vehicles(frame)
            vehicle_count = len(vehicles)
            kmeans_system.add_sample(vehicle_count)

            # Non-blocking put: drop oldest result if full
            try:
                result_queue.put_nowait((frame_idx, frame, vehicles, vehicle_count))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put_nowait((frame_idx, frame, vehicles, vehicle_count))
                except queue.Full:
                    pass


class KMeansTrainThread(threading.Thread):
    """
    Thread 3: Waits on retrain_event, then retrains K-means completely
    off the hot path so the video stream never stalls during retraining.
    """

    def __init__(self):
        super().__init__(daemon=True, name="KMeansTrain")

    def run(self):
        while True:
            retrain_event.wait(timeout=10800)  # wake on signal or every 3 hours
            retrain_event.clear()
            if kmeans_system is not None:
                kmeans_system.train()
                kmeans_system.save_model()


# ---------------------------
# Error / Placeholder Frame
# ---------------------------
def make_placeholder_frame(message="No camera available"):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(0, 640, 40):
        cv2.line(frame, (i, 0), (i, 480), (20, 20, 20), 1)
    for i in range(0, 480, 40):
        cv2.line(frame, (0, i), (640, i), (20, 20, 20), 1)
    cv2.putText(frame, message, (80, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
    cv2.putText(frame, "AI Traffic Control System", (100, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


# ---------------------------
# Video Processing Generator (Flask encoding — Thread 4)
# ---------------------------
def process_frame():
    global cap, is_video_file

    last_rendered_idx = -1

    while True:
        try:
            result = result_queue.get(timeout=2.0)
            if len(result) == 4:
                frame_idx, frame, vehicles, vehicle_count = result
            else:
                frame_idx = last_rendered_idx + 1
                frame, vehicles, vehicle_count = result
        except queue.Empty:
            # Timeout: yield placeholder and keep waiting
            if cap is None:
                frame_bytes = make_placeholder_frame("No camera / video source found")
            else:
                frame_bytes = make_placeholder_frame("Camera disconnected or loading...")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        # Skip out-of-order older frames to prevent video jitter/stutter
        if frame_idx < last_rendered_idx:
            continue
        last_rendered_idx = frame_idx

        # Draw bounding boxes
        for class_id, bbox in vehicles:
            x1, y1, x2, y2 = bbox
            label = VEHICLE_CLASSES[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # K-Means classification
        cluster, density = kmeans_system.classify(vehicle_count)

        traffic_state["vehicle_count"] = vehicle_count
        traffic_state["traffic_density"] = density
        traffic_state["cluster"] = cluster

        # AI-driven signal logic
        current_time = time.time()
        elapsed_time = current_time - traffic_state["last_change"]

        if elapsed_time >= traffic_state["timer"]:
            if traffic_state["signal"] == "red":
                if density == "HIGH":
                    traffic_state["signal"] = "green"
                    traffic_state["timer"] = 20
                elif density == "MEDIUM":
                    traffic_state["signal"] = "yellow"
                    traffic_state["timer"] = 8
                else:
                    traffic_state["signal"] = "red"
                    traffic_state["timer"] = 10
            elif traffic_state["signal"] == "green":
                traffic_state["signal"] = "yellow"
                traffic_state["timer"] = 4
            elif traffic_state["signal"] == "yellow":
                traffic_state["signal"] = "red"
                traffic_state["timer"] = 10

            traffic_state["last_change"] = current_time

        # Overlay rendering removed as requested

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ---------------------------
# Queue drain helper (used on source switch)
# ---------------------------
def _drain_queues():
    """
    Flush frame_queue and result_queue after a source switch so stale frames
    from the previous source don't bleed into the new stream.
    Called from set_video_source() — safe to call from any thread.
    """
    for q in (frame_queue, result_queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/api/set_video_source', methods=['POST'])
def set_video_source():
    return jsonify({"status": "error", "message": "Video source switching disabled"}), 400


@app.route('/api/video_feed')
def video_feed():
    return Response(process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/traffic_status')
def traffic_status():
    stats = kmeans_system.get_stats()
    active_devices = [t.device_name for t in _inference_threads if t.is_alive()]
    return jsonify({
        "traffic_light": traffic_state["signal"],
        "vehicle_count": traffic_state["vehicle_count"],
        "traffic_density": traffic_state["traffic_density"],
        "cluster": traffic_state["cluster"],
        "model_trained": stats["trained"],
        "samples_collected": stats["samples_collected"],
        "cluster_centers": stats["cluster_centers"],
        "inference_devices": active_devices,  # e.g. ["GPU", "NPU"]
    })


@app.route('/api/model_info')
def model_info():
    """Get detailed K-means model information"""
    return jsonify(kmeans_system.get_stats())


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Manually trigger model retraining"""
    kmeans_system.train()
    kmeans_system.save_model()
    return jsonify({
        "status": "success",
        "message": "Model retrained successfully",
        "stats": kmeans_system.get_stats()
    })


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


# ---------------------------
# Startup sequence
# ---------------------------

# Initialize K-means
kmeans_system = TrafficKMeansOptimized()

# Initialize capture
cap, is_video_file = init_camera()

traffic_state = {
    "signal": "red",
    "timer": 15,
    "last_change": time.time(),
    "vehicle_count": 0,
    "traffic_density": "LOW",
    "cluster": 0
}

# Start capture thread
_capture_thread = FrameCaptureThread()
_capture_thread.start()

# Start multiple inference workers for multithreading on the Intel Core Ultra CPU
_inference_threads = []
# Using 4 threads instead of 2 to take advantage of 14 cores
for i in range(4):
    t = InferenceThread(f"Worker-{i}")
    t.start()
    _inference_threads.append(t)

# Start K-means background retraining thread
_kmeans_train_thread = KMeansTrainThread()
_kmeans_train_thread.start()


# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("AI Traffic Control System - Starting")
    print("=" * 50)
    print(f"Camera/Video status: {'Ready' if cap is not None else 'NOT FOUND (placeholder mode)'}")
    print("Flask server: http://localhost:5000")
    print("=" * 50)

    try:
        app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")
