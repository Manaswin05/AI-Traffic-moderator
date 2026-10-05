import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

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
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# retrain_event signals KMeansTrainThread to wake and retrain
retrain_event = threading.Event()

# Lazy-loaded YOLO model — each InferenceThread owns its own instance
_device = None  # last successfully loaded device name (for status reporting)


def _load_model_for_device(device_name):
    """
    Load a YOLO OpenVINO model instance for a specific device.
    Each InferenceThread gets its own model instance — OpenVINO requires
    separate Core/model objects per execution context.
    Returns the loaded YOLO model or raises on failure.
    """
    OV_MODEL = "models/yolov8n_openvino_model"
    m = YOLO(OV_MODEL)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    m.predict(dummy, device=device_name, imgsz=640, verbose=False)
    print(f"[+] YOLO worker ready on {device_name} ✓")
    return m


def _load_best_model():
    """
    Single-device fallback loader used when dual-device startup fails.
    Priority: GPU → NPU → CPU (OpenVINO) → CPU-PT (plain PyTorch)
    """
    OV_MODEL = "models/yolov8n_openvino_model"
    PT_MODEL  = "models/yolov8n.pt"

    if os.path.isdir(OV_MODEL):
        for device in ("GPU", "NPU", "CPU"):
            try:
                m = _load_model_for_device(device)
                return m, device
            except Exception as e:
                print(f"[-] OpenVINO {device} unavailable: {e}")

    print("[!] Falling back to PyTorch CPU model.")
    m = YOLO(PT_MODEL)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    m.predict(dummy, imgsz=640, verbose=False)
    print("[+] YOLO loaded via PyTorch CPU ✓")
    return m, "CPU-PT"


VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


# ---------------------------
# Camera / Video Initialization
# ---------------------------
def init_camera():
    # 1. Try a demo video file first (good for cloud/Render deployment)
    video_path = os.environ.get("VIDEO_SOURCE", "demo_traffic.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"SUCCESS: Using video file: {video_path}")
            return cap, True  # (capture, is_video_file)
        else:
            print(f"WARNING: Found file {video_path} but could not open it (might be a Git LFS pointer).")

    # 2. Try physical webcam only if NOT running in a cloud container
    is_cloud = os.environ.get("SPACE_ID") or os.environ.get("RENDER") or os.environ.get("PORT") == "7860"
    if not is_cloud:
        for index in [0, 1, 2]:
            try:
                cam = cv2.VideoCapture(index)
                if cam.isOpened():
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cam.set(cv2.CAP_PROP_FPS, 30)

                    for _ in range(5):
                        cam.read()

                    ret, frame = cam.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"SUCCESS: Camera {index} opened. Shape: {frame.shape}")
                        return cam, False
                    else:
                        cam.release()
            except Exception:
                pass
    else:
        print("INFO: Headless cloud environment detected. Skipping webcam detection loop.")

    print("WARNING: No camera or video file found. Will stream placeholder frames.")
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

            # Non-blocking put: drop oldest frame if full, keep freshest
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(frame)
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
            imgsz=640,
            conf=0.4,   # ignore detections below 40% confidence
            iou=0.45,   # NMS threshold — removes duplicate boxes on same vehicle
            device=self.device_name,
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
            frame = frame_queue.get()  # blocking — fair between both workers

            vehicles = self._detect_vehicles(frame)
            vehicle_count = len(vehicles)
            kmeans_system.add_sample(vehicle_count)

            # Non-blocking put: drop oldest result if full
            try:
                result_queue.put_nowait((frame, vehicles, vehicle_count))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put_nowait((frame, vehicles, vehicle_count))
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

    while True:
        try:
            frame, vehicles, vehicle_count = result_queue.get(timeout=2.0)
        except queue.Empty:
            # Timeout: yield placeholder and keep waiting
            if cap is None:
                frame_bytes = make_placeholder_frame("No camera / video source found")
            else:
                frame_bytes = make_placeholder_frame("Camera disconnected or loading...")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

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

        # Overlay
        signal_colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
        density_colors = {"LOW": (0, 255, 0), "MEDIUM": (0, 255, 255), "HIGH": (0, 0, 255)}
        sig_color = signal_colors.get(traffic_state["signal"], (255, 255, 255))
        dens_color = density_colors.get(density, (255, 255, 255))

        cv2.putText(frame, f"Signal: {traffic_state['signal'].upper()}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, sig_color, 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"AI Density: {density}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dens_color, 2)
        cv2.putText(frame, f"Cluster: {cluster}",
                    (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

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
@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global cap, is_video_file
    data = request.json
    source_type = data.get('source', 'video')

    if source_type == 'webcam':
        new_cap = None
        for index in [0, 1, 2]:
            temp_cap = cv2.VideoCapture(index)
            if temp_cap.isOpened():
                temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                temp_cap.set(cv2.CAP_PROP_FPS, 30)
                # Flush the first few frames so the webcam stabilises before streaming
                for _ in range(5):
                    temp_cap.read()
                ret, frame = temp_cap.read()
                if ret and frame is not None and frame.size > 0:
                    new_cap = temp_cap
                    new_is_file = False
                    found_index = index
                    break
                else:
                    temp_cap.release()

        if new_cap is None:
            return jsonify({"status": "error", "message": "No webcam found"}), 404

        # Swap atomically under the lock so FrameCaptureThread never reads
        # a half-released capture object
        with video_lock:
            old_cap = cap
            cap = new_cap
            is_video_file = new_is_file

        if old_cap is not None:
            old_cap.release()

        # Flush stale frames from previous source out of the pipeline queues
        _drain_queues()

        return jsonify({"status": "success", "message": f"Switched to webcam {found_index}"})

    else:
        video_path = os.environ.get("VIDEO_SOURCE", "demo_traffic.mp4")
        if not os.path.exists(video_path):
            return jsonify({"status": "error", "message": "Demo video not found"}), 404

        new_cap = cv2.VideoCapture(video_path)
        if not new_cap.isOpened():
            new_cap.release()
            return jsonify({"status": "error", "message": "Could not open demo video"}), 500

        with video_lock:
            old_cap = cap
            cap = new_cap
            is_video_file = True

        if old_cap is not None:
            old_cap.release()

        _drain_queues()

        return jsonify({"status": "success", "message": "Switched to demo video"})


@app.route('/video_feed')
def video_feed():
    return Response(process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/traffic_status')
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


@app.route('/model_info')
def model_info():
    """Get detailed K-means model information"""
    return jsonify(kmeans_system.get_stats())


@app.route('/train_model', methods=['POST'])
def train_model():
    """Manually trigger model retraining"""
    kmeans_system.train()
    kmeans_system.save_model()
    return jsonify({
        "status": "success",
        "message": "Model retrained successfully",
        "stats": kmeans_system.get_stats()
    })


# Serve React frontend for all non-API routes (SPA support)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


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

# Start one inference worker per hardware device (GPU + NPU run concurrently)
_inference_threads = []
for _dev in ("GPU", "NPU"):
    t = InferenceThread(_dev)
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
