import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler,
} from 'chart.js'
import './Dashboard.css'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler)

const INITIAL_LOGS = [
  { time: '--:--:--', msg: 'Awaiting first data cycle…', type: 'primary' },
]

// Maximum number of data points to keep in history
const MAX_CHART_POINTS = 24
const MAX_LOG_ENTRIES = 30
const POLL_INTERVAL_MS = 5000

function Dashboard() {
  const [traffic, setTraffic] = useState({ traffic_light: 'red', vehicle_count: 0 })
  const [logs, setLogs] = useState(INITIAL_LOGS)
  const [videoSource, setVideoSource] = useState('video')
  const [chartLabels, setChartLabels] = useState([])
  const [chartValues, setChartValues] = useState([])
  const [isVideoVisible, setIsVideoVisible] = useState(true)

  const prevSignal = useRef(traffic.traffic_light)
  const pollTimerRef = useRef(null)
  const videoRef = useRef(null)

  // ─── Memoized chart dataset (only re-creates when data changes) ───
  const chartData = useMemo(() => ({
    labels: chartLabels,
    datasets: [{
      data: chartValues,
      borderColor: 'rgba(255,255,255,0.7)',
      backgroundColor: 'rgba(255,255,255,0.04)',
      tension: 0.4,
      fill: true,
      pointRadius: 0,
      borderWidth: 1.5,
    }],
  }), [chartLabels, chartValues])

  // ─── Memoized chart options (static — never changes) ───
  const chartOpts = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1c1b1b',
        borderColor: '#262626',
        borderWidth: 1,
        padding: 8,
        titleColor: '#8e9192',
        bodyColor: '#ffffff',
        titleFont: { size: 10, family: 'JetBrains Mono' },
        bodyFont: { size: 12, family: 'JetBrains Mono', weight: '700' },
        cornerRadius: 4,
        displayColors: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
        border: { display: false },
        ticks: { color: '#8e9192', font: { size: 10, family: 'JetBrains Mono' }, padding: 4 },
      },
      x: {
        grid: { display: false },
        border: { display: false },
        ticks: { color: '#8e9192', font: { size: 9, family: 'JetBrains Mono' }, maxTicksLimit: 4, maxRotation: 0 },
      },
    },
  }), [])

  // ─── Stable polling callback ───
  const poll = useCallback(async () => {
    try {
      const { data } = await axios.get('/traffic_status')
      setTraffic(data)

      const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

      setLogs(prev => {
        const next = [...prev]
        if (next.length === 1 && next[0].msg.includes('Awaiting')) next.pop()

        next.unshift({ time: t, msg: `Vehicle stream registered (${data.vehicle_count})`, type: '' })

        if (data.traffic_light !== prevSignal.current) {
          next.unshift({ time: t, msg: `Signal cycle changed → ${data.traffic_light.toUpperCase()}`, type: 'primary' })
          prevSignal.current = data.traffic_light
        }

        if (data.vehicle_count >= 15) {
          next.unshift({ time: t, msg: 'Density threshold exceeded (High)', type: 'primary' })
        }

        return next.length > MAX_LOG_ENTRIES ? next.slice(0, MAX_LOG_ENTRIES) : next
      })

      // Update chart arrays separately (avoids recreating the entire dataset object)
      setChartLabels(prev => {
        const next = [...prev, t]
        return next.length > MAX_CHART_POINTS ? next.slice(1) : next
      })
      setChartValues(prev => {
        const next = [...prev, data.vehicle_count]
        return next.length > MAX_CHART_POINTS ? next.slice(1) : next
      })
    } catch (_) { /* network error — silently retry next cycle */ }
  }, [])

  // ─── Polling lifecycle ───
  useEffect(() => {
    poll()
    pollTimerRef.current = setInterval(poll, POLL_INTERVAL_MS)
    return () => clearInterval(pollTimerRef.current)
  }, [poll])

  // ─── Visibility API: pause video stream when tab is hidden ───
  useEffect(() => {
    const handleVisibility = () => {
      setIsVideoVisible(!document.hidden)
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => document.removeEventListener('visibilitychange', handleVisibility)
  }, [])

  // ─── Switch video source ───
  const switchVideoSource = useCallback(async (source) => {
    try {
      await axios.post('/set_video_source', { source })
      setVideoSource(source)

      const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      setLogs(prev => {
        const next = [...prev]
        next.unshift({ time: t, msg: `Video source switched to ${source}`, type: 'primary' })
        return next.slice(0, MAX_LOG_ENTRIES)
      })
    } catch (err) {
      console.error('Failed to switch source', err)
      const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      setLogs(prev => {
        const next = [...prev]
        next.unshift({ time: t, msg: `Failed to switch to ${source}`, type: 'error' })
        return next.slice(0, MAX_LOG_ENTRIES)
      })
    }
  }, [])

  // ─── Derived display values ───
  const signalLabel = { red: 'Stop', yellow: 'Caution', green: 'Clear' }[traffic.traffic_light] || 'Stop'
  const density = traffic.vehicle_count < 5 ? 'Low' : traffic.vehicle_count < 15 ? 'Moderate' : 'High'
  const densityPct = Math.min(100, Math.round((traffic.vehicle_count / 25) * 100))
  const signalColorClass = `text-${traffic.traffic_light === 'red' ? 'red' : traffic.traffic_light === 'yellow' ? 'yellow' : 'green'}`

  // Video feed URL with cache-bust to force reconnect after tab becomes visible again
  const videoSrc = isVideoVisible ? '/video_feed' : ''

  return (
    <>
      {/* Header */}
      <div className="dash-header">
        <div>
          <h2 className="dash-title">Traffic Dashboard</h2>
          <p className="dash-subtitle">Real-time AI-powered monitoring · Kothrud, Pune</p>
        </div>
      </div>

      {/* KPI cards */}
      <div className="kpi-grid">
        <div className="data-card kpi-card">
          <span className="kpi-label">Vehicle Count</span>
          <span className="kpi-value">{traffic.vehicle_count.toLocaleString()}</span>
          <span className="kpi-meta">Detected this cycle</span>
        </div>

        <div className="data-card kpi-card">
          <span className="kpi-label">Signal Status</span>
          <div className="kpi-signal-row">
            <span className="kpi-value-sm">{signalLabel}</span>
            <div className="kpi-signal-lights">
              <div className={`kpi-light${traffic.traffic_light === 'red' ? ' on-red' : ''}`} />
              <div className={`kpi-light${traffic.traffic_light === 'yellow' ? ' on-yellow' : ''}`} />
              <div className={`kpi-light${traffic.traffic_light === 'green' ? ' on-green' : ''}`} />
            </div>
          </div>
          <span className={`kpi-meta ${signalColorClass}`}>{traffic.traffic_light.toUpperCase()}</span>
        </div>

        <div className="data-card kpi-card">
          <span className="kpi-label">Traffic Density</span>
          <span className="kpi-value-sm">{density}</span>
          <div className="density-bar-track">
            <div className="density-bar-fill" style={{ width: `${densityPct}%` }} />
          </div>
          <span className="kpi-meta">AI classification</span>
        </div>
      </div>

      {/* Main grid */}
      <div className="dash-main">

        {/* Left column — Camera + Chart */}
        <div className="dash-left">
          <div className="data-card cam-panel">
            <div className="cam-header">
              <div className="cam-header-left">
                <span className="material-symbols-outlined">videocam</span>
                <span className="cam-header-label">Live Camera Feed</span>
              </div>
              
              <div className="source-toggle" style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)' }} onClick={() => switchVideoSource(videoSource === 'video' ? 'webcam' : 'video')}>
                <div className={`toggle-slider ${videoSource === 'webcam' ? 'webcam' : ''}`} />
                <span className={`toggle-label ${videoSource === 'video' ? 'active' : ''}`}>VIDEO</span>
                <span className={`toggle-label ${videoSource === 'webcam' ? 'active' : ''}`}>WEBCAM</span>
              </div>

              <span className="cam-badge">
                <span className="pulse-dot" />
                CAM-01
              </span>
            </div>
            <div className="cam-feed">
              {/* Only render the MJPEG stream when the tab is visible.
                  This prevents the browser from buffering frames in the background,
                  which is the #1 cause of memory bloat and eventual lag. */}
              {videoSrc ? (
                <img ref={videoRef} src={videoSrc} alt="Live Traffic Feed" />
              ) : (
                <div className="cam-paused">
                  <span className="material-symbols-outlined">visibility_off</span>
                  <span>Feed paused — tab inactive</span>
                </div>
              )}
              <div className="cam-feed-overlay">
                <span className="cam-chip">HD 1080p</span>
                <span className="cam-chip">30 FPS</span>
              </div>
            </div>
          </div>

          <div className="data-card chart-panel">
            <div className="chart-panel-header">
              <span className="chart-panel-label">Flow Patterns (Live)</span>
              <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--t-variant)' }}>show_chart</span>
            </div>
            <div className="chart-wrap">
              <Line data={chartData} options={chartOpts} />
            </div>
          </div>
        </div>

        {/* Right column — System Logs only (full height) */}
        <div className="dash-right">
          <div className="data-card logs-panel">
            <div className="logs-panel-header">
              <span className="panel-header-text">System Logs</span>
              <span style={{ fontFamily: 'var(--f-mono)', fontSize: 10, letterSpacing: '0.08em', color: 'var(--t-variant)' }}>
                Last {logs.length} entries
              </span>
            </div>
            <div className="logs-list">
              {logs.map((entry, i) => (
                <div key={i} className={`log-entry${i > 0 ? ' dim' : ''}`}>
                  <span className="log-time">{entry.time}</span>
                  <span className={`log-msg${entry.type === 'error' ? ' error' : entry.type === 'primary' ? ' primary' : ''}`}>
                    {entry.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default Dashboard
