import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import axios from 'axios'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import './Analytics.css'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Tooltip, Legend, Filler
)

/* ── Color palette (module-level constant) ─────────── */
const C = {
  white:  'rgba(255,255,255,0.85)',
  white2: 'rgba(255,255,255,0.55)',
  white3: 'rgba(255,255,255,0.35)',
  white4: 'rgba(255,255,255,0.20)',
  white5: 'rgba(255,255,255,0.10)',
  red:    '#ffb4ab',
  yellow: '#eab308',
  green:  '#22c55e',
  blue:   '#60a5fa',
  purple: '#a78bfa',
  cyan:   '#22d3ee',
}

/* ── Shared chart tooltip (module-level constant) ──── */
const tooltipStyle = {
  backgroundColor: '#1c1b1b',
  borderColor: '#262626',
  borderWidth: 1,
  padding: 8,
  titleColor: '#8e9192',
  bodyColor: '#ffffff',
  titleFont: { size: 10, family: 'JetBrains Mono' },
  bodyFont: { size: 12, family: 'JetBrains Mono', weight: '700' },
  cornerRadius: 4,
  displayColors: true,
  boxWidth: 8,
  boxHeight: 8,
  boxPadding: 4,
}

const gridColor = 'rgba(255,255,255,0.04)'
const tickFont = { size: 10, family: 'JetBrains Mono' }

const POLL_INTERVAL_MS = 5000
const MAX_HISTORY = 30

const typeLabels = ['Cars', 'Bikes', 'Buses', 'Trucks', 'Auto']
const typeColors = [C.white, C.white2, C.white3, C.white4, C.white5]

/* ── LocalStorage helpers ──────────────────────────── */
function loadJSON(key, fallback) {
  try {
    const saved = localStorage.getItem(key)
    return saved ? JSON.parse(saved) : fallback
  } catch {
    return fallback
  }
}

function saveJSON(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* quota */ }
}

function Analytics() {
  const [flowData, setFlowData] = useState({ labels: [], datasets: [{ data: [] }] })
  const [histData, setHistData] = useState({ labels: [], datasets: [] })
  const [pieData, setPieData] = useState({ labels: [], datasets: [] })
  const [freqData, setFreqData] = useState({ labels: [], datasets: [] })
  const [kpis, setKpis] = useState({ total: 0, avgPerCycle: 0, peakCount: 0, cycles: 0 })

  // Mutable refs for data that persists across polls without triggering renders
  const historyRef = useRef(loadJSON('analyticsHistory', []))
  const typeAccumRef = useRef(loadJSON('analyticsTypeAccum', { Cars: 0, Bikes: 0, Buses: 0, Trucks: 0, Auto: 0 }))
  const isVisibleRef = useRef(!document.hidden)

  // ─── Stable polling callback ───
  const poll = useCallback(async () => {
    // Skip network request if tab is hidden — saves bandwidth and CPU
    if (!isVisibleRef.current) return

    try {
      const { data } = await axios.get('/traffic_status')
      const count = data.vehicle_count || 0
      const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

      const history = historyRef.current
      history.push({ time: t, count })
      if (history.length > MAX_HISTORY) history.shift()
      saveJSON('analyticsHistory', history)

      // Simulate vehicle type breakdown from count
      const cars  = Math.round(count * 0.45) + Math.floor(Math.random() * 3)
      const bikes = Math.round(count * 0.25) + Math.floor(Math.random() * 2)
      const buses = Math.round(count * 0.08) + Math.floor(Math.random() * 2)
      const trucks = Math.round(count * 0.07) + Math.floor(Math.random() * 1)
      const auto  = Math.max(0, count - cars - bikes - buses - trucks)

      const ta = typeAccumRef.current
      ta.Cars  += cars
      ta.Bikes += bikes
      ta.Buses += buses
      ta.Trucks += trucks
      ta.Auto  += auto
      saveJSON('analyticsTypeAccum', ta)

      // KPIs
      const total = history.reduce((s, h) => s + h.count, 0)
      const peak  = Math.max(...history.map(h => h.count))
      setKpis({
        total,
        avgPerCycle: history.length ? Math.round(total / history.length) : 0,
        peakCount: peak,
        cycles: history.length,
      })

      // Flow line chart
      const flowDisplay = history.slice(-MAX_HISTORY)
      setFlowData({
        labels: flowDisplay.map(h => h.time),
        datasets: [{
          label: 'Vehicles',
          data: flowDisplay.map(h => h.count),
          borderColor: C.white,
          backgroundColor: C.white5,
          tension: 0.4,
          fill: true,
          pointRadius: 0,
          borderWidth: 1.5,
        }],
      })

      // Histogram — last 12 cycle counts
      const last12 = flowDisplay.slice(-12)
      setHistData({
        labels: last12.map(h => h.time),
        datasets: [{
          label: 'Vehicles per Cycle',
          data: last12.map(h => h.count),
          backgroundColor: last12.map(h =>
            h.count >= 15 ? C.red : h.count >= 5 ? C.yellow : C.green
          ),
          borderColor: 'transparent',
          borderRadius: 3,
          barThickness: 18,
        }],
      })

      // Pie — vehicle type distribution (cumulative)
      setPieData({
        labels: ['Cars', 'Bikes', 'Buses', 'Trucks', 'Auto-rickshaw'],
        datasets: [{
          data: [ta.Cars, ta.Bikes, ta.Buses, ta.Trucks, ta.Auto],
          backgroundColor: [C.white, C.white2, C.white3, C.white4, C.white5],
          borderColor: '#141414',
          borderWidth: 2,
          hoverOffset: 6,
        }],
      })

      // Frequency distribution bar — current cycle breakdown
      setFreqData({
        labels: ['Cars', 'Bikes', 'Buses', 'Trucks', 'Auto'],
        datasets: [{
          label: 'Current Cycle',
          data: [cars, bikes, buses, trucks, auto],
          backgroundColor: [C.white, C.white2, C.white3, C.white4, C.white5],
          borderColor: 'transparent',
          borderRadius: 3,
          barThickness: 28,
        }],
      })
    } catch (_) { /* network error — silently retry next cycle */ }
  }, [])

  // ─── Lifecycle: poll + visibility ───
  useEffect(() => {
    const handleVisibility = () => {
      isVisibleRef.current = !document.hidden
      // Immediately poll when tab becomes visible again to refresh stale data
      if (!document.hidden) poll()
    }
    document.addEventListener('visibilitychange', handleVisibility)

    poll()
    const id = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [poll])

  /* ── Memoized chart options (static — never change) ── */
  const lineOpts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false }, tooltip: tooltipStyle },
    scales: {
      y: { beginAtZero: true, grid: { color: gridColor, drawBorder: false }, border: { display: false }, ticks: { color: '#8e9192', font: tickFont, padding: 4 } },
      x: { grid: { display: false }, border: { display: false }, ticks: { color: '#8e9192', font: tickFont, maxTicksLimit: 6, maxRotation: 0 } },
    },
  }), [])

  const barOpts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false }, tooltip: { ...tooltipStyle, displayColors: false } },
    scales: {
      y: { beginAtZero: true, grid: { color: gridColor, drawBorder: false }, border: { display: false }, ticks: { color: '#8e9192', font: tickFont, padding: 4 } },
      x: { grid: { display: false }, border: { display: false }, ticks: { color: '#8e9192', font: { size: 8, family: 'JetBrains Mono' }, maxRotation: 45, minRotation: 45 } },
    },
  }), [])

  const freqBarOpts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    indexAxis: 'y',
    animation: { duration: 300 },
    plugins: { legend: { display: false }, tooltip: { ...tooltipStyle, displayColors: false } },
    scales: {
      x: { beginAtZero: true, grid: { color: gridColor, drawBorder: false }, border: { display: false }, ticks: { color: '#8e9192', font: tickFont } },
      y: { grid: { display: false }, border: { display: false }, ticks: { color: '#c4c7c8', font: { size: 11, family: 'JetBrains Mono', weight: '500' } } },
    },
  }), [])

  const pieOpts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      legend: { display: false },
      tooltip: { ...tooltipStyle, displayColors: true },
    },
    cutout: '55%',
  }), [])

  // Memoized cumulative bar data derived from pieData
  const cumulativeBarData = useMemo(() => ({
    labels: typeLabels,
    datasets: [{
      label: 'Total',
      data: pieData.datasets?.[0]?.data || [0, 0, 0, 0, 0],
      backgroundColor: typeColors,
      borderColor: 'transparent',
      borderRadius: 3,
      barThickness: 36,
    }],
  }), [pieData])

  const cumulativeBarOpts = useMemo(() => ({
    ...barOpts,
    scales: {
      ...barOpts.scales,
      x: { ...barOpts.scales.x, ticks: { ...barOpts.scales.x.ticks, maxRotation: 0, minRotation: 0, font: { size: 11, family: 'JetBrains Mono' } } },
    },
  }), [barOpts])

  return (
    <>
      {/* Header */}
      <div className="analytics-header">
        <div>
          <h2 className="analytics-title">Analytics</h2>
          <p className="analytics-subtitle">Historical traffic data analysis · Kothrud, Pune</p>
        </div>
      </div>

      {/* KPI summary row */}
      <div className="an-kpi-row">
        <div className="data-card an-kpi">
          <span className="an-kpi-label">Total Vehicles</span>
          <span className="an-kpi-value">{kpis.total.toLocaleString()}</span>
          <span className="an-kpi-meta">Cumulative count</span>
        </div>
        <div className="data-card an-kpi">
          <span className="an-kpi-label">Avg / Cycle</span>
          <span className="an-kpi-value">{kpis.avgPerCycle}</span>
          <span className="an-kpi-meta">Per 5s interval</span>
        </div>
        <div className="data-card an-kpi">
          <span className="an-kpi-label">Peak Count</span>
          <span className="an-kpi-value">{kpis.peakCount}</span>
          <span className="an-kpi-meta up">Highest recorded</span>
        </div>
        <div className="data-card an-kpi">
          <span className="an-kpi-label">Data Points</span>
          <span className="an-kpi-value">{kpis.cycles}</span>
          <span className="an-kpi-meta">Cycles captured</span>
        </div>
      </div>

      {/* Flow patterns — full width line chart */}
      <div className="an-grid-full">
        <div className="data-card an-panel">
          <div className="an-panel-header">
            <span className="an-panel-title">
              <span className="material-symbols-outlined">show_chart</span>
              Flow Patterns (Live)
            </span>
            <span className="an-panel-meta">Last 30 readings</span>
          </div>
          <div className="an-chart-area-lg">
            <Line data={flowData} options={lineOpts} />
          </div>
        </div>
      </div>

      {/* 2-col: Histogram + Frequency */}
      <div className="an-grid">
        {/* Histogram */}
        <div className="data-card an-panel">
          <div className="an-panel-header">
            <span className="an-panel-title">
              <span className="material-symbols-outlined">bar_chart</span>
              Vehicle Count Histogram
            </span>
            <span className="an-panel-meta">Last 12 cycles</span>
          </div>
          <div className="an-chart-area">
            <Bar data={histData} options={barOpts} />
          </div>
          <div className="an-legend">
            <div className="an-legend-item"><span className="an-legend-dot" style={{ background: C.green }} /> Low (&lt;5)</div>
            <div className="an-legend-item"><span className="an-legend-dot" style={{ background: C.yellow }} /> Moderate (5-14)</div>
            <div className="an-legend-item"><span className="an-legend-dot" style={{ background: C.red }} /> High (≥15)</div>
          </div>
        </div>

        {/* Frequency distribution — horizontal bar */}
        <div className="data-card an-panel">
          <div className="an-panel-header">
            <span className="an-panel-title">
              <span className="material-symbols-outlined">align_horizontal_left</span>
              Frequency Distribution
            </span>
            <span className="an-panel-meta">Current cycle</span>
          </div>
          <div className="an-chart-area">
            <Bar data={freqData} options={freqBarOpts} />
          </div>
        </div>
      </div>

      {/* 2-col: Pie chart + type breakdown table */}
      <div className="an-grid">
        {/* Doughnut — vehicle type share */}
        <div className="data-card an-panel">
          <div className="an-panel-header">
            <span className="an-panel-title">
              <span className="material-symbols-outlined">donut_large</span>
              Vehicle Type Share
            </span>
            <span className="an-panel-meta">Cumulative</span>
          </div>
          <div className="an-chart-area-pie">
            <Doughnut data={pieData} options={pieOpts} />
          </div>
          <div className="an-legend">
            {typeLabels.map((lbl, i) => (
              <div key={lbl} className="an-legend-item">
                <span className="an-legend-dot" style={{ background: typeColors[i] }} />
                {lbl}
              </div>
            ))}
          </div>
        </div>

        {/* Another angle: per-type cumulative bar */}
        <div className="data-card an-panel">
          <div className="an-panel-header">
            <span className="an-panel-title">
              <span className="material-symbols-outlined">stacked_bar_chart</span>
              Cumulative by Type
            </span>
            <span className="an-panel-meta">All cycles</span>
          </div>
          <div className="an-chart-area">
            <Bar data={cumulativeBarData} options={cumulativeBarOpts} />
          </div>
        </div>
      </div>
    </>
  )
}

export default Analytics
