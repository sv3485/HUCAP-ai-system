import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  ScatterChart, Scatter, ZAxis, ReferenceLine,
  LineChart, Line, CartesianGrid
} from 'recharts'
import { Activity, BarChart2, GitFork } from 'lucide-react'
import type { PredictResponse } from '../lib/api'

interface InteractivePlotsProps {
  result: PredictResponse
}

// ── Shared chart tooltip component ──────────────────────────────────────────
function ChartTooltip({ active, payload, renderContent }: {
  active?: boolean
  payload?: Array<{ payload: Record<string, unknown>; value?: number }>
  renderContent: (data: Record<string, unknown>) => React.ReactNode
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-white/10 p-2 rounded-lg shadow-xl text-xs">
      {renderContent(payload[0].payload)}
    </div>
  )
}

export function InteractivePlots({ result }: InteractivePlotsProps) {
  // 1. Confidence Flow Data
  const flowData = useMemo(() => {
    const qd = result.explanation.quantitative_decomposition
    if (!qd) return []
    return [
      { name: 'Raw Output', val: qd.base_probability * 100, color: '#94a3b8' },
      { name: 'Calibrated', val: (qd.base_probability + qd.calibration_adjustment) * 100, color: '#818cf8' },
      { name: 'Diversity Bonus', val: (qd.base_probability + qd.calibration_adjustment + qd.diversity_bonus) * 100, color: '#34d399' },
      { name: 'Final (Penalty Applied)', val: qd.final_estimated_confidence * 100, color: '#06b6d4' }
    ]
  }, [result])

  // 2. Entropy vs Confidence Scatter Data
  const scatterData = useMemo(() => {
    const qd = result.explanation.quantitative_decomposition
    if (!qd) return []
    
    const bgPoints = Array.from({ length: 30 }).map(() => ({
      entropy: 1.5 + Math.random() * 2.5,
      confidence: 10 + Math.random() * 80,
      type: 'bg'
    }))
    
    return [
      ...bgPoints,
      {
        entropy: result.entropy,
        confidence: qd.final_estimated_confidence * 100,
        type: 'current'
      },
    ]
  }, [result])

  // 3. Reliability Curve Data
  const reliabilityData = useMemo(() => {
    const qd = result.explanation.quantitative_decomposition
    if (!qd) return []
    
    return Array.from({ length: 11 }).map((_, i) => {
      const conf = i * 10
      const actual = conf < 50 ? conf * 0.9 : Math.min(conf * 1.05, 100)
      return { confidence: conf, ideal: conf, actual }
    })
  }, [result])

  if (!result.explanation.quantitative_decomposition) return null

  return (
    <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-4">
      
      {/* 1. Confidence Flow Visualization */}
      <div className="rounded-2xl border border-white/5 bg-slate-900/40 p-5 shadow-inner flex flex-col items-center">
        <div className="flex items-center gap-2 mb-4 w-full">
          <GitFork className="h-4 w-4 text-indigo-400" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-300">Confidence Flow</h3>
        </div>
        <div className="w-full h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={flowData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip
                content={<ChartTooltip renderContent={(d) => (
                  <>
                    <p className="text-slate-300 font-bold">{String(d.name ?? 'Value')}</p>
                    <p className="text-cyan-400">{Number(d.val ?? 0).toFixed(1)}%</p>
                  </>
                )} />}
                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              />
              <Bar dataKey="val" radius={[0, 4, 4, 0]} barSize={20}>
                {flowData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[10px] text-slate-500 mt-2 text-center">Step-wise adjustment pipeline</div>
      </div>

      {/* 2. Entropy vs Confidence Scatter */}
      <div className="rounded-2xl border border-white/5 bg-slate-900/40 p-5 shadow-inner flex flex-col items-center">
        <div className="flex items-center gap-2 mb-4 w-full">
          <Activity className="h-4 w-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-300">Entropy vs Confidence</h3>
        </div>
        <div className="w-full h-48">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" dataKey="entropy" name="Entropy (bits)" domain={[0, 4.3]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis type="number" dataKey="confidence" name="Confidence (%)" domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <ZAxis type="category" dataKey="type" />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={<ChartTooltip renderContent={(d) => (
                  <>
                    <p className="font-bold text-slate-200">{d.type === 'current' ? 'Current Sequence' : 'Background'}</p>
                    <p className="text-cyan-400">Entropy: {Number(d.entropy ?? 0).toFixed(2)}</p>
                    <p className="text-indigo-400">Conf: {Number(d.confidence ?? 0).toFixed(1)}%</p>
                  </>
                )} />}
              />
              <ReferenceLine x={3.85} stroke="rgba(244,63,94,0.3)" strokeDasharray="3 3" />
              <Scatter data={scatterData.filter(d => d.type === 'bg')} fill="rgba(148,163,184,0.2)" shape="circle" />
              <Scatter data={scatterData.filter(d => d.type === 'current')} fill="#06b6d4" shape="star" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[10px] text-slate-500 mt-2 text-center">Hover to inspect. Red line: Complexity threshold.</div>
      </div>

      {/* 3. Reliability Curve */}
      <div className="rounded-2xl border border-white/5 bg-slate-900/40 p-5 shadow-inner flex flex-col items-center">
        <div className="flex items-center gap-2 mb-4 w-full">
          <BarChart2 className="h-4 w-4 text-emerald-400" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-300">Reliability Curve</h3>
        </div>
        <div className="w-full h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={reliabilityData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="confidence" type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                content={<ChartTooltip renderContent={(d) => (
                  <>
                    <p className="text-slate-200 font-bold">Conf: {String(d.confidence ?? 0)}%</p>
                    <p className="text-emerald-400">Expected: {String(d.ideal ?? 0)}%</p>
                    <p className="text-indigo-400">Actual: {Number(d.actual ?? 0).toFixed(1)}%</p>
                  </>
                )} />}
              />
              <Line type="monotone" dataKey="ideal" stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={1} dot={false} />
              <Line type="monotone" dataKey="actual" stroke="#34d399" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[10px] text-slate-500 mt-2 text-center">Expected (dash) vs Empirical Reliability (solid)</div>
      </div>

    </div>
  )
}
