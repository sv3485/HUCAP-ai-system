import { motion } from 'framer-motion'
import type { TermScore } from '../lib/api'
import { CheckCircle2, AlertTriangle, ChevronRight, HelpCircle, Info } from 'lucide-react'

interface PredictionListProps {
  aspectLabel: string
  predictions: TermScore[]
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x))
}

function formatPercent01(x: number) {
  return `${(clamp01(x) * 100).toFixed(1)}%`
}

function scoreColor(p: number) {
  if (p >= 0.60) return 'from-emerald-400 to-emerald-600 shadow-[0_0_15px_rgba(52,211,153,0.3)]'
  if (p >= 0.40) return 'from-amber-400 to-yellow-500 shadow-[0_0_15px_rgba(251,191,36,0.3)]'
  return 'from-rose-400 to-red-500 shadow-[0_0_15px_rgba(244,63,94,0.3)]'
}

function PredictionRow({ item, index }: { item: TermScore; index: number }) {
  const calWidth = `${Math.round(clamp01(item.uac) * 100)}%`
  const isHighConf = item.uac >= 0.60
  const isMedConf = item.uac >= 0.40 && item.uac < 0.60
  
  let borderColor = 'border-white/5 bg-slate-950/40'
  if (isHighConf) borderColor = 'border-emerald-500/20 bg-emerald-950/10'
  else if (isMedConf) borderColor = 'border-amber-500/20 bg-amber-950/10'

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.4, type: "spring", stiffness: 100 }}
      className={`group relative overflow-hidden rounded-2xl border ${borderColor} p-4 transition-all hover:bg-slate-800/60 shadow-lg`}
    >
      <div className="relative z-10 flex items-start sm:items-center justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-900 text-cyan-300 shadow-inner border border-white/5">{item.term}</span>
            {isHighConf && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />}
          </div>
          {item.name ? 
            <div className="text-sm font-bold tracking-wide text-slate-100 leading-snug">{item.name}</div> 
            : <div className="text-sm italic text-slate-500">Unknown Function</div>
          }
        </div>
        
        <div className="flex shrink-0 gap-6 text-right items-center">
          <div className="hidden sm:flex flex-col items-end">
            <div className="text-sm font-semibold text-slate-400">{formatPercent01(item.scaled_confidence)}</div>
            <div className="text-[9px] uppercase tracking-widest text-slate-600 font-bold">Uncalibrated</div>
          </div>
          <div className="flex flex-col items-end pr-2">
            <div className={`text-xl font-black drop-shadow-md ${isHighConf ? 'text-emerald-400' : isMedConf ? 'text-amber-400' : 'text-rose-400'}`}>
              {formatPercent01(item.uac)}
            </div>
            <div className="text-[9px] uppercase tracking-widest text-slate-400 font-bold flex items-center gap-1 group-hover:text-cyan-400 transition-colors cursor-help">
              UAC Score <HelpCircle className="h-2.5 w-2.5 opacity-50" />
            </div>
          </div>
        </div>
      </div>
      
      <div className="relative z-10 mt-4 w-full group/bounds">
        {/* Underlying Bounds Indicator line */}
        {item.lower_bound !== undefined && item.upper_bound !== undefined && (
          <div 
            className="absolute -inset-y-1 pointer-events-none z-0 border-x border-white/20 bg-white/5" 
            style={{
              left: `${clamp01(item.lower_bound) * 100}%`,
              width: `${(clamp01(item.upper_bound) - clamp01(item.lower_bound)) * 100}%`
            }}
          />
        )}
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-950 shadow-inner relative z-10">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: calWidth }}
            transition={{ duration: 1, delay: index * 0.05 + 0.1, ease: "easeOut" }}
            className={`h-full bg-gradient-to-r ${scoreColor(item.uac)} rounded-full`} 
          />
        </div>
        
        {/* Hover variance tooltip (mobile visible) */}
        {item.lower_bound !== undefined && item.upper_bound !== undefined && (
           <div className="text-[10px] text-slate-500 mt-1 flex justify-between px-0.5 opacity-60 transition-opacity group-hover/bounds:opacity-100">
             <span>95% CI: {formatPercent01(item.lower_bound)} – {formatPercent01(item.upper_bound)}</span>
             <span>Var: {item.variance?.toFixed(4)}</span>
           </div>
        )}
      </div>
    </motion.div>
  )
}

export function PredictionList({ aspectLabel, predictions }: PredictionListProps) {
  const highConf = predictions.filter(p => p.uac >= 0.60)
  const medConf = predictions.filter(p => p.uac >= 0.40 && p.uac < 0.60)
  const lowConf = predictions.filter(p => p.uac < 0.40)

  return (
    <div className="pt-2">
      <div className="flex items-center gap-2 pb-3 mb-2 border-b border-white/10">
        <ChevronRight className="h-5 w-5 text-indigo-400" />
        <h3 className="text-base font-bold uppercase tracking-widest text-slate-200">
          {aspectLabel} 
        </h3>
        <div className="group relative ml-2 cursor-help">
          <HelpCircle className="h-4 w-4 text-slate-500 hover:text-cyan-400 transition-colors" />
          <div className="pointer-events-none absolute left-0 bottom-full mb-2 w-64 rounded-xl border border-white/10 bg-slate-900 p-3 text-xs text-slate-300 shadow-xl opacity-0 transition-opacity group-hover:opacity-100 backdrop-blur-xl z-50">
            <strong>Molecular Function:</strong> Automatically predicts biochemical activities (e.g., binding or catalysis) at the molecular level using structural patterns.
          </div>
        </div>
      </div>

      {/* Confidence Legend */}
      <div className="flex items-center gap-4 text-xs text-slate-400 bg-slate-900/50 p-2 rounded-lg mb-6 border border-white/5">
        <strong className="text-slate-300 flex items-center gap-1"><Info className="w-3.5 h-3.5"/> Confidence:</strong>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> &ge; 60% Strong</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500"></span> 40-60% Moderate</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-500"></span> &lt; 40% Weak</span>
      </div>

      {predictions.length === 0 ? (
         <div className="rounded-xl border border-dashed border-white/10 p-8 text-center text-sm text-slate-500 italic">
           No confident functional mappings identified matching the biological structural constraints.
         </div>
      ) : (
        <div className="space-y-8">
          {/* High Confidence Block */}
          {highConf.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-4 px-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                <span className="text-xs font-bold uppercase tracking-widest text-emerald-500">Strong Confidence Matches</span>
                <div className="h-px flex-1 bg-gradient-to-r from-emerald-500/20 to-transparent" />
              </div>
              <div className="grid gap-3">
                {highConf.map((item, i) => <PredictionRow key={item.term} item={item} index={i} />)}
              </div>
            </div>
          )}

          {/* Medium Confidence Block */}
          {medConf.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-4 px-2 mt-8">
                <span className="w-2 h-2 rounded-full bg-amber-500/70" />
                <span className="text-xs font-bold uppercase tracking-widest text-amber-400">Moderate Confidence Matches</span>
                <div className="h-px flex-1 bg-gradient-to-r from-amber-500/20 to-transparent" />
              </div>
              <div className="grid gap-3 opacity-90">
                {medConf.map((item, i) => <PredictionRow key={item.term} item={item} index={highConf.length + i} />)}
              </div>
            </div>
          )}

          {/* Low Confidence Block */}
          {lowConf.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-4 px-2 mt-8">
                <AlertTriangle className="h-4 w-4 text-rose-500/70" />
                <span className="text-xs font-bold uppercase tracking-widest text-rose-400">Weak Confidence Matches</span>
                <div className="h-px flex-1 bg-gradient-to-r from-rose-500/20 to-transparent" />
              </div>
              <div className="grid gap-3 opacity-70 grayscale-[30%]">
                {lowConf.map((item, i) => <PredictionRow key={item.term} item={item} index={highConf.length + medConf.length + i} />)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
