import { AlertTriangle, ShieldCheck, Activity, Brain, Fingerprint } from 'lucide-react'
import type { PredictResponse } from '../lib/api'
import { InteractivePlots } from './InteractivePlots'

interface InsightPanelProps {
  result: PredictResponse
  sequence: string
}

export function InsightPanel({ result, sequence }: InsightPanelProps) {
  const getGaugeColor = (u: string) => {
    if (u === 'LOW') return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10 shadow-[0_0_15px_rgba(52,211,153,0.15)]'
    if (u === 'MEDIUM') return 'text-amber-400 border-amber-500/30 bg-amber-500/10 shadow-[0_0_15px_rgba(251,191,36,0.15)]'
    return 'text-rose-400 border-rose-500/30 bg-rose-500/10 shadow-[0_0_15px_rgba(244,63,94,0.15)]'
  }

  return (
    <div className="space-y-6 mb-8">
      {/* 1. Uncertainty Gauge */}
      <div className={`p-4 rounded-2xl border backdrop-blur-md flex items-center justify-between transition-all ${getGaugeColor(result.uncertainty)}`}>
        <div className="flex items-center gap-4">
          {result.uncertainty === 'LOW' ? <ShieldCheck className="h-8 w-8" /> : 
           result.uncertainty === 'MEDIUM' ? <Activity className="h-8 w-8" /> : 
           <AlertTriangle className="h-8 w-8" />}
          <div>
            <div className="text-xs font-bold tracking-widest uppercase opacity-70 mb-1">System Uncertainty Gauge</div>
            <div className="text-xl font-black tracking-wide">{result.uncertainty} UNCERTAINTY</div>
          </div>
        </div>
        <div className="text-right text-xs opacity-80">
          <div>Entropy: {result.entropy.toFixed(2)} bits</div>
          <div>Complexity: {(result.complexity_score * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* 2. Decision Transparency Panel (Rejection System) */}
      {result.prediction_withheld && (
        <div className="rounded-2xl border border-rose-500/40 bg-rose-950/40 p-5 shadow-inner flex gap-4">
          <AlertTriangle className="h-6 w-6 text-rose-500 shrink-0" />
          <div>
            <h3 className="text-lg font-bold text-rose-400 mb-1">Predictions Withheld</h3>
            <p className="text-sm font-medium text-rose-200/80 mb-2">
              The HUCAP framework decision system has rejected these predictions due to unreliability.
            </p>
            <div className="rounded-lg bg-rose-950/60 border border-rose-500/20 p-3 text-sm text-rose-300 font-mono">
              <strong className="text-rose-400">Reason:</strong> {result.withhold_reason}
            </div>
          </div>
        </div>
      )}

      {/* 3. Advanced XAI (Quantitative Decomposition) */}
      {result.explanation.quantitative_decomposition && !result.prediction_withheld && (
        <div className="rounded-2xl border border-indigo-500/30 bg-indigo-950/20 p-5 shadow-inner">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="h-5 w-5 text-indigo-400" />
            <h3 className="text-sm font-bold uppercase tracking-widest text-indigo-300">Confidence Decomposition</h3>
          </div>
          
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
            <div className="bg-slate-900/50 rounded-lg p-3 border border-white/5 text-center">
              <div className="text-[10px] text-slate-400 mb-1 leading-tight h-6">Base<br/>Raw Prob</div>
              <div className="font-mono text-sm text-slate-300">{(result.explanation.quantitative_decomposition.base_probability * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-indigo-500/20 text-center">
              <div className="text-[10px] text-indigo-400 mb-1 leading-tight h-6">Temp<br/>Calibration</div>
              <div className="font-mono text-sm text-indigo-300">{(result.explanation.quantitative_decomposition.calibration_adjustment * 100 > 0 ? '+' : '')}{(result.explanation.quantitative_decomposition.calibration_adjustment * 100).toFixed(1)}%</div>
            </div>
            <div className={`bg-slate-900/50 rounded-lg p-3 border ${result.explanation.quantitative_decomposition.entropy_penalty < 0 ? 'border-rose-500/20' : 'border-white/5'} text-center`}>
              <div className="text-[10px] text-rose-400 mb-1 leading-tight h-6">Entropy<br/>Penalty</div>
              <div className="font-mono text-sm text-rose-300">{(result.explanation.quantitative_decomposition.entropy_penalty * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-emerald-500/20 text-center">
              <div className="text-[10px] text-emerald-400 mb-1 leading-tight h-6">Diversity<br/>Bonus</div>
              <div className="font-mono text-sm text-emerald-300">+{(result.explanation.quantitative_decomposition.diversity_bonus * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-cyan-950/30 rounded-lg p-3 border border-cyan-500/30 text-center shadow-[0_0_10px_rgba(6,182,212,0.1)]">
              <div className="text-[10px] text-cyan-400 font-bold mb-1 leading-tight h-6">Final<br/>Estimate</div>
              <div className="font-mono text-sm text-cyan-300 font-bold">{(result.explanation.quantitative_decomposition.final_estimated_confidence * 100).toFixed(1)}%</div>
            </div>
          </div>
          
          
          {/* Advanced Insights & Confidence Flags */}
          <div className="mt-5 pt-4 border-t border-indigo-500/20 text-sm flex flex-col gap-3">
            <div className="text-indigo-200/80">
              <strong className="text-indigo-400">Counterfactual:</strong> {result.explanation.counterfactual}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 rounded-full border border-white/5 bg-slate-900/60 px-3 py-1.5 text-xs">
                <Fingerprint className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-slate-400">Explanation Confidence:</span>
                <span className="text-cyan-300 font-bold">{((result.explanation.explanation_confidence ?? 0.8) * 100).toFixed(0)}%</span>
              </div>
              {result.explanation.expected_accuracy_range && (
                <div className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-950/30 px-3 py-1.5 text-xs shadow-[0_0_10px_rgba(52,211,153,0.1)]">
                  <Activity className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-slate-300">Expected Empirical Accuracy:</span>
                  <span className="text-emerald-300 font-bold">{result.explanation.expected_accuracy_range}</span>
                </div>
              )}
              {result.explanation.prediction_reliability_score && (
                <div className="flex items-center gap-2 rounded-full border border-purple-500/20 bg-purple-950/30 px-3 py-1.5 text-xs">
                  <ShieldCheck className="w-3.5 h-3.5 text-purple-400" />
                  <span className="text-slate-300">Reliability Score:</span>
                  <span className="text-purple-300 font-bold">{result.explanation.prediction_reliability_score.toFixed(1)} / 10</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4. Interactive Analytical Plots */}
      {!result.prediction_withheld && (
        <InteractivePlots result={result} />
      )}

      {/* 4. Sequence Viewer (Token Attributions) */}
      <div className="rounded-2xl border border-white/10 bg-slate-950 p-5 shadow-inner overflow-hidden flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold uppercase tracking-widest text-slate-300">Residue Importance Viewer</h3>
          <div className="text-[10px] text-slate-500">Attention-based Attribution (Last Layer)</div>
        </div>
        
        {!result.token_attributions || result.token_attributions.length === 0 ? (
          <div className="text-xs text-slate-500 italic p-4 bg-slate-900 rounded-lg text-center">
            Token attributions not available for this sequence length.
          </div>
        ) : (
          <div className="relative">
            <div className="flex flex-wrap gap-[1px] font-mono text-xs max-h-48 overflow-y-auto custom-scrollbar p-2 bg-slate-900 rounded-lg border border-white/5">
              {Array.from(sequence.slice(0, result.token_attributions.length)).map((aa, i) => {
                const score = result.token_attributions![i]
                // Normalize visual scale for typical attention weights (which are small and sum to 1.0)
                const relativeScore = Math.min(score * 20, 1.0) 
                return (
                  <span 
                    key={i} 
                    className="inline-flex w-4 h-5 items-center justify-center rounded-[2px]"
                    style={{ 
                      backgroundColor: `rgba(6, 182, 212, ${relativeScore * 0.8})`,
                      color: relativeScore > 0.4 ? '#fff' : 'rgba(148, 163, 184, 0.8)',
                      fontWeight: relativeScore > 0.4 ? 'bold' : 'normal'
                    }}
                    title={`Pos ${i+1}: ${aa} (Score: ${score.toFixed(4)})`}
                  >
                    {aa}
                  </span>
                )
              })}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}
