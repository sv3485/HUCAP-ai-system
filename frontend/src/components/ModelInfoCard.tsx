import { useEffect, useState } from 'react'
import { Database, Network, Target, Settings2, AlertTriangle, CheckCircle2, Info } from 'lucide-react'
import { fetchDatasetInfo, fetchModelInfo, type DatasetInfo, type ModelInfo } from '../lib/api'

export function ModelInfoCard() {
  const [ds, setDs] = useState<DatasetInfo | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)

  useEffect(() => {
    fetchDatasetInfo().then(setDs).catch(console.error)
    fetchModelInfo().then(setModelInfo).catch(console.error)
  }, [])

  return (
    <div className="rounded-xl border border-white/10 bg-slate-950/40 p-5 mt-6 backdrop-blur-md shadow-inner">
      {modelInfo && !modelInfo.is_synced && (
        <div className="mb-4 bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex gap-3 text-red-400">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <div className="text-sm">
            <strong>⚠ Model trained on outdated dataset</strong>
            <p className="text-xs opacity-80 mt-1">
              Architecture profile reflects training dataset ({modelInfo.trained_on_samples?.toLocaleString()} proteins), not available dataset.
            </p>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400">Research Architecture Profile</h3>
        <div className="flex items-center gap-2">
          {modelInfo && (
            <span className="text-xs text-slate-500 flex items-center gap-1 group relative cursor-help">
              <Info className="h-3 w-3" />
              <div className="absolute hidden group-hover:block right-0 top-full mt-2 w-48 text-[10px] bg-slate-800 text-slate-300 p-2 rounded shadow-lg border border-slate-700 z-10">
                Architecture profile reflects training dataset, not available dataset.
              </div>
            </span>
          )}
          {modelInfo && (
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wider ${modelInfo.is_synced ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
              {modelInfo.is_synced ? <CheckCircle2 className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
              {modelInfo.is_synced ? 'SYNCED' : 'DATA MISMATCH'}
            </span>
          )}
          {modelInfo?.dataset_version && (
             <span className="text-[10px] uppercase font-bold text-slate-500 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
               {modelInfo.dataset_version}
             </span>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm text-slate-300">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Network className="h-3.5 w-3.5" />
            <span className="font-semibold text-[10px] uppercase tracking-wider">Backbone</span>
          </div>
          <span className="font-bold text-slate-200">{ds ? `${ds.model} (${ds.parameters})` : 'ESM2 (35M params)'}</span>
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Database className="h-3.5 w-3.5" />
             <span className="font-semibold text-[10px] uppercase tracking-wider">Training Scale</span>
          </div>
          <span className="font-bold text-slate-200">~{modelInfo ? modelInfo.trained_on_samples?.toLocaleString() : '3,319'} proteins</span>
        </div>
        <div className="flex flex-col gap-1.5">
           <div className="flex items-center gap-1.5 text-slate-400">
            <Target className="h-3.5 w-3.5" />
             <span className="font-semibold text-[10px] uppercase tracking-wider">Label Subset</span>
          </div>
          <span className="font-bold text-slate-200">~{ds?.go_terms || 61} strict functions</span>
        </div>
        <div className="flex flex-col gap-1.5">
           <div className="flex items-center gap-1.5 text-slate-400">
            <Settings2 className="h-3.5 w-3.5" />
             <span className="font-semibold text-[10px] uppercase tracking-wider">Calibration</span>
          </div>
          <span className="font-bold text-cyan-400 group relative">
             Dynamic Threshold
             {modelInfo?.calibration_msg && (
               <div className="absolute hidden group-hover:block left-0 top-full mt-2 w-48 text-[10px] bg-slate-800 text-slate-300 p-2 rounded shadow-lg border border-slate-700 z-10 leading-relaxed font-normal normal-case">
                 {modelInfo.calibration_msg}
               </div>
             )}
          </span>
        </div>
      </div>
      
      {/* Precision-Recall Curve UI Hook */}
      <div className="mt-6 pt-5 border-t border-white/5 opacity-80 grid grid-cols-2 gap-4">
         <div>
            <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Test Validation Limits</div>
            <div className="flex flex-col gap-1">
               <span className="text-sm font-bold text-slate-300">Fmax: <span className="text-emerald-400">&ge; 0.40</span></span>
               <span className="text-xs text-slate-400">Seed: 42</span>
            </div>
         </div>
         <div className="flex justify-end">
            <div className="h-16 w-full rounded border border-white/10 bg-slate-900/50 flex items-center justify-center overflow-hidden">
               <span className="text-[9px] uppercase tracking-widest text-slate-500">PR Curve Output Awaiting Epochs</span>
            </div>
         </div>
      </div>
    </div>
  )
}
