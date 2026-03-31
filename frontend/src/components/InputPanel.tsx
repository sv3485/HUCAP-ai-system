import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Fingerprint, Cpu, Loader2, Beaker } from 'lucide-react'

interface InputPanelProps {
  sequence: string
  setSequence: (s: string) => void
  topK: number
  setTopK: (k: number) => void
  maxLen: number
  setMaxLen: (l: number) => void
  busy: boolean
  error: string | null
  onPredict: () => void
  invalidChars: string[]
  sequenceLength: number
}

export function InputPanel({
  sequence, setSequence, topK, setTopK, maxLen, setMaxLen, busy, error, onPredict, invalidChars, sequenceLength
}: InputPanelProps) {

  const tryExample = () => setSequence("MTRQELGYAFYPRKLV")

  return (
    <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-slate-900/40 p-6 shadow-2xl backdrop-blur-xl">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
      
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
         <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-400 shadow-inner">
            <Fingerprint className="h-5 w-5" />
          </div>
          <h2 className="text-xl font-bold tracking-tight text-white drop-shadow-sm">Sequence Input</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button 
            onClick={tryExample}
            disabled={busy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-indigo-500/30 bg-indigo-500/10 text-xs font-semibold text-indigo-300 hover:bg-indigo-500/20 hover:text-indigo-200 transition-colors"
          >
            <Beaker className="h-3.5 w-3.5" /> Try Example
          </button>
          <div className="flex items-center gap-2 rounded-lg bg-slate-950/50 px-3 py-1.5 text-xs font-medium text-slate-400 border border-white/5">
            <Activity className="h-3 w-3 text-cyan-400" />
            Length: <span className="text-white font-mono">{sequenceLength} AA</span>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {invalidChars.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm flex items-center justify-between"
          >
            <span className="font-medium text-red-200">
              <strong className="text-red-400 mr-2">Invalid Sequence:</strong> 
              Contains non-standard amino acids ({invalidChars.join(', ')})
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="group relative">
        <textarea
          className="relative h-48 w-full resize-none rounded-2xl border border-white/10 bg-slate-950/60 p-5 font-mono text-[13px] leading-relaxed text-slate-100 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all custom-scrollbar shadow-inner"
          value={sequence}
          onChange={(e) => setSequence(e.target.value.toUpperCase())}
          placeholder="Enter amino acid sequence..."
          spellCheck="false"
        />
      </div>

      <div className="mt-8 grid gap-6 sm:grid-cols-2">
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Top-K Responses</label>
          <div className="flex items-center gap-4">
            <input
              type="range"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="h-1.5 w-full appearance-none rounded-full bg-slate-800 accent-cyan-400"
            />
            <span className="w-8 text-right font-mono text-sm font-semibold text-cyan-400">{topK}</span>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Max Sequence Truncation</label>
          <input
            type="number"
            min={64}
            max={4096}
            value={maxLen}
            onChange={(e) => setMaxLen(Number(e.target.value))}
            className="w-full rounded-xl border border-white/5 bg-slate-950/80 px-4 py-2 text-sm font-mono text-slate-200 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 outline-none shadow-inner"
          />
        </div>
      </div>

      <button
        onClick={onPredict}
        disabled={busy || invalidChars.length > 0 || sequenceLength === 0}
        className="group relative mt-8 flex w-full items-center justify-center gap-3 overflow-hidden rounded-2xl bg-gradient-to-r from-cyan-500 to-indigo-600 px-6 py-4 font-bold text-white shadow-[0_0_20px_rgba(6,182,212,0.3)] transition-all hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:scale-[1.01] active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]"
      >
        {busy ? <Loader2 className="h-5 w-5 animate-spin text-white" /> : <Cpu className="h-5 w-5" />}
        <span className="tracking-wide">{busy ? 'Analyzing Protein Sequence...' : 'Initialize Analysis Pipeline'}</span>
      </button>

      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 rounded-xl border border-red-500/20 bg-red-950/50 p-4 text-sm border-l-4 border-l-red-500 shadow-lg"
          >
            <div className="font-bold text-red-400 mb-1 tracking-wide uppercase text-xs">Prediction Failure</div>
            <div className="text-red-200">{error}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
