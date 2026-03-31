import { useCallback, useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Beaker, WifiOff, Wifi, AlertTriangle } from 'lucide-react'
import ForceGraph2D from 'react-force-graph-2d'

import { predict, checkHealth, type PredictResponse, type HealthStatus } from './lib/api'
import { Header } from './components/Header'
import { InputPanel } from './components/InputPanel'
import { ModelInfoCard } from './components/ModelInfoCard'
import { PredictionList } from './components/PredictionList'
import { InsightPanel } from './components/InsightPanel'
import { ExportConfig } from './components/ExportConfig'
import { AccuracyDashboard } from './components/AccuracyDashboard'
import { ResearchDashboard } from './components/ResearchDashboard'

export default function App() {
  const [sequence, setSequence] = useState('')
  const [topK, setTopK] = useState(5)
  const [maxLen, setMaxLen] = useState<number>(1024)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [viewMode, setViewMode] = useState<'list' | 'graph'>('list')
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null)

  // ── Health-check monitoring ───────────────────────────────────────────
  const pollHealth = useCallback(async () => {
    try {
      const h = await checkHealth()
      setHealthStatus(h)
      setBackendReachable(true)
    } catch {
      setHealthStatus(null)
      setBackendReachable(false)
    }
  }, [])

  useEffect(() => {
    pollHealth()
    const interval = setInterval(pollHealth, 30_000)
    return () => clearInterval(interval)
  }, [pollHealth])

  const sequenceLength = useMemo(() => {
    return sequence.trim().length
  }, [sequence])

  const invalidChars = useMemo(() => {
    const VALID = new Set('ACDEFGHIKLMNPQRSTVWY')
    const upper = sequence.trim().toUpperCase()
    const bad = new Set<string>()
    for (const ch of upper) {
      if (!VALID.has(ch)) bad.add(ch)
    }
    return Array.from(bad).sort()
  }, [sequence])

  const graphData = useMemo(() => {
    if (!result) return { nodes: [], links: [] }
    
    const nodes: any[] = []
    const links: any[] = []
    
    nodes.push({ id: 'sequence', name: 'Input Sequence', val: 5, color: '#06b6d4' })

    Object.entries(result.results).forEach(([aspectKey, aspectData]) => {
      const aspectNodeId = `aspect_${aspectKey}`
      nodes.push({ id: aspectNodeId, name: aspectData.aspect_label, val: 3, color: '#818cf8' })
      links.push({ source: 'sequence', target: aspectNodeId })
      
      aspectData.top_predictions.forEach((item) => {
         const termNodeId = item.term
         let color = '#94a3b8'
         if (item.scaled_confidence >= 0.70) color = '#34d399'
         else if (item.scaled_confidence >= 0.40) color = '#fbbf24'
         else if (item.scaled_confidence < 0.40) color = '#f43f5e'
         
         nodes.push({ id: termNodeId, name: item.name || item.term, val: item.scaled_confidence * 10, color })
         links.push({ source: aspectNodeId, target: termNodeId })
      })
    })
    return { nodes, links }
  }, [result])

  async function onPredict() {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const data = await predict({
        sequence,
        top_k: topK,
        max_seq_len: maxLen || null,
      })
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Detailed error occurred while predicting.')
    } finally {
      setBusy(false)
    }
  }



  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 selection:bg-cyan-500/30 font-sans pb-20">
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-20" />
        <div className="absolute -left-[10%] -top-[10%] h-[40%] w-[40%] rounded-full bg-cyan-600/20 blur-[120px]" />
        <div className="absolute -right-[10%] top-[20%] h-[30%] w-[30%] rounded-full bg-purple-600/20 blur-[100px]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <Header />

        {/* Backend connection status banner */}
        <AnimatePresence>
          {backendReachable === false && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 flex items-center justify-center gap-3 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3"
            >
              <WifiOff className="h-4 w-4 text-red-400" />
              <span className="text-sm font-medium text-red-300">Backend not connected — Service may be starting up, please wait...</span>
            </motion.div>
          )}
          {backendReachable === true && healthStatus && !healthStatus.model_loaded && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 flex items-center justify-center gap-3 rounded-xl border border-amber-500/20 bg-amber-500/10 px-4 py-3"
            >
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <span className="text-sm font-medium text-amber-300">Backend running but model loading — Please wait...</span>
            </motion.div>
          )}
          {backendReachable === true && healthStatus?.model_loaded && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-6 flex items-center justify-center gap-2 text-xs text-emerald-400/60"
            >
              <Wifi className="h-3 w-3" />
              <span>Backend connected • Model loaded</span>
            </motion.div>
          )}
        </AnimatePresence>

        <main className="grid gap-8 lg:grid-cols-12 lg:gap-12 items-start">
          {/* LEFT COLUMN: Input & Metas */}
          <motion.section 
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-4 flex flex-col"
          >
            <InputPanel 
              sequence={sequence}
              setSequence={setSequence}
              topK={topK}
              setTopK={setTopK}
              maxLen={maxLen}
              setMaxLen={setMaxLen}
              busy={busy}
              error={error}
              onPredict={onPredict}
              invalidChars={invalidChars}
              sequenceLength={sequenceLength}
            />
            <ModelInfoCard />
          </motion.section>

          {/* RIGHT COLUMN: Results */}
          <motion.section 
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-8"
          >
            <div className="relative h-full min-h-[700px] overflow-hidden rounded-3xl border border-white/10 bg-slate-900/50 p-1 shadow-2xl backdrop-blur-xl">
               <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-purple-500/5" />
               
               <div className="relative h-full rounded-[20px] bg-slate-950/80 p-6 sm:p-8 flex flex-col">
                  {!result ? (
                    <div className="flex flex-1 flex-col items-center justify-center text-center">
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.5, duration: 2, repeat: Infinity, repeatType: "reverse" }}
                        className="mb-8 rounded-full border border-white/5 bg-slate-900/80 p-6 shadow-[0_0_40px_rgba(56,189,248,0.1)]"
                      >
                        <Beaker className="h-14 w-14 text-cyan-500" strokeWidth={1.5} />
                      </motion.div>
                      <h3 className="text-2xl font-bold tracking-tight text-white mb-3">Awaiting Sequence</h3>
                      <p className="max-w-md text-sm text-slate-400 font-medium leading-relaxed">
                        Submit a valid amino acid sequence to generate detailed, multi-branch biological predictions validated continuously via the 35M ESM2 Transformer.
                      </p>
                    </div>
                  ) : (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex flex-col h-full"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6 mb-8">
                        <div>
                          <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400 mb-1">Analysis Complete</h2>
                          <div className="text-2xl font-bold text-white drop-shadow-sm mb-2">{result.model_name}</div>
                          <div className="inline-flex items-center text-sm font-semibold rounded-lg bg-cyan-950/40 border border-cyan-500/20 px-3 py-1.5 shadow-sm">
                            <span className="text-slate-400 mr-2">Detected Protein:</span> 
                            <span className="text-cyan-300">{result.protein_name || "Unknown Protein"}</span>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button 
                            onClick={() => setViewMode('list')}
                            className={`px-4 py-2 text-xs font-bold uppercase tracking-widest rounded-lg transition-colors border ${viewMode === 'list' ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.2)]' : 'bg-transparent text-slate-500 hover:text-slate-300 border-transparent'}`}
                          >
                            List View
                          </button>
                          <button 
                            onClick={() => setViewMode('graph')}
                             className={`px-4 py-2 text-xs font-bold uppercase tracking-widest rounded-lg transition-colors border ${viewMode === 'graph' ? 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.2)]' : 'bg-transparent text-slate-500 hover:text-slate-300 border-transparent'}`}
                          >
                            Explore Hierarchy
                          </button>
                        </div>
                      </div>

                      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-2 pb-6">
                        {viewMode === 'list' && (
                          <InsightPanel 
                            result={result}
                            sequence={sequence}
                          />
                        )}

                        {viewMode === 'graph' ? (
                          <div className="h-[500px] w-full rounded-2xl border border-white/10 overflow-hidden bg-slate-950/80 shadow-inner">
                            <ForceGraph2D
                              graphData={graphData}
                              width={800}
                              height={500}
                              nodeLabel="name"
                              nodeColor={(node: any) => node.color}
                              nodeRelSize={6}
                              linkColor={() => 'rgba(255,255,255,0.05)'}
                              backgroundColor="transparent"
                              d3VelocityDecay={0.8}
                              d3AlphaDecay={0.02}
                            />
                          </div>
                        ) : (
                          <>
                            {Object.entries(result.results).map(([aspectKey, aspectData]) => (
                               <PredictionList 
                                 key={aspectKey}
                                 aspectLabel={`${aspectData.aspect_label} (${aspectKey})`} 
                                 predictions={aspectData.top_predictions} 
                               />
                            ))}
                          </>
                        )}
                        {viewMode === 'list' && (
                           <div className="mt-8 rounded-xl border border-rose-500/10 bg-rose-950/20 p-4 text-xs text-slate-400 shadow-inner">
                             <strong className="text-slate-300">Important Note:</strong> Low-confidence or biologically inconsistent terms are automatically filtered to improve interpretability and lower systematic false positives.
                           </div>
                        )}
                        <ExportConfig result={result} sequenceLength={sequenceLength} />
                      </div>
                    </motion.div>
                  )}
               </div>
            </div>
          </motion.section>
        </main>
        
        {/* Global Accuracy Analysis Panel */}
        <section className="mt-16 pt-8 border-t border-white/5">
          <AccuracyDashboard />
          <ResearchDashboard />
        </section>
        
        <footer className="mt-12 text-center border-t border-white/5 pt-8">
           <p className="text-xs text-slate-500 max-w-2xl mx-auto italic font-medium">
             Predictions are based on sequence patterns and may not fully reflect native biological or biochemical activity in vivo. Ensure secondary experimental validations for critical downstream applications.
           </p>
        </footer>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.02);
          border-radius: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}</style>
    </div>
  )
}
