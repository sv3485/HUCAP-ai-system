import { Download } from 'lucide-react'
import type { PredictResponse } from '../lib/api'

interface ExportConfigProps {
  result: PredictResponse | null
  sequenceLength: number
}

export function ExportConfig({ result, sequenceLength }: ExportConfigProps) {
  if (!result) return null

  const handleExport = () => {
    const exportData = {
      protein_name: result.protein_name || "Unknown",
      sequence_length: sequenceLength,
      sequence_type: result.sequence_type,
      uncertainty: result.uncertainty,
      entropy: result.entropy,
      complexity_score: result.complexity_score,
      model: "ESM2-35M",
      threshold: 0.349,
      evaluation: "CAFA-standard",
      explanation: result.explanation,
      predictions: result.results
    }
    const dataStr = JSON.stringify(exportData, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `protein_analysis_${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <button
      onClick={handleExport}
      className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl border border-indigo-500/30 bg-indigo-500/10 px-4 py-3 text-sm font-bold uppercase tracking-widest text-indigo-300 transition-colors hover:bg-indigo-500/20 hover:text-indigo-200 shadow-inner"
    >
      <Download className="h-4 w-4" /> Download Results (JSON)
    </button>
  )
}
