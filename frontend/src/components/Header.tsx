import { motion } from 'framer-motion'
import { Share2 } from 'lucide-react'

export function Header() {
  return (
    <motion.header 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center text-center mb-12"
    >
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mb-6 inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-4 py-1.5 text-xs font-medium text-cyan-300 backdrop-blur-md shadow-lg shadow-cyan-500/5"
      >
        <Share2 className="h-4 w-4" />
        <span className="tracking-wide">Transformer-based GO Annotation (ESM2)</span>
      </motion.div>
      <h1 className="bg-gradient-to-br from-white via-slate-200 to-slate-400 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl lg:text-6xl mb-4 drop-shadow-sm">
        Protein Function <br className="hidden sm:block" /> Prediction System
      </h1>
      <p className="max-w-2xl mx-auto text-slate-400 text-sm sm:text-base font-medium mt-2">
        A research-grade interface mapping biological protein sequences into Hierarchical Gene Ontology structures, validated across 35 Million embedding parameters.
      </p>
    </motion.header>
  )
}
