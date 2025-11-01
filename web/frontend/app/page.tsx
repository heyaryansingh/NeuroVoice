'use client'

import { useState } from 'react'
import { Upload, Brain, Activity, BarChart3, FileVideo, Music } from 'lucide-react'
import PredictionForm from './components/PredictionForm'
import AnalysisResults from './components/AnalysisResults'
import ModelStats from './components/ModelStats'

export default function Home() {
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 shadow-sm border-b border-slate-200 dark:border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-primary-600" />
              <div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
                  NeuroVoice
                </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Multimodal AI Disease Detection
                </p>
              </div>
            </div>
            <nav className="flex space-x-4">
              <a href="#predict" className="text-slate-600 hover:text-primary-600 dark:text-slate-300">
                Predict
              </a>
              <a href="#analyze" className="text-slate-600 hover:text-primary-600 dark:text-slate-300">
                Analyze
              </a>
              <a href="#stats" className="text-slate-600 hover:text-primary-600 dark:text-slate-300">
                Statistics
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Detect Neurodegenerative Diseases from Speech & Facial Expressions
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            Advanced multimodal AI system for early detection of Alzheimer's, Parkinson's, 
            and Depression using state-of-the-art deep learning models.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-6 border border-slate-200 dark:border-slate-700">
            <Music className="h-10 w-10 text-primary-600 mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
              Audio Analysis
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Wav2Vec2 transformer embeddings for speech pattern analysis
            </p>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-6 border border-slate-200 dark:border-slate-700">
            <FileVideo className="h-10 w-10 text-primary-600 mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
              Video Analysis
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Facial landmarks and emotion recognition via MediaPipe
            </p>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-6 border border-slate-200 dark:border-slate-700">
            <Activity className="h-10 w-10 text-primary-600 mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
              Multimodal Fusion
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Cross-modal attention for enhanced diagnostic accuracy
            </p>
          </div>
        </div>

        {/* Prediction Section */}
        <section id="predict" className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-8 mb-12 border border-slate-200 dark:border-slate-700">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6 flex items-center">
            <Upload className="h-6 w-6 mr-2 text-primary-600" />
            Disease Prediction
          </h2>
          <PredictionForm onResults={setResults} loading={loading} setLoading={setLoading} />
        </section>

        {/* Results Section */}
        {results && (
          <section id="analyze" className="mb-12">
            <AnalysisResults results={results} />
          </section>
        )}

        {/* Statistics Section */}
        <section id="stats" className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-8 border border-slate-200 dark:border-slate-700">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6 flex items-center">
            <BarChart3 className="h-6 w-6 mr-2 text-primary-600" />
            Model Statistics
          </h2>
          <ModelStats />
        </section>
      </section>

      {/* Footer */}
      <footer className="bg-slate-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-slate-400">
            NeuroVoice - Research tool for disease detection. Not for clinical diagnosis.
          </p>
        </div>
      </footer>
    </main>
  )
}

