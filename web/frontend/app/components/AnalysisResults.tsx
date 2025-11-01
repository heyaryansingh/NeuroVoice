'use client'

import { AlertCircle, TrendingUp, TrendingDown, Brain } from 'lucide-react'

interface AnalysisResultsProps {
  results: any
}

export default function AnalysisResults({ results }: AnalysisResultsProps) {
  const { prediction, confidence, disease, status } = results

  const diseaseNames: Record<string, string> = {
    alzheimer: "Alzheimer's Disease",
    parkinson: "Parkinson's Disease",
    depression: "Depression",
  }

  const predictionText = prediction === 1 ? 'Positive' : 'Negative'
  const predictionColor = prediction === 1 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
  const confidenceLevel = confidence > 0.7 ? 'High' : confidence > 0.4 ? 'Medium' : 'Low'

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-8 border border-slate-200 dark:border-slate-700">
      <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6 flex items-center">
        <Brain className="h-6 w-6 mr-2 text-primary-600" />
        Prediction Results
      </h2>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Prediction Card */}
        <div className="bg-gradient-to-br from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 rounded-lg p-6 border border-primary-200 dark:border-primary-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
              Diagnosis
            </h3>
            {prediction === 1 ? (
              <TrendingUp className="h-6 w-6 text-red-600" />
            ) : (
              <TrendingDown className="h-6 w-6 text-green-600" />
            )}
          </div>
          <div className="space-y-2">
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Disease: <span className="font-semibold">{diseaseNames[disease] || disease}</span>
            </p>
            <p className={`text-2xl font-bold ${predictionColor}`}>
              {predictionText}
            </p>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Confidence: <span className="font-semibold">{confidenceLevel}</span> ({Math.round(confidence * 100)}%)
            </p>
          </div>
        </div>

        {/* Confidence Card */}
        <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-6 border border-slate-200 dark:border-slate-600">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Confidence Score
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-600 dark:text-slate-400">Confidence</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {Math.round(confidence * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all ${
                    confidence > 0.7
                      ? 'bg-red-500'
                      : confidence > 0.4
                      ? 'bg-yellow-500'
                      : 'bg-green-500'
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm">
              {confidence > 0.7 ? (
                <>
                  <AlertCircle className="h-4 w-4 text-red-600" />
                  <span className="text-slate-600 dark:text-slate-400">
                    High confidence prediction
                  </span>
                </>
              ) : confidence > 0.4 ? (
                <>
                  <AlertCircle className="h-4 w-4 text-yellow-600" />
                  <span className="text-slate-600 dark:text-slate-400">
                    Moderate confidence - consult healthcare provider
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4 text-green-600" />
                  <span className="text-slate-600 dark:text-slate-400">
                    Low confidence - may require additional testing
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <AlertCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
          <div className="text-sm text-yellow-800 dark:text-yellow-300">
            <p className="font-semibold mb-1">Important Disclaimer</p>
            <p>
              This tool is for research purposes only and should not be used for clinical diagnosis. 
              Always consult with qualified healthcare professionals for medical advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

