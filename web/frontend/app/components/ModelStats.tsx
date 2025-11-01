'use client'

import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function ModelStats() {
  const [models, setModels] = useState<any>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/models`)
      setModels(response.data.models || {})
    } catch (error) {
      console.error('Failed to fetch model stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading statistics...</div>
  }

  const chartData = Object.entries(models).map(([disease, data]: [string, any]) => ({
    disease: disease.charAt(0).toUpperCase() + disease.slice(1),
    valAUC: typeof data.val_auc === 'number' ? data.val_auc : 0,
    valLoss: typeof data.val_loss === 'number' ? data.val_loss : 0,
  }))

  return (
    <div className="space-y-6">
      {/* Model Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        {Object.entries(models).map(([disease, data]: [string, any]) => (
          <div
            key={disease}
            className="bg-slate-50 dark:bg-slate-700 rounded-lg p-6 border border-slate-200 dark:border-slate-600"
          >
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 capitalize">
              {disease}
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-600 dark:text-slate-400">Epoch:</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {data.epoch || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600 dark:text-slate-400">Val Loss:</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {typeof data.val_loss === 'number' ? data.val_loss.toFixed(4) : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600 dark:text-slate-400">Val AUC:</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {typeof data.val_auc === 'number' ? data.val_auc.toFixed(4) : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      {chartData.length > 0 && (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Validation AUC by Disease
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="valAUC" fill="#0ea5e9" name="Validation AUC" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {Object.keys(models).length === 0 && (
        <div className="text-center py-8 text-slate-500 dark:text-slate-400">
          No trained models found. Train models using the training scripts.
        </div>
      )}
    </div>
  )
}

