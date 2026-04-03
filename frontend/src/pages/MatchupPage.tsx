import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import FighterSearch from '../components/FighterSearch'
import PredictionCard from '../components/PredictionCard'
import FighterSidebar from '../components/FighterSidebar'
import { predict } from '../api/client'
import type { PredictionResponse } from '../types'

export default function MatchupPage() {
  const [fighterA, setFighterA] = useState('')
  const [fighterB, setFighterB] = useState('')
  const [sidebarFighter, setSidebarFighter] = useState<string | null>(null)

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)

  const {
    mutate,
    isPending,
    error,
    reset,
  } = useMutation<PredictionResponse, Error, { a: string; b: string }>({
    mutationFn: ({ a, b }) => predict(a, b),
    onSuccess: (data) => setPrediction(data),
  })

  function handlePredict() {
    if (!fighterA || !fighterB) return
    reset()
    mutate({ a: fighterA, b: fighterB })
  }

  function handleSwap() {
    const tmp = fighterA
    setFighterA(fighterB)
    setFighterB(tmp)
    if (prediction) {
      setPrediction({
        ...prediction,
        fighter_a: prediction.fighter_b,
        fighter_b: prediction.fighter_a,
        key_differentials: {
          elo_delta: -prediction.key_differentials.elo_delta,
          reach_delta: prediction.key_differentials.reach_delta != null ? -prediction.key_differentials.reach_delta : null,
          height_delta: prediction.key_differentials.height_delta != null ? -prediction.key_differentials.height_delta : null,
          age_delta: prediction.key_differentials.age_delta != null ? -prediction.key_differentials.age_delta : null,
        },
      })
    }
  }

  const canPredict = fighterA.trim() !== '' && fighterB.trim() !== '' && fighterA !== fighterB

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center space-y-2 pt-4">
        <h1 className="text-3xl font-extrabold tracking-tight">Fight Predictor</h1>
        <p className="text-ufc-muted text-sm">
          ML-powered win probability, method breakdown &amp; implied odds
        </p>
      </div>

      {/* Inputs */}
      <div className="card p-6">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-end gap-4">
          <FighterSearch
            label="Fighter A"
            value={fighterA}
            onChange={setFighterA}
            onProfileClick={setSidebarFighter}
            excludeName={fighterB}
          />

          {/* Swap */}
          <button
            type="button"
            onClick={handleSwap}
            className="sm:mb-0 self-center sm:self-end text-ufc-muted hover:text-white transition-colors px-2 py-2 text-lg shrink-0"
            title="Swap fighters"
          >
            ⇄
          </button>

          <FighterSearch
            label="Fighter B"
            value={fighterB}
            onChange={setFighterB}
            onProfileClick={setSidebarFighter}
            excludeName={fighterA}
          />
        </div>

        <div className="mt-5 flex flex-col sm:flex-row items-center gap-3">
          <button
            type="button"
            onClick={handlePredict}
            disabled={!canPredict || isPending}
            className="btn-primary w-full sm:w-auto"
          >
            {isPending ? 'Calculating…' : 'Predict Fight'}
          </button>

          {(prediction || error) && (
            <button
              type="button"
              onClick={() => { reset(); setPrediction(null); setFighterA(''); setFighterB('') }}
              className="text-sm text-ufc-muted hover:text-white transition-colors"
            >
              Clear
            </button>
          )}
        </div>

        {!canPredict && fighterA !== '' && fighterB !== '' && fighterA === fighterB && (
          <p className="mt-2 text-xs text-red-400">Select two different fighters.</p>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="card p-4 border-red-500/30">
          <p className="text-red-400 text-sm font-medium">Prediction failed</p>
          <p className="text-ufc-muted text-xs mt-1">{error.message}</p>
          <p className="text-ufc-muted text-xs mt-1">
            Make sure both fighters have fight history in the database and the model is trained.
          </p>
        </div>
      )}

      {/* Prediction result */}
      {prediction && !error && (
        <div>
          <h2 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-3">
            Prediction
          </h2>
          <PredictionCard prediction={prediction} onFighterClick={setSidebarFighter} />
        </div>
      )}

      {/* Empty state */}
      {!prediction && !error && !isPending && (
        <div className="text-center py-16 text-ufc-muted text-sm space-y-2">
          <p className="text-4xl">🥊</p>
          <p>Search for two fighters and click <strong className="text-white">Predict Fight</strong></p>
          <p className="text-xs">Click any fighter name to view their full profile</p>
        </div>
      )}

      {/* Fighter sidebar */}
      {sidebarFighter && (
        <FighterSidebar name={sidebarFighter} onClose={() => setSidebarFighter(null)} />
      )}
    </div>
  )
}
