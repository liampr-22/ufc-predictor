import type { PredictionResponse } from '../types'
import { formatAmericanOdds } from '../api/client'

interface Props {
  prediction: PredictionResponse
  onFighterClick: (name: string) => void
}

function WinProbBar({ prediction, onFighterClick }: Props) {
  const { fighter_a, fighter_b } = prediction
  const probA = Math.round(fighter_a.win_prob * 100)
  const probB = Math.round(fighter_b.win_prob * 100)
  const favA = fighter_a.win_prob >= fighter_b.win_prob

  return (
    <div className="space-y-3">
      {/* Fighter A */}
      <div>
        <div className="flex justify-between items-baseline mb-1.5">
          <button
            onClick={() => onFighterClick(fighter_a.name)}
            className="font-bold text-lg hover:text-ufc-gold transition-colors truncate max-w-[60%]"
          >
            {fighter_a.name}
          </button>
          <div className="flex items-center gap-3 shrink-0">
            <span className={`text-xs font-mono px-2 py-0.5 rounded ${favA ? 'bg-ufc-gold/20 text-ufc-gold' : 'bg-white/5 text-ufc-muted'}`}>
              {formatAmericanOdds(fighter_a.odds.american)}
            </span>
            <span className="font-bold text-xl tabular-nums">{probA}%</span>
          </div>
        </div>
        <div className="h-3 bg-white/5 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${favA ? 'bg-ufc-gold' : 'bg-white/40'}`}
            style={{ width: `${probA}%` }}
          />
        </div>
      </div>

      {/* Fighter B */}
      <div>
        <div className="flex justify-between items-baseline mb-1.5">
          <button
            onClick={() => onFighterClick(fighter_b.name)}
            className="font-bold text-lg hover:text-ufc-gold transition-colors truncate max-w-[60%]"
          >
            {fighter_b.name}
          </button>
          <div className="flex items-center gap-3 shrink-0">
            <span className={`text-xs font-mono px-2 py-0.5 rounded ${!favA ? 'bg-ufc-gold/20 text-ufc-gold' : 'bg-white/5 text-ufc-muted'}`}>
              {formatAmericanOdds(fighter_b.odds.american)}
            </span>
            <span className="font-bold text-xl tabular-nums">{probB}%</span>
          </div>
        </div>
        <div className="h-3 bg-white/5 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${!favA ? 'bg-ufc-gold' : 'bg-white/40'}`}
            style={{ width: `${probB}%` }}
          />
        </div>
      </div>
    </div>
  )
}

const METHOD_ROWS = [
  { key: 'decision' as const, label: 'Decision', color: '#4B9FE1' },
  { key: 'ko_tko' as const, label: 'KO / TKO', color: '#E74C3C' },
  { key: 'submission' as const, label: 'Submission', color: '#9B59B6' },
]

function FinishRateBar({ value, color, align }: { value: number; color: string; align: 'left' | 'right' }) {
  const pct = Math.round(value * 100)
  return (
    <div className={`flex items-center gap-1.5 ${align === 'right' ? 'flex-row-reverse' : ''}`}>
      <span className="text-xs font-mono text-ufc-muted w-8 shrink-0 text-right">{pct}%</span>
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.55 }}
        />
      </div>
    </div>
  )
}

function MethodBreakdown({ prediction }: { prediction: PredictionResponse }) {
  const { method_probs, fighter_a_finish_rates, fighter_b_finish_rates, fighter_a, fighter_b } = prediction

  return (
    <div>
      <h3 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-3">
        Method of Victory
      </h3>
      {/* Header */}
      <div className="grid grid-cols-[1fr_auto_1fr] gap-x-3 mb-2">
        <span className="text-xs text-ufc-muted truncate">{fighter_a.name.split(' ').pop()} history</span>
        <span className="text-xs text-ufc-muted text-center w-24">This fight</span>
        <span className="text-xs text-ufc-muted text-right truncate">{fighter_b.name.split(' ').pop()} history</span>
      </div>
      {METHOD_ROWS.map(({ key, label, color }) => (
        <div key={key} className="grid grid-cols-[1fr_auto_1fr] gap-x-3 items-center py-1">
          <FinishRateBar value={fighter_a_finish_rates[key]} color={color} align="left" />
          <div className="flex flex-col items-center w-24">
            <span className="text-xs font-mono text-white font-semibold">{Math.round(method_probs[key] * 100)}%</span>
            <span className="text-[10px] text-ufc-muted">{label}</span>
          </div>
          <FinishRateBar value={fighter_b_finish_rates[key]} color={color} align="right" />
        </div>
      ))}
    </div>
  )
}

function StatRow({ label, value, positive }: { label: string; value: string; positive: boolean | null }) {
  const colourClass =
    positive === null
      ? 'text-white'
      : positive
      ? 'text-green-400'
      : 'text-red-400'

  return (
    <div className="flex justify-between items-center py-2 border-b border-ufc-border last:border-0">
      <span className="text-sm text-ufc-muted">{label}</span>
      <span className={`text-sm font-mono font-semibold ${colourClass}`}>{value}</span>
    </div>
  )
}

function KeyDiffs({ prediction }: { prediction: PredictionResponse }) {
  const { key_differentials, fighter_a } = prediction
  const favA = prediction.fighter_a.win_prob >= prediction.fighter_b.win_prob

  const rows: { label: string; value: string; positive: boolean | null }[] = [
    {
      label: `Elo advantage (${favA ? fighter_a.name.split(' ').pop() : prediction.fighter_b.name.split(' ').pop()})`,
      value: `${Math.abs(Math.round(key_differentials.elo_delta))} pts`,
      positive: null,
    },
  ]

  if (key_differentials.reach_delta !== null) {
    const d = key_differentials.reach_delta
    rows.push({
      label: `Reach Δ (${fighter_a.name.split(' ').pop()})`,
      value: `${d >= 0 ? '+' : ''}${d.toFixed(1)}"`,
      positive: d > 0 ? true : d < 0 ? false : null,
    })
  }

  return (
    <div>
      <h3 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-2">
        Key Differentials
      </h3>
      <div>
        {rows.map((r) => (
          <StatRow key={r.label} {...r} />
        ))}
      </div>
    </div>
  )
}

export default function PredictionCard({ prediction, onFighterClick }: Props) {
  return (
    <div className="card p-6 space-y-6">
      <WinProbBar prediction={prediction} onFighterClick={onFighterClick} />
      <div className="border-t border-ufc-border" />
      <MethodBreakdown prediction={prediction} />
      <div className="border-t border-ufc-border" />
      <KeyDiffs prediction={prediction} />
    </div>
  )
}
