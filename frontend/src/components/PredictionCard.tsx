import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
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

function MethodBreakdown({ prediction }: { prediction: PredictionResponse }) {
  const { method_probs } = prediction
  const data = [
    { name: 'Decision', value: Math.round(method_probs.decision * 100), color: '#4B9FE1' },
    { name: 'KO / TKO', value: Math.round(method_probs.ko_tko * 100), color: '#E74C3C' },
    { name: 'Submission', value: Math.round(method_probs.submission * 100), color: '#9B59B6' },
  ]

  return (
    <div>
      <h3 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-3">
        Method of Victory
      </h3>
      <ResponsiveContainer width="100%" height={110}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 40, top: 0, bottom: 0 }}>
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={80}
            tick={{ fill: '#888', fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(v: number) => [`${v}%`, 'Probability']}
            contentStyle={{ background: '#111', border: '1px solid #222', borderRadius: 6 }}
            labelStyle={{ color: '#fff' }}
            itemStyle={{ color: '#ccc' }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} label={{ position: 'right', fill: '#ccc', fontSize: 12, formatter: (v: number) => `${v}%` }}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
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
