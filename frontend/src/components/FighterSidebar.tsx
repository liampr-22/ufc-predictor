import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import { getFighterProfile, getFighterHistory } from '../api/client'
import type { FightHistoryEntry } from '../types'

interface Props {
  name: string
  onClose: () => void
}

function ResultBadge({ result }: { result: FightHistoryEntry['result'] }) {
  const colours: Record<string, string> = {
    Win: 'bg-green-500/20 text-green-400',
    Loss: 'bg-red-500/20 text-red-400',
    Draw: 'bg-yellow-500/20 text-yellow-400',
    NC: 'bg-white/10 text-white/60',
  }
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded ${colours[result] ?? ''}`}>
      {result}
    </span>
  )
}

export default function FighterSidebar({ name, onClose }: Props) {
  const { data: profile, isLoading: profileLoading, error: profileError } = useQuery({
    queryKey: ['fighter', name],
    queryFn: () => getFighterProfile(name),
    enabled: !!name,
  })

  const { data: history } = useQuery({
    queryKey: ['history', name],
    queryFn: () => getFighterHistory(name, 1, 5),
    enabled: !!name,
  })

  // Close on Escape
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  // Lock body scroll
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  const radarData = profile
    ? [
        { axis: 'Striker', value: Math.round(profile.style.striker * 100) },
        { axis: 'Wrestler', value: Math.round(profile.style.wrestler * 100) },
        { axis: 'Grappler', value: Math.round(profile.style.grappler * 100) },
        { axis: 'Brawler', value: Math.round(profile.style.brawler * 100) },
      ]
    : []

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed right-0 top-0 h-full w-full max-w-sm bg-ufc-card border-l border-ufc-border z-50 overflow-y-auto flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-ufc-border sticky top-0 bg-ufc-card">
          <h2 className="font-bold text-lg truncate pr-4">{name}</h2>
          <button
            onClick={onClose}
            className="text-ufc-muted hover:text-white transition-colors text-xl leading-none"
          >
            ×
          </button>
        </div>

        <div className="p-5 space-y-6 flex-1">
          {profileLoading && (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 bg-white/10 rounded w-3/4" />
              <div className="h-4 bg-white/10 rounded w-1/2" />
              <div className="h-32 bg-white/10 rounded" />
            </div>
          )}

          {profileError && (
            <p className="text-red-400 text-sm">Failed to load fighter profile.</p>
          )}

          {profile && (
            <>
              {/* Meta */}
              <div className="grid grid-cols-2 gap-2 text-sm">
                {profile.weight_class && (
                  <div>
                    <span className="text-ufc-muted text-xs">Division</span>
                    <p className="font-medium">{profile.weight_class}</p>
                  </div>
                )}
                {profile.elo_rating !== null && (
                  <div>
                    <span className="text-ufc-muted text-xs">Elo Rating</span>
                    <p className="font-medium text-ufc-gold">{Math.round(profile.elo_rating)}</p>
                  </div>
                )}
                {profile.reach !== null && (
                  <div>
                    <span className="text-ufc-muted text-xs">Reach</span>
                    <p className="font-medium">{profile.reach}"</p>
                  </div>
                )}
                {profile.stance && (
                  <div>
                    <span className="text-ufc-muted text-xs">Stance</span>
                    <p className="font-medium">{profile.stance}</p>
                  </div>
                )}
              </div>

              {/* Style radar */}
              <div>
                <h3 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-2">
                  Fight Style
                </h3>
                <ResponsiveContainer width="100%" height={180}>
                  <RadarChart data={radarData} margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
                    <PolarGrid stroke="#333" />
                    <PolarAngleAxis
                      dataKey="axis"
                      tick={{ fill: '#888', fontSize: 11 }}
                    />
                    <Tooltip
                      formatter={(v: number) => [`${v}`, 'Score']}
                      contentStyle={{ background: '#111', border: '1px solid #222', borderRadius: 6 }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Radar
                      dataKey="value"
                      stroke="#D4AC0D"
                      fill="#D4AC0D"
                      fillOpacity={0.2}
                      dot={{ r: 3, fill: '#D4AC0D' }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Career stats */}
              <div>
                <h3 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-2">
                  Career Stats
                </h3>
                <div className="grid grid-cols-3 gap-2 text-center">
                  {[
                    { label: 'W', value: profile.career.wins, colour: 'text-green-400' },
                    { label: 'L', value: profile.career.losses, colour: 'text-red-400' },
                    { label: 'D', value: profile.career.draws, colour: 'text-yellow-400' },
                  ].map((s) => (
                    <div key={s.label} className="bg-white/5 rounded p-2">
                      <p className={`text-xl font-bold ${s.colour}`}>{s.value}</p>
                      <p className="text-xs text-ufc-muted">{s.label}</p>
                    </div>
                  ))}
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                  {[
                    { label: 'KO%', value: `${Math.round(profile.career.ko_rate * 100)}%` },
                    { label: 'Sub%', value: `${Math.round(profile.career.sub_rate * 100)}%` },
                    { label: 'Dec%', value: `${Math.round(profile.career.dec_rate * 100)}%` },
                  ].map((s) => (
                    <div key={s.label} className="bg-white/5 rounded p-2">
                      <p className="text-base font-semibold">{s.value}</p>
                      <p className="text-xs text-ufc-muted">{s.label}</p>
                    </div>
                  ))}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-center">
                  <div className="bg-white/5 rounded p-2">
                    <p className="text-base font-semibold">{profile.career.avg_sig_strikes_landed.toFixed(1)}</p>
                    <p className="text-xs text-ufc-muted">Avg Sig. Strikes</p>
                  </div>
                  <div className="bg-white/5 rounded p-2">
                    <p className="text-base font-semibold">{profile.career.avg_takedowns_landed.toFixed(1)}</p>
                    <p className="text-xs text-ufc-muted">Avg TDs</p>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Fight history */}
          {history && history.fights.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-2">
                Last {history.fights.length} Fights
              </h3>
              <div className="space-y-2">
                {history.fights.map((f: FightHistoryEntry) => (
                  <div
                    key={f.fight_id}
                    className="flex items-center gap-3 bg-white/5 rounded p-2.5 text-sm"
                  >
                    <ResultBadge result={f.result} />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{f.opponent}</p>
                      {f.method && (
                        <p className="text-xs text-ufc-muted">
                          {f.method}{f.round ? ` · R${f.round}` : ''}
                        </p>
                      )}
                    </div>
                    <span className="text-xs text-ufc-muted shrink-0">
                      {new Date(f.date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
