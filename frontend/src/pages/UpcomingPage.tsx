import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getUpcomingEvents, formatAmericanOdds } from '../api/client'
import FighterSidebar from '../components/FighterSidebar'
import type { ScheduledFight, UpcomingEvent } from '../types'

function MethodPill({ label, value, colour }: { label: string; value: number; colour: string }) {
  return (
    <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded font-medium ${colour}`}>
      {label} {Math.round(value * 100)}%
    </span>
  )
}

function FightRow({
  fight,
  onFighterClick,
}: {
  fight: ScheduledFight
  onFighterClick: (name: string) => void
}) {
  const { fighter_a, fighter_b, prediction } = fight
  const favA =
    prediction && prediction.fighter_a.win_prob >= prediction.fighter_b.win_prob

  return (
    <div className="card p-4 space-y-3">
      {/* Fighter matchup */}
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <button
            onClick={() => onFighterClick(fighter_a)}
            className={`font-bold text-base hover:text-ufc-gold transition-colors truncate block w-full text-left ${
              favA ? 'text-white' : 'text-white/70'
            }`}
          >
            {fighter_a}
          </button>
          {prediction && (
            <div className="flex items-center gap-2 mt-0.5">
              <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-ufc-gold rounded-full"
                  style={{ width: `${Math.round(prediction.fighter_a.win_prob * 100)}%` }}
                />
              </div>
              <span className="text-xs font-mono text-ufc-muted shrink-0">
                {Math.round(prediction.fighter_a.win_prob * 100)}%
              </span>
              <span className={`text-xs font-mono shrink-0 ${favA ? 'text-ufc-gold' : 'text-ufc-muted'}`}>
                {formatAmericanOdds(prediction.fighter_a.odds.american)}
              </span>
            </div>
          )}
        </div>

        <span className="text-ufc-muted text-sm font-medium shrink-0">vs</span>

        <div className="flex-1 min-w-0 text-right">
          <button
            onClick={() => onFighterClick(fighter_b)}
            className={`font-bold text-base hover:text-ufc-gold transition-colors truncate block w-full text-right ${
              !favA ? 'text-white' : 'text-white/70'
            }`}
          >
            {fighter_b}
          </button>
          {prediction && (
            <div className="flex items-center gap-2 mt-0.5 flex-row-reverse">
              <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white/40 rounded-full ml-auto"
                  style={{ width: `${Math.round(prediction.fighter_b.win_prob * 100)}%` }}
                />
              </div>
              <span className="text-xs font-mono text-ufc-muted shrink-0">
                {Math.round(prediction.fighter_b.win_prob * 100)}%
              </span>
              <span className={`text-xs font-mono shrink-0 ${!favA ? 'text-ufc-gold' : 'text-ufc-muted'}`}>
                {formatAmericanOdds(prediction.fighter_b.odds.american)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Method pills */}
      {prediction && (
        <div className="flex flex-wrap gap-1.5">
          <MethodPill
            label="KO/TKO"
            value={prediction.method_probs.ko_tko}
            colour="bg-red-500/15 text-red-400"
          />
          <MethodPill
            label="Sub"
            value={prediction.method_probs.submission}
            colour="bg-purple-500/15 text-purple-400"
          />
          <MethodPill
            label="Dec"
            value={prediction.method_probs.decision}
            colour="bg-blue-500/15 text-blue-400"
          />
        </div>
      )}

      {!prediction && (
        <p className="text-xs text-ufc-muted italic">Prediction unavailable</p>
      )}
    </div>
  )
}

function EventSection({
  event,
  onFighterClick,
}: {
  event: UpcomingEvent
  onFighterClick: (name: string) => void
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-baseline gap-3">
        <h2 className="font-bold text-lg">{event.event_name}</h2>
        <span className="text-sm text-ufc-muted">
          {new Date(event.date).toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            year: 'numeric',
          })}
        </span>
      </div>
      <div className="space-y-2">
        {event.fights.map((fight, i) => (
          <FightRow key={i} fight={fight} onFighterClick={onFighterClick} />
        ))}
      </div>
    </section>
  )
}

function Skeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {[1, 2].map((i) => (
        <div key={i} className="space-y-3">
          <div className="h-5 bg-white/10 rounded w-64" />
          {[1, 2, 3].map((j) => (
            <div key={j} className="card p-4 h-20" />
          ))}
        </div>
      ))}
    </div>
  )
}

export default function UpcomingPage() {
  const [sidebarFighter, setSidebarFighter] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['upcoming'],
    queryFn: getUpcomingEvents,
  })

  return (
    <div className="space-y-8">
      <div className="pt-4 space-y-1">
        <h1 className="text-3xl font-extrabold tracking-tight">Upcoming Card</h1>
        <p className="text-ufc-muted text-sm">Next UFC events with ML-generated predictions and implied odds</p>
      </div>

      {isLoading && <Skeleton />}

      {error && (
        <div className="card p-6 text-center space-y-2">
          <p className="text-red-400 font-medium">Failed to load upcoming events</p>
          <p className="text-ufc-muted text-sm">{(error as Error).message}</p>
        </div>
      )}

      {data && data.events.length === 0 && (
        <div className="text-center py-16 text-ufc-muted space-y-2">
          <p className="text-4xl">📅</p>
          <p>No upcoming events found in the database.</p>
          <p className="text-xs">Run an incremental scrape to fetch the latest UFC schedule.</p>
        </div>
      )}

      {data && data.events.length > 0 && (
        <div className="space-y-10">
          {data.events.map((event) => (
            <EventSection
              key={event.event_name + event.date}
              event={event}
              onFighterClick={setSidebarFighter}
            />
          ))}
        </div>
      )}

      {sidebarFighter && (
        <FighterSidebar name={sidebarFighter} onClose={() => setSidebarFighter(null)} />
      )}
    </div>
  )
}
