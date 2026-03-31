import { NavLink, Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getHealth } from '../api/client'

export default function Layout() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 60_000,
  })

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-ufc-border bg-ufc-card sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <span className="font-extrabold text-lg tracking-tight">
              <span className="text-ufc-gold">UFC</span> PREDICTOR
            </span>
            <nav className="flex gap-1">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                    isActive ? 'bg-ufc-gold/10 text-ufc-gold' : 'text-ufc-muted hover:text-white'
                  }`
                }
              >
                Matchup
              </NavLink>
              <NavLink
                to="/events"
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                    isActive ? 'bg-ufc-gold/10 text-ufc-gold' : 'text-ufc-muted hover:text-white'
                  }`
                }
              >
                Upcoming Card
              </NavLink>
            </nav>
          </div>
          <div className="flex items-center gap-2 text-xs text-ufc-muted">
            <span
              className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`}
            />
            {health ? (
              <span>
                Last fight data:{' '}
                {health.last_scrape
                  ? new Date(health.last_scrape).toLocaleDateString()
                  : 'N/A'}
              </span>
            ) : (
              <span>Connecting…</span>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
        <Outlet />
      </main>

      <footer className="border-t border-ufc-border py-4 text-center text-xs text-ufc-muted">
        ML predictions are probabilistic estimates, not guaranteed outcomes.
      </footer>
    </div>
  )
}
