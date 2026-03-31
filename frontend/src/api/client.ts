import type {
  FighterSearchResult,
  FighterProfile,
  FightHistoryResponse,
  PredictionResponse,
  UpcomingEventsResponse,
  HealthResponse,
} from '../types'

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export function searchFighters(q: string): Promise<FighterSearchResult[]> {
  return get<FighterSearchResult[]>(`/fighters/search?q=${encodeURIComponent(q)}`)
}

export function getFighterProfile(name: string): Promise<FighterProfile> {
  return get<FighterProfile>(`/fighters/${encodeURIComponent(name)}`)
}

export function getFighterHistory(name: string, page = 1, pageSize = 5): Promise<FightHistoryResponse> {
  return get<FightHistoryResponse>(
    `/fighters/${encodeURIComponent(name)}/history?page=${page}&page_size=${pageSize}`,
  )
}

export function predict(fighter_a: string, fighter_b: string): Promise<PredictionResponse> {
  return post<PredictionResponse>('/predict', { fighter_a, fighter_b })
}

export function getUpcomingEvents(): Promise<UpcomingEventsResponse> {
  return get<UpcomingEventsResponse>('/events/upcoming')
}

export function getHealth(): Promise<HealthResponse> {
  return get<HealthResponse>('/health')
}

export function formatAmericanOdds(odds: number): string {
  return odds >= 0 ? `+${Math.round(odds)}` : `${Math.round(odds)}`
}
