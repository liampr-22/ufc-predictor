// Mirrors Pydantic response models from models/pydantic_models.py

export interface HealthResponse {
  status: string
  timestamp: string
  last_scrape: string | null
  last_successful_scrape: string | null
}

export interface FighterSearchResult {
  id: number
  name: string
  weight_class: string | null
  elo_rating: number | null
}

export interface CareerAverages {
  fights: number
  wins: number
  losses: number
  draws: number
  avg_sig_strikes_landed: number
  avg_takedowns_landed: number
  avg_control_time_seconds: number
  ko_rate: number
  sub_rate: number
  dec_rate: number
}

export interface StyleScores {
  striker: number
  wrestler: number
  grappler: number
  brawler: number
}

export interface FighterProfile {
  id: number
  name: string
  height: number | null
  reach: number | null
  stance: string | null
  dob: string | null
  weight_class: string | null
  elo_rating: number | null
  career: CareerAverages
  style: StyleScores
}

export interface FightHistoryEntry {
  fight_id: number
  date: string
  event: string | null
  opponent: string
  result: 'Win' | 'Loss' | 'Draw' | 'NC'
  method: string | null
  round: number | null
  time: string | null
}

export interface FightHistoryResponse {
  fighter: string
  page: number
  page_size: number
  total: number
  fights: FightHistoryEntry[]
}

export interface MethodProbs {
  ko_tko: number
  submission: number
  decision: number
}

export interface FighterOdds {
  american: number
  decimal: number
}

export interface KeyDifferentials {
  elo_delta: number
  reach_delta: number | null
  height_delta: number | null
  age_delta: number | null
}

export interface FighterPrediction {
  name: string
  win_prob: number
  odds: FighterOdds
}

export interface PredictionResponse {
  fighter_a: FighterPrediction
  fighter_b: FighterPrediction
  method_probs: MethodProbs
  key_differentials: KeyDifferentials
}

export interface ScheduledFight {
  fighter_a: string
  fighter_b: string
  prediction: PredictionResponse | null
}

export interface UpcomingEvent {
  event_name: string
  date: string
  fights: ScheduledFight[]
}

export interface UpcomingEventsResponse {
  events: UpcomingEvent[]
}
