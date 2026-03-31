import { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { searchFighters } from '../api/client'
import type { FighterSearchResult } from '../types'

interface Props {
  label: string
  value: string
  onChange: (name: string) => void
  onProfileClick?: (name: string) => void
  excludeName?: string
}

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return debounced
}

export default function FighterSearch({ label, value, onChange, onProfileClick, excludeName }: Props) {
  const [inputValue, setInputValue] = useState(value)
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(-1)
  const containerRef = useRef<HTMLDivElement>(null)
  const debouncedQ = useDebounce(inputValue, 300)

  const { data: results = [] } = useQuery({
    queryKey: ['search', debouncedQ],
    queryFn: () => searchFighters(debouncedQ),
    enabled: debouncedQ.length >= 2,
    staleTime: 1000 * 30,
  })

  const filtered = results.filter(
    (r: FighterSearchResult) => r.name.toLowerCase() !== (excludeName ?? '').toLowerCase(),
  )

  const select = useCallback(
    (name: string) => {
      setInputValue(name)
      onChange(name)
      setOpen(false)
      setActiveIdx(-1)
    },
    [onChange],
  )

  // Close on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Keep input in sync if parent resets value
  useEffect(() => {
    if (value === '') setInputValue('')
  }, [value])

  function handleKey(e: React.KeyboardEvent) {
    if (!open || filtered.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIdx((i) => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && activeIdx >= 0) {
      e.preventDefault()
      select(filtered[activeIdx].name)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  const isConfirmed = value !== '' && value === inputValue

  return (
    <div ref={containerRef} className="relative flex-1 min-w-0">
      <label className="block text-xs font-semibold text-ufc-muted uppercase tracking-widest mb-2">
        {label}
      </label>
      <div className="relative">
        <input
          type="text"
          className={`input-dark pr-8 ${isConfirmed ? 'border-ufc-gold/60' : ''}`}
          placeholder="Search fighter…"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value)
            setOpen(true)
            setActiveIdx(-1)
            if (e.target.value === '') onChange('')
          }}
          onFocus={() => inputValue.length >= 2 && setOpen(true)}
          onKeyDown={handleKey}
          autoComplete="off"
        />
        {isConfirmed && onProfileClick && (
          <button
            type="button"
            onClick={() => onProfileClick(value)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-ufc-muted hover:text-ufc-gold transition-colors text-xs"
            title="View profile"
          >
            ⓘ
          </button>
        )}
      </div>

      {open && filtered.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 bg-[#1a1a1a] border border-ufc-border rounded-lg shadow-xl max-h-60 overflow-y-auto">
          {filtered.map((f: FighterSearchResult, idx: number) => (
            <li
              key={f.id}
              className={`flex items-center justify-between px-4 py-2.5 cursor-pointer transition-colors ${
                idx === activeIdx ? 'bg-ufc-gold/10 text-white' : 'hover:bg-white/5 text-white'
              }`}
              onMouseDown={() => select(f.name)}
              onMouseEnter={() => setActiveIdx(idx)}
            >
              <span className="font-medium">{f.name}</span>
              <span className="text-xs text-ufc-muted ml-2 shrink-0">{f.weight_class ?? ''}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
