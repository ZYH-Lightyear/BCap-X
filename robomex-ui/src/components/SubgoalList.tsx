import type { SubgoalIndex } from '../types'
import { statusTone } from '../status'

type Props = {
  subgoals: SubgoalIndex[]
  selected: number
  onSelect: (index: number) => void
}

export function SubgoalList({ subgoals, selected, onSelect }: Props) {
  return (
    <div className="space-y-2">
      {subgoals.map((sg) => (
        <button
          key={sg.index}
          onClick={() => onSelect(sg.index)}
          className={`w-full rounded border p-2 text-left ${statusTone(sg.authoring_status || (sg.success ? 'succeeded' : ''))} ${
            selected === sg.index ? 'ring-2 ring-ink-700' : ''
          }`}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold">Subgoal {sg.index}</span>
            <span className="text-[10px]">{sg.authoring_status || (sg.has_swarm ? 'swarm' : '—')}</span>
          </div>
          <div className="mt-1 line-clamp-2 text-xs leading-snug">{sg.goal || '(no goal)'}</div>
          {sg.note && <div className="mt-1 line-clamp-2 text-[11px] opacity-80">{sg.note}</div>}
        </button>
      ))}
      {subgoals.length === 0 && (
        <div className="rounded border border-dashed border-ink-200 p-3 text-xs text-ink-500">
          Select a run to load subgoals.
        </div>
      )}
    </div>
  )
}
