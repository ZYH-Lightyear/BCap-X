import type { RunSummary } from '../types'
import { shortName } from '../status'

type Props = {
  root: string
  runs: RunSummary[]
  selected: string
  onRootChange: (root: string) => void
  onSelect: (path: string) => void
  onRefresh: () => void
}

export function RunPicker({ root, runs, selected, onRootChange, onSelect, onRefresh }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          value={root}
          onChange={(e) => onRootChange(e.target.value)}
          className="flex-1 rounded border border-ink-200 bg-white px-2 py-1.5 font-mono text-xs"
          placeholder="outputs/robomex_planner_live"
        />
        <button
          onClick={onRefresh}
          className="rounded border border-ink-300 bg-white px-2 py-1.5 text-xs hover:bg-ink-50"
        >
          Refresh
        </button>
      </div>
      <div className="max-h-48 space-y-1 overflow-y-auto">
        {runs.map((run) => (
          <button
            key={run.path}
            onClick={() => onSelect(run.path)}
            className={`w-full rounded border px-2 py-2 text-left ${
              selected === run.path
                ? 'border-ink-700 bg-ink-900 text-white'
                : 'border-ink-200 bg-white hover:border-ink-400'
            }`}
          >
            <div className="truncate font-mono text-[11px]">{run.name}</div>
            <div className={`mt-1 truncate text-[11px] ${selected === run.path ? 'text-ink-200' : 'text-ink-500'}`}>
              {run.task || shortName(run.path)}
            </div>
            <div className={`mt-1 flex gap-2 text-[10px] ${selected === run.path ? 'text-ink-300' : 'text-ink-500'}`}>
              <span>sg={run.n_subgoals}</span>
              <span>{run.planner_status || '—'}</span>
              <span>env={String(run.env_success)}</span>
            </div>
          </button>
        ))}
        {runs.length === 0 && (
          <div className="rounded border border-dashed border-ink-200 p-3 text-xs text-ink-500">
            No runs with events.jsonl under this root.
          </div>
        )}
      </div>
    </div>
  )
}
