import type { LiveEvent } from '../types'

type Props = {
  events: LiveEvent[]
  connected: boolean
}

export function EventStream({ events, connected }: Props) {
  return (
    <div className="flex h-full min-h-0 flex-col rounded border border-ink-200 bg-white">
      <div className="flex items-center justify-between border-b border-ink-200 px-3 py-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink-700">Live Events</div>
        <div className={`text-[11px] ${connected ? 'text-signal-live' : 'text-ink-500'}`}>
          {connected ? 'SSE connected' : 'disconnected'}
        </div>
      </div>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto p-2 font-mono text-[11px]">
        {events.map((event, index) => (
          <div key={`${event.id || index}-${event.ts || index}`} className="rounded border border-ink-100 bg-ink-50 px-2 py-1">
            <div className="flex gap-2 text-ink-500">
              <span>{event.ts ? new Date(event.ts).toLocaleTimeString() : '--:--:--'}</span>
              <span className="font-semibold text-ink-800">{event.event}</span>
              {typeof event.subgoal_index === 'number' && <span>sg{event.subgoal_index}</span>}
              {typeof event.turn === 'number' && <span>t{event.turn}</span>}
            </div>
            <div className="truncate text-ink-700">
              {event.agent_label || event.node_id || event.action || event.message || ''}
            </div>
          </div>
        ))}
        {events.length === 0 && (
          <div className="p-3 text-ink-500">Waiting for events…</div>
        )}
      </div>
    </div>
  )
}
