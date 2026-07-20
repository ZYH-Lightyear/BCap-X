type Props = {
  open: boolean
  path: string
  content: unknown
  loading: boolean
  error: string | null
  onClose: () => void
}

export function LlmDrawer({ open, path, content, loading, error, onClose }: Props) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-ink-900/30">
      <button className="flex-1 cursor-default" onClick={onClose} aria-label="Close drawer" />
      <aside className="flex h-full w-full max-w-2xl flex-col border-l border-ink-300 bg-white shadow-xl">
        <div className="flex items-start justify-between gap-3 border-b border-ink-200 px-4 py-3">
          <div className="min-w-0">
            <div className="text-xs font-semibold uppercase tracking-wide text-ink-700">LLM I/O</div>
            <div className="truncate font-mono text-[11px] text-ink-500">{path}</div>
          </div>
          <button onClick={onClose} className="rounded border border-ink-300 px-2 py-1 text-xs hover:bg-ink-50">
            Close
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-4">
          {loading && <div className="text-sm text-ink-500">Loading…</div>}
          {error && <div className="text-sm text-signal-fail">{error}</div>}
          {!loading && !error && (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs text-ink-800">
              {typeof content === 'string' ? content : JSON.stringify(content, null, 2)}
            </pre>
          )}
        </div>
      </aside>
    </div>
  )
}
