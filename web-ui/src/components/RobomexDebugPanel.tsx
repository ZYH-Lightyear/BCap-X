import { useEffect, useMemo, useState } from 'react';
import type { ArtifactItem, RobomexEvent, RobomexEventsResponse, RobomexRun } from '../types/messages';

const DEFAULT_ROOT = 'outputs';

function formatTime(value?: string): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString();
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function statusClass(event: RobomexEvent): string {
  const text = `${event.status || ''} ${event.ok === false ? 'failed' : ''} ${event.success === false ? 'failed' : ''}`.toLowerCase();
  if (text.includes('fail') || text.includes('error') || text.includes('exception')) return 'border-red-500/50 bg-red-950/20 text-red-200';
  if (text.includes('uncertain')) return 'border-yellow-500/50 bg-yellow-950/20 text-yellow-100';
  if (text.includes('pass') || text.includes('success') || event.ok === true) return 'border-green-500/40 bg-green-950/20 text-green-100';
  if (event.agent_role === 'verifier') return 'border-blue-500/40 bg-blue-950/20 text-blue-100';
  if (event.agent_role === 'act') return 'border-accent/40 bg-accent/10 text-text-primary';
  return 'border-surface-border bg-surface-sunken text-text-secondary';
}

function eventTitle(event: RobomexEvent): string {
  const role = event.agent_role ? `${event.agent_role}` : 'system';
  const turn = event.turn !== undefined ? ` T${event.turn}` : '';
  return `${role}${turn} · ${event.event}`;
}

function isMedia(file: ArtifactItem, ext: string): boolean {
  return file.path.toLowerCase().endsWith(ext);
}

export function RobomexDebugPanel() {
  const [root, setRoot] = useState(DEFAULT_ROOT);
  const [runs, setRuns] = useState<RobomexRun[]>([]);
  const [runDir, setRunDir] = useState('');
  const [data, setData] = useState<RobomexEventsResponse | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [roleFilter, setRoleFilter] = useState('all');
  const [subgoalFilter, setSubgoalFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadRuns() {
      try {
        const resp = await fetch(`/api/robomex/runs?root=${encodeURIComponent(root)}`);
        if (!resp.ok) throw new Error(await resp.text());
        const json = await resp.json() as { runs: RobomexRun[] };
        if (cancelled) return;
        setRuns(json.runs);
        if (!runDir && json.runs[0]) setRunDir(json.runs[0].path);
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load runs');
      }
    }
    loadRuns();
    return () => { cancelled = true; };
  }, [root]);

  useEffect(() => {
    if (!runDir) return;
    let cancelled = false;
    async function loadEvents(showLoading = false) {
      if (showLoading) setLoading(true);
      try {
        const resp = await fetch(`/api/robomex/events?dir=${encodeURIComponent(runDir)}`);
        if (!resp.ok) throw new Error(await resp.text());
        const json = await resp.json() as RobomexEventsResponse;
        if (cancelled) return;
        setData(json);
        setSelectedEventId((prev) => {
          if (prev && json.events.some((event) => event.id === prev)) return prev;
          return json.events[0]?.id || null;
        });
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load events');
      } finally {
        if (!cancelled && showLoading) setLoading(false);
      }
    }
    loadEvents(true);
    const timer = window.setInterval(() => loadEvents(false), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [runDir]);

  const events = data?.events || [];
  const subgoals = useMemo(() => {
    const values = new Set<number>();
    for (const event of events) {
      if (typeof event.subgoal_number === 'number') values.add(event.subgoal_number);
    }
    return [...values].sort((a, b) => a - b);
  }, [events]);

  const filteredEvents = useMemo(() => {
    const q = query.trim().toLowerCase();
    return events.filter((event) => {
      if (roleFilter !== 'all' && event.agent_role !== roleFilter) return false;
      if (subgoalFilter !== 'all' && String(event.subgoal_number || '') !== subgoalFilter) return false;
      if (!q) return true;
      const blob = JSON.stringify(event).toLowerCase();
      return blob.includes(q);
    });
  }, [events, roleFilter, subgoalFilter, query]);

  const selectedEvent = useMemo(() => {
    return events.find((event) => event.id === selectedEventId) || filteredEvents[0] || null;
  }, [events, filteredEvents, selectedEventId]);

  const videos = (data?.files || []).filter((file) => isMedia(file, '.mp4'));
  const images = (data?.files || []).filter((file) => ['.png', '.jpg', '.jpeg'].some((ext) => isMedia(file, ext)));
  const textFiles = (data?.files || []).filter((file) => ['.json', '.jsonl', '.txt', '.py', '.log'].some((ext) => isMedia(file, ext)));

  return (
    <div className="h-full flex flex-col bg-surface text-text-primary">
      <header className="flex-shrink-0 border-b border-surface-border bg-surface-raised px-5 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="min-w-0">
            <div className="text-sm font-bold font-display tracking-wide uppercase">RoboMEx Debug</div>
            <div className="text-xs text-text-tertiary truncate">{data?.dir || 'Select an events.jsonl run'}</div>
          </div>
          <a href="/" className="px-3 py-2 rounded-md bg-surface-overlay text-xs text-text-secondary hover:text-accent transition-colors">
            CaP-X UI
          </a>
        </div>
      </header>

      <div className="flex-shrink-0 border-b border-surface-border bg-surface px-5 py-3 grid grid-cols-[minmax(160px,240px)_minmax(260px,1fr)_120px_120px_minmax(160px,260px)] gap-3">
        <input
          value={root}
          onChange={(e) => setRoot(e.target.value)}
          className="px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs font-mono focus:outline-none focus:border-accent/50"
          placeholder="scan root"
        />
        <select
          value={runDir}
          onChange={(e) => setRunDir(e.target.value)}
          className="px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs font-mono focus:outline-none focus:border-accent/50"
        >
          {runs.map((run) => (
            <option key={run.path} value={run.path}>
              {run.path}
            </option>
          ))}
        </select>
        <select
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs focus:outline-none focus:border-accent/50"
        >
          <option value="all">all agents</option>
          <option value="act">act</option>
          <option value="verifier">verifier</option>
          <option value="coder">coder</option>
        </select>
        <select
          value={subgoalFilter}
          onChange={(e) => setSubgoalFilter(e.target.value)}
          className="px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs focus:outline-none focus:border-accent/50"
        >
          <option value="all">all subgoals</option>
          {subgoals.map((value) => <option key={value} value={value}>subgoal {value}</option>)}
        </select>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs focus:outline-none focus:border-accent/50"
          placeholder="filter text"
        />
      </div>

      {error && (
        <div className="flex-shrink-0 px-5 py-2 border-b border-red-800/30 bg-red-950/20 text-xs text-red-300">
          {error}
        </div>
      )}

      <main className="flex-1 min-h-0 grid grid-cols-[360px_minmax(0,1fr)_320px]">
        <section className="min-h-0 border-r border-surface-border bg-surface-raised flex flex-col">
          <div className="flex-shrink-0 px-4 py-3 border-b border-surface-border flex items-center justify-between">
            <span className="text-xs font-display font-bold uppercase tracking-wide">Timeline</span>
            <span className="text-xs text-text-tertiary">{filteredEvents.length}/{events.length}</span>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2">
            {filteredEvents.map((event) => (
              <button
                key={event.id}
                onClick={() => setSelectedEventId(event.id || null)}
                className={`w-full text-left border rounded-md px-3 py-2 transition-colors ${statusClass(event)} ${
                  selectedEvent?.id === event.id ? 'ring-1 ring-accent/60' : 'hover:border-accent/50'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold truncate">{eventTitle(event)}</span>
                  <span className="text-[11px] opacity-70 flex-shrink-0">{formatTime(event.ts)}</span>
                </div>
                <div className="mt-1 text-xs opacity-80 line-clamp-2">{event.message || event.goal || event.question || event.raw_preview}</div>
              </button>
            ))}
            {filteredEvents.length === 0 && (
              <div className="h-40 border border-dashed border-surface-border rounded-md flex items-center justify-center text-xs text-text-tertiary">
                No events match the current filters
              </div>
            )}
          </div>
        </section>

        <section className="min-h-0 flex flex-col bg-surface">
          <div className="flex-shrink-0 px-5 py-3 border-b border-surface-border flex items-center justify-between">
            <div>
              <div className="text-sm font-display font-bold tracking-wide">{selectedEvent ? eventTitle(selectedEvent) : 'No event selected'}</div>
              <div className="text-xs text-text-tertiary">{loading ? 'refreshing events' : data?.summary?.task || 'events.jsonl'}</div>
            </div>
            {selectedEvent?.duration_s !== undefined && (
              <span className="px-2 py-1 rounded bg-surface-overlay text-xs text-text-secondary">{selectedEvent.duration_s}s</span>
            )}
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-5 space-y-4">
            {selectedEvent ? (
              <>
                <div className="grid grid-cols-4 gap-2">
                  {['event', 'agent_role', 'subgoal_number', 'turn'].map((key) => (
                    <div key={key} className="border border-surface-border bg-surface-sunken rounded-md px-3 py-2">
                      <div className="text-[11px] uppercase text-text-tertiary">{key}</div>
                      <div className="text-xs font-mono text-text-primary truncate">{String(selectedEvent[key] ?? '-')}</div>
                    </div>
                  ))}
                </div>

                {selectedEvent.code && (
                  <section className="border border-surface-border rounded-md overflow-hidden">
                    <div className="px-3 py-2 border-b border-surface-border bg-surface-raised text-xs font-display font-bold uppercase">Code</div>
                    <pre className="p-3 bg-black text-xs overflow-x-auto whitespace-pre-wrap font-mono">{selectedEvent.code}</pre>
                  </section>
                )}

                {(selectedEvent.stdout || selectedEvent.stderr || selectedEvent.feedback || selectedEvent.raw || selectedEvent.raw_preview) && (
                  <section className="grid grid-cols-2 gap-3">
                    <pre className="min-h-40 max-h-80 overflow-auto rounded-md border border-surface-border bg-surface-sunken p-3 text-xs whitespace-pre-wrap text-text-secondary">
                      {selectedEvent.stdout || selectedEvent.feedback || selectedEvent.raw_preview || '(no stdout/feedback)'}
                    </pre>
                    <pre className="min-h-40 max-h-80 overflow-auto rounded-md border border-surface-border bg-red-950/20 p-3 text-xs whitespace-pre-wrap text-red-200">
                      {selectedEvent.stderr || selectedEvent.raw || '(no stderr/raw)'}
                    </pre>
                  </section>
                )}

                <section className="border border-surface-border rounded-md overflow-hidden">
                  <div className="px-3 py-2 border-b border-surface-border bg-surface-raised text-xs font-display font-bold uppercase">Raw Event</div>
                  <pre className="p-3 bg-surface-sunken text-xs overflow-x-auto whitespace-pre-wrap font-mono text-text-secondary">
                    {JSON.stringify(selectedEvent, null, 2)}
                  </pre>
                </section>
              </>
            ) : (
              <div className="h-full border border-dashed border-surface-border rounded-md flex items-center justify-center text-sm text-text-tertiary">
                Load a RoboMEx run to inspect agent events
              </div>
            )}
          </div>
        </section>

        <aside className="min-h-0 border-l border-surface-border bg-surface-raised flex flex-col">
          <div className="flex-shrink-0 px-4 py-3 border-b border-surface-border">
            <div className="text-xs font-display font-bold uppercase tracking-wide">Artifacts</div>
            <div className="text-xs text-text-tertiary">{data?.files.length || 0} files</div>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-4">
            {videos[0] && (
              <div>
                <div className="text-xs text-text-tertiary mb-2">Latest video</div>
                <video src={videos[0].url} controls muted playsInline className="w-full aspect-video bg-black border border-surface-border rounded-md" />
              </div>
            )}
            {images[0] && (
              <div>
                <div className="text-xs text-text-tertiary mb-2">Latest image</div>
                <img src={images[0].url} className="w-full max-h-56 object-contain bg-black border border-surface-border rounded-md" />
              </div>
            )}
            <div className="space-y-1">
              {textFiles.slice(0, 80).map((file) => (
                <a
                  key={file.path}
                  href={file.url}
                  target="_blank"
                  rel="noreferrer"
                  className="block px-2 py-2 rounded-md border border-surface-border bg-surface-sunken hover:border-accent/50 transition-colors"
                >
                  <div className="text-xs font-mono text-text-primary truncate">{file.path}</div>
                  <div className="text-[11px] text-text-tertiary">{formatSize(file.size)}</div>
                </a>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
