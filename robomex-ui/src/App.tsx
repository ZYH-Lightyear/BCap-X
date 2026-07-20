import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  fetchAgent,
  fetchLlm,
  fetchRun,
  fetchSwarm,
  fetchTurn,
  listRuns,
  liveUrl,
} from './api'
import { EventStream } from './components/EventStream'
import { LlmDrawer } from './components/LlmDrawer'
import { MediaPanel } from './components/MediaPanel'
import { RunPicker } from './components/RunPicker'
import { SubgoalList } from './components/SubgoalList'
import { SwarmGraph } from './components/SwarmGraph'
import { TurnWorkbench } from './components/TurnWorkbench'
import type {
  AgentDetail,
  LiveEvent,
  RunDetail,
  RunSummary,
  SwarmView,
  TurnDetail,
} from './types'

const DEFAULT_ROOT = 'outputs/robomex_planner_live'

function readQuery() {
  const params = new URLSearchParams(window.location.search)
  return {
    root: params.get('root') || DEFAULT_ROOT,
    dir: params.get('dir') || '',
    subgoal: Number(params.get('subgoal') || '0'),
    agent: params.get('agent') || '',
    turn: params.get('turn') != null && params.get('turn') !== '' ? Number(params.get('turn')) : null,
  }
}

function writeQuery(state: {
  root: string
  dir: string
  subgoal: number
  agent: string
  turn: number | null
}) {
  const params = new URLSearchParams()
  params.set('root', state.root)
  if (state.dir) params.set('dir', state.dir)
  params.set('subgoal', String(state.subgoal))
  if (state.agent) params.set('agent', state.agent)
  if (state.turn != null) params.set('turn', String(state.turn))
  const next = `${window.location.pathname}?${params.toString()}`
  window.history.replaceState(null, '', next)
}

export default function App() {
  const initial = useMemo(() => readQuery(), [])
  const [root, setRoot] = useState(initial.root)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runDir, setRunDir] = useState(initial.dir)
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null)
  const [subgoal, setSubgoal] = useState(initial.subgoal)
  const [swarm, setSwarm] = useState<SwarmView | null>(null)
  const [agentDir, setAgentDir] = useState(initial.agent)
  const [agent, setAgent] = useState<AgentDetail | null>(null)
  const [turn, setTurn] = useState<number | null>(initial.turn)
  const [turnDetail, setTurnDetail] = useState<TurnDetail | null>(null)
  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([])
  const [liveConnected, setLiveConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [llmOpen, setLlmOpen] = useState(false)
  const [llmPath, setLlmPath] = useState('')
  const [llmContent, setLlmContent] = useState<unknown>(null)
  const [llmLoading, setLlmLoading] = useState(false)
  const [llmError, setLlmError] = useState<string | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)

  const refreshRuns = useCallback(async () => {
    try {
      const data = await listRuns(root)
      setRuns(data.runs)
      if (!runDir && data.runs[0]) {
        setRunDir(data.runs[0].path)
      }
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [root, runDir])

  useEffect(() => {
    refreshRuns()
  }, [refreshRuns])

  useEffect(() => {
    writeQuery({ root, dir: runDir, subgoal, agent: agentDir, turn })
  }, [root, runDir, subgoal, agentDir, turn])

  useEffect(() => {
    if (!runDir) return
    let cancelled = false
    ;(async () => {
      try {
        const detail = await fetchRun(runDir)
        if (cancelled) return
        setRunDetail(detail)
        if (detail.subgoals.length && !detail.subgoals.some((sg) => sg.index === subgoal)) {
          setSubgoal(detail.subgoals[0].index)
        }
        setError(null)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runDir, refreshToken])

  useEffect(() => {
    if (!runDir) return
    let cancelled = false
    ;(async () => {
      try {
        const view = await fetchSwarm(runDir, subgoal)
        if (cancelled) return
        setSwarm(view)
        if (agentDir && !view.agents.some((a) => a.dir === agentDir)) {
          setAgentDir(view.agents[0]?.dir || '')
        } else if (!agentDir && view.agents[0]) {
          setAgentDir(view.agents[0].dir)
        }
        setError(null)
      } catch (err) {
        if (!cancelled) {
          setSwarm(null)
          setError(err instanceof Error ? err.message : String(err))
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runDir, subgoal, refreshToken])

  useEffect(() => {
    if (!runDir || !agentDir) {
      setAgent(null)
      setTurnDetail(null)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const detail = await fetchAgent(runDir, subgoal, agentDir)
        if (cancelled) return
        setAgent(detail)
        const nextTurn =
          turn != null && detail.turns.some((t) => t.turn === turn)
            ? turn
            : detail.turns[0]?.turn ?? null
        setTurn(nextTurn)
        setError(null)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runDir, subgoal, agentDir, refreshToken])

  useEffect(() => {
    if (!runDir || !agentDir || turn == null) {
      setTurnDetail(null)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const detail = await fetchTurn(runDir, subgoal, agentDir, turn)
        if (!cancelled) setTurnDetail(detail)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runDir, subgoal, agentDir, turn, refreshToken])

  useEffect(() => {
    if (!runDir) {
      setLiveConnected(false)
      setLiveEvents([])
      return
    }
    const source = new EventSource(liveUrl(runDir, false))
    setLiveEvents([])
    source.onopen = () => setLiveConnected(true)
    source.onerror = () => setLiveConnected(false)
    source.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as LiveEvent
        setLiveEvents((prev) => [...prev.slice(-199), event])
        const name = event.event || ''
        if (
          name.startsWith('subgoal_') ||
          name.startsWith('agent_') ||
          name.startsWith('code_') ||
          name.includes('swarm') ||
          name.includes('graph')
        ) {
          setRefreshToken((n) => n + 1)
        }
      } catch {
        // ignore malformed live frames
      }
    }
    return () => {
      source.close()
      setLiveConnected(false)
    }
  }, [runDir])

  const openLlm = async (path: string) => {
    if (!runDir) return
    setLlmOpen(true)
    setLlmPath(path)
    setLlmLoading(true)
    setLlmError(null)
    try {
      const data = await fetchLlm(runDir, path, 'text')
      setLlmContent(data.content)
    } catch (err) {
      setLlmError(err instanceof Error ? err.message : String(err))
    } finally {
      setLlmLoading(false)
    }
  }

  const media = agent?.media?.length ? agent.media : swarm?.media || []

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex flex-shrink-0 items-center justify-between border-b border-ink-200 bg-white px-4 py-3">
        <div>
          <div className="text-sm font-bold tracking-wide">RoboMEx Trace</div>
          <div className="truncate text-xs text-ink-500">
            {runDetail?.summary?.task ? String(runDetail.summary.task) : 'offline / live run explorer'}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`rounded border px-2 py-1 ${liveConnected ? 'border-blue-300 bg-blue-50 text-blue-800' : 'border-ink-200 bg-ink-50 text-ink-600'}`}>
            {liveConnected ? 'live' : 'idle'}
          </span>
          <span className="rounded border border-ink-200 bg-ink-50 px-2 py-1 font-mono">
            :8300 API · :5174 UI
          </span>
        </div>
      </header>

      {error && (
        <div className="flex-shrink-0 border-b border-red-200 bg-red-50 px-4 py-2 text-xs text-red-800">
          {error}
        </div>
      )}

      <main className="grid min-h-0 flex-1 grid-cols-[280px_minmax(0,1fr)_320px]">
        <aside className="min-h-0 space-y-4 overflow-y-auto border-r border-ink-200 bg-white p-3">
          <section>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-700">Runs</div>
            <RunPicker
              root={root}
              runs={runs}
              selected={runDir}
              onRootChange={setRoot}
              onSelect={(path) => {
                setRunDir(path)
                setAgentDir('')
                setTurn(null)
                setSubgoal(0)
              }}
              onRefresh={refreshRuns}
            />
          </section>
          <section>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-700">Subgoals</div>
            <SubgoalList
              subgoals={runDetail?.subgoals || []}
              selected={subgoal}
              onSelect={(index) => {
                setSubgoal(index)
                setAgentDir('')
                setTurn(null)
              }}
            />
          </section>
          {swarm?.manager_turns?.length ? (
            <section>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-700">
                Manager turns
              </div>
              <div className="space-y-1">
                {swarm.manager_turns.map((mt) => (
                  <button
                    key={mt.turn}
                    onClick={() => mt.response_path && openLlm(mt.response_path)}
                    className="w-full rounded border border-ink-200 bg-ink-50 px-2 py-1.5 text-left text-[11px] hover:border-ink-400"
                  >
                    <div className="font-mono">manager t{mt.turn}</div>
                    <div className="line-clamp-2 text-ink-500">{mt.response_preview}</div>
                  </button>
                ))}
              </div>
            </section>
          ) : null}
        </aside>

        <section className="grid min-h-0 grid-rows-[minmax(220px,0.9fr)_minmax(280px,1.2fr)] gap-3 overflow-hidden bg-ink-50 p-3">
          <div className="grid min-h-0 grid-cols-[minmax(0,1.4fr)_280px] gap-3">
            <div className="min-h-0 overflow-y-auto rounded border border-ink-200 bg-white p-3">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-700">
                Swarm Graph
              </div>
              {swarm ? (
                <SwarmGraph
                  nodes={swarm.nodes}
                  edges={swarm.edges}
                  agents={swarm.agents}
                  selectedAgent={agentDir}
                  onSelectAgent={(dir) => {
                    setAgentDir(dir)
                    setTurn(null)
                  }}
                  entry={swarm.entry}
                  successNode={swarm.success_node}
                  outcomeStatus={swarm.outcome_status}
                  note={swarm.note}
                />
              ) : (
                <div className="text-sm text-ink-500">No swarm projection for this subgoal.</div>
              )}
            </div>
            <EventStream events={liveEvents} connected={liveConnected} />
          </div>
          <div className="min-h-0 overflow-hidden">
            <TurnWorkbench
              agent={agent}
              turns={agent?.turns || []}
              selectedTurn={turn}
              turnDetail={turnDetail}
              onSelectTurn={setTurn}
              onOpenLlm={openLlm}
            />
          </div>
        </section>

        <aside className="min-h-0 border-l border-ink-200 bg-ink-50 p-3">
          <MediaPanel media={media} title={agent ? `Media · ${agent.agent_dir}` : 'Media'} />
        </aside>
      </main>

      <LlmDrawer
        open={llmOpen}
        path={llmPath}
        content={llmContent}
        loading={llmLoading}
        error={llmError}
        onClose={() => setLlmOpen(false)}
      />
    </div>
  )
}
