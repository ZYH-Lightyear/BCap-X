import { useCallback, useEffect, useMemo, useState } from 'react'
import { artifactUrl, fetchAgent, fetchSnapshot, listRuns, streamUrl } from './api'
import { AgentInspector } from './components/AgentInspector'
import { CandidateArena } from './components/CandidateArena'
import { SwarmGraph } from './components/SwarmGraph'
import type {
  ActivityEvent,
  AgentDetail,
  IntentView,
  RunSummary,
  Snapshot,
  SwarmConfigView,
} from './types'

function initialQuery() {
  const query = new URLSearchParams(window.location.search)
  return {
    run: query.get('run') || '',
    intent: query.get('intent') || '',
    config: query.get('config') || '',
    agent: query.get('agent') || '',
  }
}

export default function App() {
  const initial = useMemo(initialQuery, [])
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runId, setRunId] = useState(initial.run)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [intentId, setIntentId] = useState(initial.intent)
  const [configId, setConfigId] = useState(initial.config)
  const [agentId, setAgentId] = useState(initial.agent)
  const [agent, setAgent] = useState<AgentDetail | null>(null)
  const [connected, setConnected] = useState(false)
  const [consoleOpen, setConsoleOpen] = useState(true)
  const [error, setError] = useState('')

  const refreshRuns = useCallback(async () => {
    try {
      const response = await listRuns()
      setRuns(response.runs)
      setRunId((current) => current || response.runs[0]?.run_id || '')
    } catch (reason) {
      setError(String(reason))
    }
  }, [])

  const refreshSnapshot = useCallback(async () => {
    if (!runId) return
    try {
      const value = await fetchSnapshot(runId)
      setSnapshot(value)
      setIntentId((current) =>
        value.intents.some((item) => item.intent_id === current)
          ? current
          : value.intents[value.intents.length - 1]?.intent_id || '',
      )
      setError('')
    } catch (reason) {
      setError(String(reason))
    }
  }, [runId])

  useEffect(() => {
    void refreshRuns()
  }, [refreshRuns])

  useEffect(() => {
    setSnapshot(null)
    setAgent(null)
    setAgentId('')
    void refreshSnapshot()
  }, [runId, refreshSnapshot])

  const intent: IntentView | undefined = snapshot?.intents.find(
    (item) => item.intent_id === intentId,
  )
  const config: SwarmConfigView | undefined =
    intent?.configs.find((item) => item.config_id === configId) ||
    intent?.configs[intent.configs.length - 1]

  useEffect(() => {
    if (config && config.config_id !== configId) setConfigId(config.config_id)
  }, [config, configId])

  useEffect(() => {
    const query = new URLSearchParams()
    if (runId) query.set('run', runId)
    if (intentId) query.set('intent', intentId)
    if (config?.config_id) query.set('config', config.config_id)
    if (agentId) query.set('agent', agentId)
    window.history.replaceState(null, '', `?${query.toString()}`)
  }, [runId, intentId, config, agentId])

  useEffect(() => {
    if (!runId || !snapshot) return
    const source = new EventSource(streamUrl(runId, snapshot.last_event_seq))
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(false)
    source.addEventListener('swarm', () => void refreshSnapshot())
    return () => {
      source.close()
      setConnected(false)
    }
  }, [runId, snapshot?.last_event_seq, refreshSnapshot])

  useEffect(() => {
    if (!runId || !agentId) {
      setAgent(null)
      return
    }
    void fetchAgent(runId, agentId)
      .then(setAgent)
      .catch((reason) => setError(String(reason)))
  }, [runId, agentId, snapshot?.last_event_seq])

  return (
    <div className="observatory">
      <header className="run-header">
        <div className="brand">
          <span className="brand-mark">RM</span>
          <div>
            <strong>Swarm Observatory</strong>
            <small>read-only agentic manipulation trace</small>
          </div>
        </div>
        <div className="task-title">
          <span>Task</span>
          <strong>{String(snapshot?.run.task || 'Select a run')}</strong>
        </div>
        <div className="observation-peek">
          {snapshot?.latest_observations.slice(-2).map((artifact) => (
            <img
              key={artifact.artifact_id}
              src={artifactUrl(runId, artifact.artifact_id)}
              title={artifact.path}
              alt="latest robot observation"
            />
          ))}
        </div>
        <select value={runId} onChange={(event) => setRunId(event.target.value)}>
          <option value="">Select run</option>
          {runs.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {run.run_id} · {run.status}
            </option>
          ))}
        </select>
        <div className={`live-pill ${connected ? 'connected' : ''}`}>
          <i /> {snapshot?.live ? (connected ? 'LIVE' : 'RECONNECTING') : 'REPLAY'}
        </div>
        <div className="run-meta">
          <span>{String(snapshot?.run.profile || '—')}</span>
          <span>{String(snapshot?.run.status || 'idle')}</span>
          <span>{elapsed(snapshot?.run.started_at, snapshot?.run.finished_at)}</span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="workspace">
        <aside className="intent-rail panel">
          <div className="panel-title">
            <span>ActionIntent rounds</span>
            <b>{snapshot?.intents.length || 0}</b>
          </div>
          <div className="intent-list">
            {snapshot?.intents.map((item, index) => (
              <button
                key={item.intent_id}
                className={`intent-card ${item.intent_id === intentId ? 'active' : ''}`}
                onClick={() => {
                  setIntentId(item.intent_id)
                  setConfigId(item.configs[item.configs.length - 1]?.config_id || '')
                  setAgentId('')
                }}
              >
                <span className="intent-index">{String(index + 1).padStart(2, '0')}</span>
                <div>
                  <strong>{item.instruction || item.intent_id}</strong>
                  <small>{item.expected_effect || 'awaiting planner decision'}</small>
                  <footer>
                    <em className={`status ${item.status}`}>{item.status}</em>
                    <span>world r{item.observation_revision ?? '—'}</span>
                  </footer>
                </div>
              </button>
            ))}
          </div>
          {intent && (
            <details className="raw-details">
              <summary>Planner decision</summary>
              <label>Raw response</label>
              <pre>{intent.planner.response}</pre>
              <label>Parsed</label>
              <pre>{pretty(intent.planner.decision)}</pre>
            </details>
          )}
        </aside>

        <section className="center-stage">
          <div className="intent-banner panel">
            <div>
              <span>Current embodied instruction</span>
              <h1>{intent?.instruction || 'Waiting for Planner'}</h1>
              <p>{intent?.expected_effect || 'No ActionIntent is available yet.'}</p>
            </div>
            <div className="config-picker">
              <label>SwarmConfig revision</label>
              <select
                value={config?.config_id || ''}
                onChange={(event) => {
                  setConfigId(event.target.value)
                  setAgentId('')
                }}
              >
                {intent?.configs.map((item) => (
                  <option key={item.config_id}>{item.config_id}</option>
                ))}
              </select>
              {config && (
                <details className="manager-details">
                  <summary>Manager I/O</summary>
                  <pre>{pretty(config.manager.request)}</pre>
                  <pre>{config.manager.response}</pre>
                  <pre>{pretty(config.manager.swarm_config)}</pre>
                </details>
              )}
            </div>
          </div>

          <div className="graph-panel panel">
            <div className="panel-title">
              <span>Agent Swarm · actual dependency DAG</span>
              <small>{config?.nodes.length || 0} agents</small>
            </div>
            <SwarmGraph
              nodes={config?.nodes || []}
              edges={config?.edges || []}
              selectedAgent={agentId}
              onSelect={setAgentId}
            />
          </div>

          <CandidateArena
            runId={runId}
            config={config}
            artifacts={snapshot?.artifacts || []}
          />
        </section>

        <AgentInspector
          summary={agentId ? snapshot?.agents[agentId] : undefined}
          detail={agent}
          runId={runId}
          onArtifact={(artifactId) =>
            window.open(artifactUrl(runId, artifactId), '_blank', 'noopener')
          }
        />
      </main>

      <section className={`activity-console ${consoleOpen ? 'open' : ''}`}>
        <button onClick={() => setConsoleOpen((value) => !value)}>
          Activity Console
          <span>{snapshot?.activity.length || 0} events · seq {snapshot?.last_event_seq || 0}</span>
        </button>
        {consoleOpen && <Activity events={snapshot?.activity || []} />}
      </section>
    </div>
  )
}

function Activity({ events }: { events: ActivityEvent[] }) {
  return (
    <div className="activity-lines">
      {events.slice(-80).reverse().map((event) => (
        <div key={event.event_seq}>
          <time>#{event.event_seq}</time>
          <b>{event.stage}/{event.event}</b>
          <span>{event.summary}</span>
          <em className={event.status}>{event.status}</em>
        </div>
      ))}
    </div>
  )
}

export function pretty(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function elapsed(start: unknown, finish: unknown) {
  const startNumber = Number(start || 0)
  const finishNumber = Number(finish || Date.now() / 1000)
  if (!startNumber) return '—'
  return `${Math.max(0, finishNumber - startNumber).toFixed(0)}s`
}
