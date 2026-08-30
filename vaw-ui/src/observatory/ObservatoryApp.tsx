import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type {
  ActionSegment,
  ControlConfig,
  ImaginationSessionView,
  ImaginationSummary,
  LaunchFormState,
  ModelIOView,
  ObservatorySnapshot,
  RunSummary,
  TurnView,
} from './types'

const EVENT_TYPES = [
  'turn_context_ready',
  'model_started',
  'model_request_saved',
  'model_response_saved',
  'model_decision_ready',
  'function_started',
  'imagination_started',
  'imagination_turn_closed',
  'imagination_closed',
  'action_segment_started',
  'action_segment_ready',
  'turn_closed',
  'episode_closed',
]

function json(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

function compact(value: unknown): string {
  return JSON.stringify(value)
}

function readable(value: unknown): string {
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function messageImageLabel(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && 'url' in value) {
    return String((value as { url?: unknown }).url ?? '（图像地址为空）')
  }
  return readable(value)
}

function MessageContent({ content }: { content: unknown }) {
  if (!Array.isArray(content)) {
    return <pre className="aos-message-text">{readable(content)}</pre>
  }

  return (
    <div className="aos-message-parts">
      {content.map((part, index) => {
        if (typeof part === 'string') {
          return <pre className="aos-message-text" key={index}>{part}</pre>
        }
        if (!part || typeof part !== 'object') {
          return <pre className="aos-message-structured" key={index}>{readable(part)}</pre>
        }

        const item = part as Record<string, unknown>
        if (item.type === 'text' && typeof item.text === 'string') {
          return <pre className="aos-message-text" key={index}>{item.text}</pre>
        }
        if (item.type === 'image_url') {
          return (
            <div className="aos-message-image" key={index}>
              <span>IMAGE INPUT</span>
              <code>{messageImageLabel(item.image_url)}</code>
            </div>
          )
        }
        return <pre className="aos-message-structured" key={index}>{readable(item)}</pre>
      })}
    </div>
  )
}

function artifactUrl(runId: string, relative: string | null | undefined): string | null {
  if (!relative) return null
  return `/api/runs/${encodeURIComponent(runId)}/${relative.split('/').map(encodeURIComponent).join('/')}`
}

function StatusDot({ status }: { status: string }) {
  return <span className={`aos-status aos-status--${status}`}>{status.replace('_', ' ')}</span>
}

function TurnRail({ turns, selected, onSelect }: {
  turns: TurnView[]
  selected: number | null
  onSelect: (turn: number) => void
}) {
  return (
    <aside className="aos-turn-rail">
      <h2>Turns</h2>
      <div className="aos-turn-list">
        {turns.map((item) => (
          <button
            type="button"
            key={item.turn}
            className={item.turn === selected ? 'aos-turn aos-turn--selected' : 'aos-turn'}
            onClick={() => onSelect(item.turn)}
          >
            <span className="aos-turn-number">{String(item.turn).padStart(2, '0')}</span>
            <span className="aos-turn-call">
              <strong>{item.decision.function || 'waiting for model'}</strong>
              <small>
                {item.decision.revision_before == null
                  ? `observation r${item.context?.revision ?? '—'}`
                  : `r${item.decision.revision_before} → r${item.decision.revision_after}`}
              </small>
            </span>
            <StatusDot status={item.status} />
          </button>
        ))}
      </div>
    </aside>
  )
}

function StateCard({ turn }: { turn: TurnView }) {
  const state = turn.context?.embodied_state_card
  const position = state?.tcp_pose?.position_xyz
  const quaternion = state?.tcp_pose?.quaternion_xyzw
  return (
    <section className="aos-card">
      <h3>Embodied state · model-visible</h3>
      <dl className="aos-state-grid">
        <div><dt>TCP xyz</dt><dd>{position ? compact(position) : 'unknown'}</dd></div>
        <div><dt>Quaternion xyzw</dt><dd>{quaternion ? compact(quaternion) : 'unknown'}</dd></div>
        <div><dt>Gripper opening</dt><dd>{state?.gripper_opening ?? 'unknown'}</dd></div>
        <div><dt>Active action</dt><dd>{state?.active_action ? compact(state.active_action) : 'none'}</dd></div>
        <div><dt>Last action</dt><dd>{state?.last_action ? compact(state.last_action) : 'none'}</dd></div>
        <div><dt>Last gripper action</dt><dd>{state?.last_gripper_action ? compact(state.last_gripper_action) : 'none'}</dd></div>
      </dl>
    </section>
  )
}

function MemoryCard({ turn }: { turn: TurnView }) {
  const lines = turn.context?.interaction_memory_prompt ?? []
  return (
    <section className="aos-card">
      <h3>Short-term interaction memory · before turn</h3>
      {lines.length ? (
        <ol className="aos-memory">
          {lines.map((line, index) => <li key={`${index}-${line}`}>{line}</li>)}
        </ol>
      ) : <p className="aos-empty">Empty before this turn</p>}
    </section>
  )
}

function ReferencesCard({ turn }: { turn: TurnView }) {
  const references = turn.context?.live_references ?? {}
  const regions = Array.isArray(references.regions) ? references.regions : []
  const points = Array.isArray(references.points) ? references.points : []
  const seeds = Array.isArray(references.seed_ids) ? references.seed_ids : []
  return (
    <section className="aos-card">
      <h3>Live references · r{turn.context?.revision ?? '—'}</h3>
      <dl className="aos-reference-list">
        <div><dt>Regions</dt><dd>{regions.length ? compact(regions) : 'none'}</dd></div>
        <div><dt>Points</dt><dd>{points.length ? compact(points) : 'none'}</dd></div>
        <div><dt>Seeds</dt><dd>{seeds.length ? compact(seeds) : 'none'}</dd></div>
        <div><dt>Action</dt><dd>{references.action_proposal ? compact(references.action_proposal) : 'none'}</dd></div>
      </dl>
      {turn.context?.protocol_feedback ? (
        <p className="aos-advisory">{turn.context.protocol_feedback}</p>
      ) : null}
    </section>
  )
}

function DecisionPanel({ turn }: { turn: TurnView }) {
  const decision = turn.decision
  return (
    <section className="aos-decision">
      <div className="aos-decision-basis">
        <h3>Decision basis</h3>
        <p>{decision.basis || (turn.status === 'thinking' ? 'Model is reasoning…' : 'No decision yet')}</p>
      </div>
      <div className="aos-function-card">
        <h3>Selected function</h3>
        <code>{decision.function ? `${decision.function}()` : 'pending'}</code>
        {decision.arguments ? (
          <dl>
            {Object.entries(decision.arguments).map(([key, value]) => (
              <div key={key}><dt>{key}</dt><dd>{compact(value)}</dd></div>
            ))}
          </dl>
        ) : null}
        {decision.error ? <p className="aos-error">{decision.error}</p> : null}
        {decision.advisory ? <p className="aos-advisory">{decision.advisory}</p> : null}
      </div>
    </section>
  )
}

function ModelIOPanel({ value, loading, owner = 'Main' }: { value: ModelIOView | null; loading: boolean; owner?: string }) {
  const [attemptIndex, setAttemptIndex] = useState(0)
  useEffect(() => setAttemptIndex(0), [value?.turn])
  if (loading) return <section className="aos-model-io"><div className="aos-loading-inline">正在读取模型请求与响应…</div></section>
  if (!value?.attempts.length) return <section className="aos-model-io"><div className="aos-empty">本轮没有可用的模型 I/O 记录。</div></section>
  const attempt = value.attempts[Math.min(attemptIndex, value.attempts.length - 1)]
  const request = attempt.request ?? {}
  const response = attempt.response ?? {}
  const messages = Array.isArray(request.messages) ? request.messages as Array<Record<string, unknown>> : []
  const tools = Array.isArray(request.tools) ? request.tools : []
  return (
    <section className="aos-model-io">
      <header>
        <div>
          <small>{owner.toUpperCase()} MODEL I/O · TURN {String(value.turn).padStart(2, '0')}</small>
          <h2>{owner} 的模型原始请求与原始响应</h2>
        </div>
        <div className="aos-attempts">
          {value.legacy ? <span className="aos-legacy">旧格式：请求不完整</span> : null}
          {value.attempts.map((item, index) => (
            <button key={item.attempt ?? index} type="button" className={index === attemptIndex ? 'active' : ''} onClick={() => setAttemptIndex(index)}>
              尝试 {item.attempt ?? index + 1}
            </button>
          ))}
        </div>
      </header>
      <div className="aos-io-grid">
        <article className="aos-io-column">
          <h3>RAW REQUEST</h3>
          {messages.length ? messages.map((message, index) => (
            <section className="aos-message" key={`${String(message.role)}-${index}`}>
              <span className={`aos-role aos-role--${String(message.role ?? 'unknown')}`}>{String(message.role ?? 'unknown')}</span>
              <MessageContent content={message.content} />
            </section>
          )) : (
            <pre>{readable(request.prompt_text ?? request.note ?? request)}</pre>
          )}
          <details className="aos-tools" open={false}>
            <summary>工具结构 · {tools.length} 项</summary>
            <pre>{readable(tools)}</pre>
          </details>
        </article>
        <article className="aos-io-column aos-io-response">
          <h3>RAW RESPONSE</h3>
          {response.error ? <p className="aos-error">{String(response.error)}</p> : null}
          <section className="aos-response-block">
            <h4>原始文本</h4>
            <pre>{readable(response.raw_response_text ?? '（空）')}</pre>
          </section>
          {response.provider_reasoning ? (
            <section className="aos-response-block">
              <h4>Provider reasoning</h4>
              <pre>{readable(response.provider_reasoning)}</pre>
            </section>
          ) : null}
          <section className="aos-response-block">
            <h4>解析结果</h4>
            <pre>{readable({ parsed_text: response.parsed_text, tool_calls: response.tool_calls, finish_reason: response.finish_reason, usage: response.usage })}</pre>
          </section>
        </article>
      </div>
    </section>
  )
}

function ImaginationPanel({ runId, summary }: {
  runId: string
  summary: ImaginationSummary | null | undefined
}) {
  const [session, setSession] = useState<ImaginationSessionView | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [modelIO, setModelIO] = useState<ModelIOView | null>(null)
  const [loading, setLoading] = useState(false)
  const [ioLoading, setIOLoading] = useState(false)

  useEffect(() => {
    if (!summary) {
      setSession(null)
      setSelected(null)
      return
    }
    const controller = new AbortController()
    setLoading(true)
    fetch(`/api/runs/${encodeURIComponent(runId)}/imagination/${encodeURIComponent(summary.session_id)}`, {
      cache: 'no-store',
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status))
        return response.json() as Promise<ImaginationSessionView>
      })
      .then((value) => {
        setSession(value)
        setSelected(value.turns.at(-1)?.turn ?? null)
      })
      .catch((reason) => { if (reason?.name !== 'AbortError') setSession(null) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [runId, summary?.session_id, summary?.turn_count, summary?.status])

  const turn = session?.turns.find((item) => item.turn === selected) ?? null
  useEffect(() => {
    if (!summary || selected == null || !turn?.model_io_available) {
      setModelIO(null)
      setIOLoading(false)
      return
    }
    const controller = new AbortController()
    setIOLoading(true)
    fetch(`/api/runs/${encodeURIComponent(runId)}/imagination/${encodeURIComponent(summary.session_id)}/turns/${selected}/model-io`, {
      cache: 'no-store',
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status))
        return response.json() as Promise<ModelIOView>
      })
      .then(setModelIO)
      .catch((reason) => { if (reason?.name !== 'AbortError') setModelIO(null) })
      .finally(() => { if (!controller.signal.aborted) setIOLoading(false) })
    return () => controller.abort()
  }, [runId, selected, summary?.session_id, turn?.model_io_available])

  if (!summary) return null
  return (
    <section className="aos-imagination">
      <header>
        <div>
          <small>IMAGINATION SUBAGENT · {summary.session_id}</small>
          <h2>动作想象的逐轮推理过程</h2>
        </div>
        <StatusDot status={session?.status ?? summary.status} />
      </header>
      <p className="aos-imagination-instruction">{session?.instruction ?? summary.instruction ?? '未记录局部任务'}</p>
      {loading ? <div className="aos-loading-inline">正在读取 Imagination 子 trace…</div> : null}
      {session?.turns.length ? (
        <>
          <nav className="aos-imagination-turns" aria-label="Imagination turns">
            {session.turns.map((item) => (
              <button key={item.turn} type="button" className={item.turn === selected ? 'active' : ''} onClick={() => setSelected(item.turn)}>
                <span>I{String(item.turn).padStart(2, '0')}</span>
                <strong>{item.function_call?.name ?? 'thinking'}</strong>
              </button>
            ))}
          </nav>
          {turn ? (
            <div className="aos-imagination-grid">
              <div className="aos-imagination-canvas">
                <div className="aos-canvas-title"><strong>Imagination Canvas</strong><span>INTERNAL TURN {turn.turn}</span></div>
                {turn.canvas ? <img src={artifactUrl(runId, turn.canvas) ?? ''} alt={`Imagination turn ${turn.turn} Canvas`} /> : <div className="aos-no-context">No Canvas</div>}
              </div>
              <div className="aos-imagination-reasoning">
                <section><h3>推理依据</h3><p>{turn.decision_basis || '没有保存文字依据'}</p></section>
                <section><h3>调用的局部工具</h3><code>{turn.function_call?.name ? `${turn.function_call.name}()` : 'none'}</code><pre>{readable(turn.function_call?.arguments ?? {})}</pre></section>
                <section><h3>执行结果</h3><pre>{readable(turn.function_result ?? {})}</pre></section>
              </div>
            </div>
          ) : null}
          <ModelIOPanel value={modelIO} loading={ioLoading} owner="Imagination" />
        </>
      ) : !loading ? <div className="aos-empty">该次 Imagination 没有完成任何内部 turn。</div> : null}
    </section>
  )
}

function EpisodeVideoPanel({ runId, videos, closed }: {
  runId: string
  videos: ObservatorySnapshot['episode_videos']
  closed: boolean
}) {
  const names = Object.keys(videos)
  const [selected, setSelected] = useState('agentview')
  useEffect(() => {
    if (!names.includes(selected)) setSelected(names[0] ?? 'agentview')
  }, [names.join('|'), selected])
  if (!names.length) {
    return closed ? null : <section className="aos-episode-video aos-episode-video--pending">完整 Episode 视频将在任务结束后生成；下方动作分段可实时观看。</section>
  }
  const current = videos[selected] ?? videos[names[0]]
  const source = artifactUrl(runId, current?.path)
  return (
    <section className="aos-episode-video">
      <header>
        <div><small>COMPLETE EPISODE</small><h2>完整任务视频</h2></div>
        <div className="aos-camera-toggle">
          {names.map((name) => <button key={name} type="button" className={name === selected ? 'active' : ''} onClick={() => setSelected(name)}>{name}</button>)}
        </div>
      </header>
      {source ? <video key={source} src={source} controls preload="metadata" /> : null}
    </section>
  )
}

function ActionTape({ runId, segments, selectedTurn }: {
  runId: string
  segments: ActionSegment[]
  selectedTurn: number | null
}) {
  const visible = useMemo(
    () => selectedTurn == null ? segments : segments.filter((item) => item.turn <= selectedTurn),
    [segments, selectedTurn],
  )
  const selectedIndex = visible.findIndex((item) => item.turn === selectedTurn)
  const defaultIndex = selectedIndex >= 0 ? selectedIndex : Math.max(0, visible.length - 1)
  const [playingIndex, setPlayingIndex] = useState(defaultIndex)
  const [camera, setCamera] = useState('agentview')
  const video = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    setPlayingIndex(defaultIndex >= 0 ? defaultIndex : Math.max(0, visible.length - 1))
  }, [defaultIndex, visible.length])

  const current = visible[playingIndex]
  const stream = current?.streams[camera] ?? current?.streams.agentview ?? current?.streams.wrist
  const source = artifactUrl(runId, stream?.path)

  const playThrough = () => {
    if (!visible.length) return
    setPlayingIndex(0)
    requestAnimationFrame(() => video.current?.play())
  }

  const onEnded = () => {
    if (playingIndex + 1 < visible.length) {
      setPlayingIndex(playingIndex + 1)
      requestAnimationFrame(() => video.current?.play())
    }
  }

  return (
    <section className="aos-action-tape">
      <header>
        <h2>Physical Action Tape</h2>
        <button type="button" onClick={playThrough} disabled={!visible.length}>▶ Play through selected</button>
        <div className="aos-camera-toggle">
          <button type="button" className={camera === 'agentview' ? 'active' : ''} onClick={() => setCamera('agentview')}>agentview</button>
          <button type="button" className={camera === 'wrist' ? 'active' : ''} onClick={() => setCamera('wrist')}>wrist</button>
        </div>
      </header>
      <div className="aos-action-body">
        <div className="aos-video-stage">
          {source ? (
            <video key={source} ref={video} src={source} controls onEnded={onEnded} />
          ) : <div className="aos-no-video">No physical frames for the selected segment</div>}
        </div>
        <div className="aos-segments">
          {segments.map((segment, index) => (
            <button
              type="button"
              key={segment.segment_id}
              className={segment === current ? 'aos-segment aos-segment--selected' : 'aos-segment'}
              onClick={() => setPlayingIndex(Math.max(0, visible.findIndex((item) => item.segment_id === segment.segment_id)))}
              disabled={selectedTurn != null && segment.turn > selectedTurn}
            >
              <span>A{String(index + 1).padStart(2, '0')}</span>
              <strong>{segment.function}</strong>
              <small>t{segment.turn} · r{segment.revision_before}→r{segment.revision_after}</small>
              <em>{segment.outcome}</em>
            </button>
          ))}
        </div>
      </div>
    </section>
  )
}

function LaunchPanel({ config, onClose, onLaunched }: {
  config: ControlConfig
  onClose: () => void
  onLaunched: (runId: string) => void
}) {
  const [form, setForm] = useState<LaunchFormState>({
    run_name: `${config.default_suite.replaceAll('_', '-')}-t0-s1`,
    suite: config.default_suite,
    task_id: 0,
    seed: 1,
    model: config.default_model,
    imagination_model: '',
    protocol: 'native',
    motion_backend: 'curobo',
    max_turns: 32,
    max_time_s: 1800,
    max_physical_ops: 30,
    collection: config.default_collection,
  })
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const update = <K extends keyof LaunchFormState>(key: K, value: LaunchFormState[K]) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setLaunching(true)
    setError(null)
    try {
      const response = await fetch('/api/control/launches', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...form,
          imagination_model: form.imagination_model.trim() || null,
        }),
      })
      if (!response.ok) throw new Error(await response.text())
      const value = await response.json() as { run_id: string }
      onLaunched(value.run_id)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setLaunching(false)
    }
  }

  return (
    <div className="aos-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <form className="aos-launch-panel" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div>
            <small>CONTROLLED EPISODE LAUNCH</small>
            <h2>Start a VAW task</h2>
          </div>
          <button type="button" onClick={onClose}>×</button>
        </header>
        <p className="aos-launch-note">
          Starts one complete Agent OS episode. It does not expose shell commands or direct Robot Functions.
        </p>
        <div className="aos-launch-grid">
          <label className="wide">独立任务名称<input required value={form.run_name} placeholder="例如 gemini-object-task0-seed1" onChange={(event) => update('run_name', event.target.value)} /></label>
          <label className="wide">Suite<input list="suite-suggestions" value={form.suite} onChange={(event) => update('suite', event.target.value)} /></label>
          <datalist id="suite-suggestions">{config.suite_suggestions.map((value) => <option key={value} value={value} />)}</datalist>
          <label>Task ID<input type="number" min="0" value={form.task_id} onChange={(event) => update('task_id', Number(event.target.value))} /></label>
          <label>Seed<input type="number" min="0" value={form.seed} onChange={(event) => update('seed', Number(event.target.value))} /></label>
          <label className="wide">Main model<input list="model-suggestions" value={form.model} onChange={(event) => update('model', event.target.value)} /></label>
          <datalist id="model-suggestions">{config.model_suggestions.map((value) => <option key={value} value={value} />)}</datalist>
          <label className="wide">Imagination model <span>optional</span><input value={form.imagination_model} placeholder="reuse Main model" onChange={(event) => update('imagination_model', event.target.value)} /></label>
          <label>Motion<select value={form.motion_backend} onChange={(event) => update('motion_backend', event.target.value as LaunchFormState['motion_backend'])}><option value="curobo">CuRobo</option><option value="pyroki">PyRoki</option></select></label>
          <label>Protocol<select value={form.protocol} onChange={(event) => update('protocol', event.target.value as LaunchFormState['protocol'])}><option value="native">Native tools</option><option value="text">Text fallback</option></select></label>
          <label>Max turns<input type="number" min="1" value={form.max_turns} onChange={(event) => update('max_turns', Number(event.target.value))} /></label>
          <label>Max physical ops<input type="number" min="1" value={form.max_physical_ops} onChange={(event) => update('max_physical_ops', Number(event.target.value))} /></label>
          <label>Timeout (s)<input type="number" min="1" value={form.max_time_s} onChange={(event) => update('max_time_s', Number(event.target.value))} /></label>
          <label>Output collection<input list="collection-suggestions" value={form.collection} onChange={(event) => update('collection', event.target.value)} /></label>
          <datalist id="collection-suggestions">{config.collections.map((value) => <option key={value} value={value} />)}</datalist>
        </div>
        {error ? <pre className="aos-launch-error">{error}</pre> : null}
        <footer>
          <span>Workspace: {config.workspace}</span>
          <button type="button" onClick={onClose}>Cancel</button>
          <button className="primary" type="submit" disabled={launching}>
            {launching ? 'Starting…' : 'Start episode'}
          </button>
        </footer>
      </form>
    </div>
  )
}

export function ObservatoryApp() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [snapshot, setSnapshot] = useState<ObservatorySnapshot | null>(null)
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null)
  const [followLive, setFollowLive] = useState(true)
  const [collection, setCollection] = useState('all')
  const [controlConfig, setControlConfig] = useState<ControlConfig | null>(null)
  const [launchOpen, setLaunchOpen] = useState(false)
  const [controlError, setControlError] = useState<string | null>(null)
  const [pathFilter, setPathFilter] = useState('')
  const [modelIO, setModelIO] = useState<ModelIOView | null>(null)
  const [modelIOLoading, setModelIOLoading] = useState(false)
  const refreshTimer = useRef<number | null>(null)

  const loadSnapshot = useCallback(async (id: string) => {
    const response = await fetch(`/api/runs/${encodeURIComponent(id)}/snapshot`, { cache: 'no-store' })
    if (!response.ok) throw new Error(await response.text())
    const next = await response.json() as ObservatorySnapshot
    setSnapshot(next)
    setSelectedTurn((current) => {
      if (followLive || current == null || !next.turns.some((turn) => turn.turn === current)) {
        return next.turns.at(-1)?.turn ?? null
      }
      return current
    })
  }, [followLive])

  const loadRuns = useCallback(async () => {
    const response = await fetch('/api/runs', { cache: 'no-store' })
    if (!response.ok) throw new Error(await response.text())
    const value = await response.json() as { runs: RunSummary[] }
    setRuns(value.runs)
    setRunId((current) => {
      if (current && value.runs.some((run) => run.run_id === current)) return current
      const requested = new URLSearchParams(location.search).get('run')
      return requested && value.runs.some((run) => run.run_id === requested)
        ? requested
        : value.runs[0]?.run_id ?? null
    })
  }, [])

  useEffect(() => {
    loadRuns().catch((reason) => setControlError(String(reason)))
    fetch('/api/control/config', { cache: 'no-store' })
      .then((response) => response.json())
      .then((value: ControlConfig) => {
        setControlConfig(value)
        setCollection((current) => current === 'all' ? value.default_collection : current)
      })
      .catch((reason) => setControlError(String(reason)))
    const timer = window.setInterval(() => loadRuns().catch(() => undefined), 2000)
    return () => window.clearInterval(timer)
  }, [loadRuns])

  useEffect(() => {
    if (!runId) return
    setSnapshot(null)
    setSelectedTurn(null)
    loadSnapshot(runId).catch((reason) => setControlError(String(reason)))
    const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream?after_seq=0`)
    const schedule = () => {
      if (refreshTimer.current != null) window.clearTimeout(refreshTimer.current)
      refreshTimer.current = window.setTimeout(() => loadSnapshot(runId), 80)
    }
    EVENT_TYPES.forEach((type) => source.addEventListener(type, schedule))
    const fallback = window.setInterval(() => loadSnapshot(runId).catch(() => undefined), 2000)
    return () => {
      source.close()
      window.clearInterval(fallback)
      if (refreshTimer.current != null) window.clearTimeout(refreshTimer.current)
    }
  }, [loadSnapshot, runId])

  const turn = snapshot?.turns.find((item) => item.turn === selectedTurn) ?? null
  const collections = useMemo(
    () => ['all', ...Array.from(new Set(runs.map((run) => run.collection))).sort()],
    [runs],
  )
  const visibleRuns = useMemo(
    () => runs.filter((run) => {
      if (collection !== 'all' && run.collection !== collection) return false
      const needle = pathFilter.trim().toLocaleLowerCase()
      return !needle || `${run.relative_path} ${run.display_name} ${run.task ?? ''}`.toLocaleLowerCase().includes(needle)
    }),
    [collection, pathFilter, runs],
  )

  useEffect(() => {
    if (!runId || selectedTurn == null || !turn?.model_io_available) {
      setModelIO(null)
      setModelIOLoading(false)
      return
    }
    const controller = new AbortController()
    setModelIOLoading(true)
    fetch(`/api/runs/${encodeURIComponent(runId)}/turns/${selectedTurn}/model-io`, { cache: 'no-store', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status))
        return response.json() as Promise<ModelIOView>
      })
      .then(setModelIO)
      .catch((reason) => { if (reason?.name !== 'AbortError') setModelIO(null) })
      .finally(() => { if (!controller.signal.aborted) setModelIOLoading(false) })
    return () => controller.abort()
  }, [runId, selectedTurn, turn?.model_io_available])

  useEffect(() => {
    if (runId && visibleRuns.some((run) => run.run_id === runId)) return
    if (visibleRuns.length) setRunId(visibleRuns[0].run_id)
  }, [runId, visibleRuns])

  const launchCompleted = (nextRunId: string) => {
    setLaunchOpen(false)
    setCollection('all')
    setFollowLive(true)
    setSelectedTurn(null)
    setRunId(nextRunId)
    loadRuns().catch((reason) => setControlError(String(reason)))
  }

  const stopCurrent = async () => {
    if (!snapshot?.run.job_id) return
    setControlError(null)
    const response = await fetch(
      `/api/control/launches/${encodeURIComponent(snapshot.run.job_id)}/stop`,
      {
        method: 'POST',
      },
    )
    if (!response.ok) setControlError(await response.text())
    else loadRuns().catch(() => undefined)
  }

  return (
    <main className="aos-shell">
      <header className="aos-header">
        <div className="aos-brand">VAW · Agent OS Observatory</div>
        <span className={snapshot?.run.live ? 'aos-live' : 'aos-replay'}>
          ● {snapshot?.run.live ? 'LIVE' : 'REPLAY'}
        </span>
        <div className="aos-task">
          <small>{snapshot?.run.display_name ?? 'TASK'}</small>
          <strong>{snapshot?.run.task || (snapshot?.run.suite ? `${snapshot.run.suite}:${snapshot.run.task_id}` : 'Select a run')}</strong>
        </div>
        <input className="aos-path-filter" value={pathFilter} placeholder="按运行路径搜索" onChange={(event) => setPathFilter(event.target.value)} />
        <select className="aos-collection-select" value={collection} onChange={(event) => setCollection(event.target.value)}>
          {collections.map((value) => <option key={value} value={value}>{value === 'all' ? 'All collections' : value}</option>)}
        </select>
        <select className="aos-run-select" value={runId ?? ''} onChange={(event) => setRunId(event.target.value)}>
          {visibleRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.relative_path} · {run.status}</option>)}
        </select>
        {snapshot?.run.job_id && ['starting', 'running'].includes(snapshot.run.launcher_status ?? '') ? (
          <button className="aos-stop-button" type="button" onClick={stopCurrent}>■ Stop</button>
        ) : null}
        <button className="aos-launch-button" type="button" onClick={() => setLaunchOpen(true)}>＋ Start task</button>
        <label className="aos-follow">
          <input type="checkbox" checked={followLive} onChange={(event) => setFollowLive(event.target.checked)} />
          follow live
        </label>
      </header>

      {controlError ? <div className="aos-control-error">{controlError}</div> : null}

      {snapshot && turn ? (
        <>
          <div className="aos-workspace">
            <TurnRail turns={snapshot.turns} selected={selectedTurn} onSelect={(value) => { setSelectedTurn(value); setFollowLive(false) }} />
            <section className="aos-center">
              <header className="aos-canvas-title">
                <strong>Model-visible Canvas</strong>
                <span>FROZEN · TURN {String(turn.turn).padStart(2, '0')} · r{turn.context?.revision ?? '—'}</span>
              </header>
              {turn.canvas ? (
                <img className="aos-canvas" src={artifactUrl(runId!, turn.canvas) ?? ''} alt={`Turn ${turn.turn} model-visible Canvas`} />
              ) : <div className="aos-no-context">This run has no frozen model-visible Context snapshot.</div>}
              <DecisionPanel turn={turn} />
            </section>
            <aside className="aos-context-panel">
              <StateCard turn={turn} />
              <ReferencesCard turn={turn} />
              <MemoryCard turn={turn} />
            </aside>
          </div>
          <ModelIOPanel value={modelIO} loading={modelIOLoading} />
          <ImaginationPanel runId={runId!} summary={turn.imagination} />
          <EpisodeVideoPanel runId={runId!} videos={snapshot.episode_videos ?? {}} closed={snapshot.closed} />
          <ActionTape runId={runId!} segments={snapshot.action_segments} selectedTurn={selectedTurn} />
        </>
      ) : <div className="aos-loading">Waiting for the first frozen Context…</div>}
      {launchOpen && controlConfig ? (
        <LaunchPanel
          config={controlConfig}
          onClose={() => setLaunchOpen(false)}
          onLaunched={launchCompleted}
        />
      ) : null}
    </main>
  )
}
