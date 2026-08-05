import type { ReactNode } from 'react'

import type { ContextSnapshot } from '../types'

type Candidate = ContextSnapshot['catalog']['candidates'][number]
type Tone = 'neutral' | 'blue' | 'green' | 'violet' | 'amber' | 'red'

function fmt(values: number[] | undefined, digits = 4): string {
  if (!values) return '不可用'
  return `[${values.map((value) => value.toFixed(digits)).join(', ')}]`
}

function compactJson(value: unknown, limit = 180): string {
  const encoded = JSON.stringify(value ?? {}, null, 0)
  return encoded.length > limit ? `${encoded.slice(0, limit - 1)}…` : encoded
}

function Chip({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`ctx-chip ctx-chip--${tone}`}>{children}</span>
}

function Raster({ snapshot, id, alt }: {
  snapshot: ContextSnapshot
  id: string | null
  alt: string
}) {
  const src = id ? snapshot.rasters[id] : null
  return src ? <img src={src} alt={alt} /> : <div className="ctx-no-raster">暂无对应视觉证据</div>
}

function CompactRobot({ snapshot }: { snapshot: ContextSnapshot }) {
  const robot = snapshot.world.robot
  const tcp = robot?.tcp_pose ?? robot?.ee_pose
  const opening = robot?.gripper_opening
  return (
    <section className="ctx-robot-compact">
      <div className="ctx-robot-grid">
        <span>GRIP</span><code>{opening?.toFixed(3) ?? 'N/A'}</code>
        <span>TCP</span><code>{fmt(tcp?.position_xyz, 3)}</code>
        <span>QUAT</span><code>{fmt(tcp?.quaternion_xyzw, 3)}</code>
        <span>JOINT</span><code>{fmt(robot?.joint_positions_rad, 3)}</code>
      </div>
    </section>
  )
}

function PersistentWorld({ snapshot }: { snapshot: ContextSnapshot }) {
  return (
    <section className="ctx-section ctx-world">
      <div className="ctx-world-grid">
        <article className="ctx-main-view">
          <div className="ctx-view-label"><strong>AGENTVIEW · PRIMARY</strong><span>CURRENT RGB</span></div>
          <Raster snapshot={snapshot} id={snapshot.world.agentviewRasterId} alt="当前主视角" />
        </article>
        <article className="ctx-near-field-view">
          <div className="ctx-view-label"><strong>GRIPPER-LOCAL · GEOMETRY</strong><span>CURRENT FUSED RGB-D</span></div>
          <Raster snapshot={snapshot} id={snapshot.world.nearFieldRasterId} alt="当前夹爪近场几何" />
        </article>
      </div>
    </section>
  )
}

function GroundingWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const regions = snapshot.decision.regionIds
    .map((id) => snapshot.catalog.regions.find((item) => item.id === id))
    .filter((item): item is ContextSnapshot['catalog']['regions'][number] => Boolean(item))
  const points = snapshot.decision.pointIds
    .map((id) => snapshot.catalog.points.find((item) => item.id === id))
    .filter((item): item is ContextSnapshot['catalog']['points'][number] => Boolean(item))
  return (
    <div className="ctx-grounding-mode">
      <div className="ctx-decision-image"><Raster snapshot={snapshot} id={snapshot.decision.primaryRasterId} alt="最新定位证据" /></div>
      <div className="ctx-grounding-facts">
        <h3>最新 Grounding Evidence</h3>
        {regions.map((region) => (
          <div className="ctx-fact-card ctx-fact-card--blue" key={region.id}>
            <div><strong>{region.id}</strong><Chip tone="blue">REGION</Chip></div>
            <p>{region.query}</p><code>bbox_px {fmt(region.bbox, 1)}</code>
            {region.withinRegionId && <small>within {region.withinRegionId}</small>}
          </div>
        ))}
        {points.map((point) => (
          <div className="ctx-fact-card ctx-fact-card--green" key={point.id}>
            <div><strong>{point.id}</strong><Chip tone="green">POINT</Chip></div>
            <p>{point.query}</p><code>pixel {fmt(point.pixel, 1)}<br />base_xyz_m {fmt(point.position, 4)}</code>
            {point.withinRegionId && <small>within {point.withinRegionId}</small>}
          </div>
        ))}
      </div>
    </div>
  )
}

function millimeters(values: number[] | null): string {
  if (!values) return '无 point anchor'
  return `[${values.map((value) => `${value >= 0 ? '+' : ''}${(value * 1000).toFixed(0)}`).join(', ')}] mm`
}

function CandidateCard({ snapshot, item, compact = false }: {
  snapshot: ContextSnapshot
  item: Candidate
  compact?: boolean
}) {
  const selected = snapshot.world.activeAction?.source_ref === item.id
  const statusTone: Tone = item.solveIk === 'returned' ? 'violet' : item.solveIk === 'error' ? 'red' : 'amber'
  return (
    <article className={`ctx-candidate${selected ? ' ctx-candidate--selected' : ''}${compact ? ' ctx-candidate--compact' : ''}`}>
      <header><strong>{item.id}</strong><Chip tone={selected ? 'green' : statusTone}>{selected ? 'SELECTED' : `FK ${item.solveIk.toUpperCase()}`}</Chip></header>
      <div className="ctx-candidate-raster"><Raster snapshot={snapshot} id={item.rasterId} alt={`候选 ${item.id}`} /></div>
      {!compact && <div className="ctx-candidate-facts">
        <span>delta <code>{millimeters(item.deltaFromAnchor)}</code></span>
        <span>approach <code>{fmt(item.approachVector ?? undefined, 2)}</code></span>
      </div>}
    </article>
  )
}

function CandidatesWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const items = snapshot.decision.candidateIds
    .map((id) => snapshot.catalog.candidates.find((item) => item.id === id))
    .filter((item): item is Candidate => Boolean(item))
  return (
    <div className="ctx-candidates-mode">
      <div className="ctx-candidate-intro"><strong>{items.length} 个候选动作</strong></div>
      <div
        className="ctx-candidate-row"
        style={{ gridTemplateColumns: `repeat(${Math.max(items.length, 1)}, minmax(0, 1fr))` }}
      >
        {items.map((item) => <CandidateCard snapshot={snapshot} item={item} key={item.id} />)}
      </div>
    </div>
  )
}

function ProposalWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const action = snapshot.world.activeAction
  const alternatives = snapshot.decision.candidateIds
    .map((id) => snapshot.catalog.candidates.find((item) => item.id === id))
    .filter((item): item is Candidate => Boolean(item))
  if (!action || action.action_id !== snapshot.decision.actionId) {
    return <ErrorWorkspace snapshot={snapshot} message="proposal mode 与 active action 不一致" />
  }
  const adjustment = action.adjustment
  const statusTone: Tone = action.prediction.solve_ik === 'returned' ? 'violet' : action.prediction.solve_ik === 'error' ? 'red' : 'amber'
  return (
    <div className="ctx-proposal-mode">
      <div className="ctx-proposal-image">
        <Raster snapshot={snapshot} id={snapshot.decision.primaryRasterId} alt="局部动作想象与整臂概览" />
        <div className="ctx-image-legend"><span className="blue">current / reference</span><span className="green">motion</span><span className="violet">imagined hand + arm</span></div>
      </div>
      <aside className="ctx-proposal-facts">
        <div className="ctx-proposal-title"><div><small>ACTION PROPOSAL · IMAGINED · NOT EXECUTED</small><h3>{action.action_id} · {action.kind}</h3></div><Chip tone={statusTone}>solve_ik {action.prediction.solve_ik}</Chip></div>
        {adjustment && <div className="ctx-target-pose">
          <small>LOCAL REFINEMENT · {adjustment.frame.toUpperCase()} FRAME</small>
          <code>
            from {adjustment.parent_action_id ?? 'current TCP'}<br />
            {adjustment.kind === 'delta_move'
              ? `delta_xyz_m ${fmt(adjustment.delta_xyz_m)}`
              : `rotate ${adjustment.axis} ${adjustment.angle_deg?.toFixed(1)} deg`}
          </code>
        </div>}
        {action.prediction.detail && <p className="ctx-proposal-error">{action.prediction.detail}</p>}
        <div className="ctx-checks">
          <Chip tone={action.prediction.trajectory_checked ? 'green' : 'amber'}>
            {action.prediction.trajectory_checked ? '路径已检查' : '路径未检查'}
          </Chip>
          <Chip tone={action.prediction.collision_checked ? 'green' : 'amber'}>
            {action.prediction.collision_checked ? '碰撞已检查' : '碰撞未检查'}
          </Chip>
        </div>
        {alternatives.length > 0 && <div className="ctx-alternative-rail"><small>可重新选择的候选</small><div>{alternatives.map((item) => <CandidateCard snapshot={snapshot} item={item} compact key={item.id} />)}</div></div>}
      </aside>
    </div>
  )
}

function ReceiptWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const receipt = snapshot.world.lastReceipt ?? {}
  const event = snapshot.world.latestEvent
  const functionName = String(receipt.function_name ?? event?.function_name ?? 'physical_action').toUpperCase()
  const actionId = typeof receipt.action_id === 'string' ? receipt.action_id : null
  const positionError = typeof receipt.position_error_m === 'number'
    ? `TCP error ${(receipt.position_error_m * 1000).toFixed(1)} mm`
    : null
  const opening = typeof receipt.gripper_opening === 'number'
    ? `opening ${receipt.gripper_opening.toFixed(3)}`
    : null
  return (
    <div className="ctx-receipt-mode">
      <div className="ctx-receipt-strip">
        <strong>{functionName}{actionId ? ` ${actionId}` : ''}</strong>
        {positionError && <span>{positionError}</span>}
        {opening && <span>{opening}</span>}
        <b>TASK EFFECT UNVERIFIED</b>
      </div>
      <div className="ctx-receipt-review">
        <div className="ctx-receipt-raster"><Raster snapshot={snapshot} id={snapshot.decision.primaryRasterId} alt="动作后目标区域" /></div>
        <div className="ctx-receipt-cue"><small>POST-ACTION CHECK</small><strong>CURRENT RGB</strong><span>LAST ACTION AREA</span></div>
      </div>
    </div>
  )
}

function ErrorWorkspace({ snapshot, message }: { snapshot: ContextSnapshot; message?: string }) {
  const event = snapshot.world.latestEvent
  const error = message ?? String(event?.result.error ?? 'unknown error')
  return (
    <div className="ctx-status-mode ctx-status-mode--error">
      <section><small>FUNCTION ERROR</small><h3>{event?.function_name ?? 'Context compiler'}</h3><p>{error}</p><code>arguments {compactJson(event?.arguments, 360)}</code></section>
      <section><small>CURRENT RGB</small><h3>RECOVERY</h3>{snapshot.world.activeAction && <Chip tone="blue">ACTIVE {snapshot.world.activeAction.action_id}</Chip>}</section>
    </div>
  )
}

function QuietWorkspace({ snapshot, terminal = false }: { snapshot: ContextSnapshot; terminal?: boolean }) {
  const claim = snapshot.world.latestEvent?.arguments.success
  return (
    <div className="ctx-status-mode ctx-status-mode--quiet">
      <section><small>{terminal ? 'AGENT TERMINAL CLAIM' : 'CURRENT DECISION STATE'}</small><h3>{terminal ? `done(success=${String(claim)})` : '等待下一次 Function Call'}</h3></section>
      <section><small>CURRENT RGB</small><h3>观察当前场景</h3></section>
    </div>
  )
}

function DynamicWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const mode = snapshot.decision.mode
  return (
    <section className={`ctx-section ctx-decision ctx-decision--${mode}`}>
      <CompactRobot snapshot={snapshot} />
      <div className="ctx-decision-body">
        {mode === 'grounding' && <GroundingWorkspace snapshot={snapshot} />}
        {mode === 'candidates' && <CandidatesWorkspace snapshot={snapshot} />}
        {mode === 'proposal' && <ProposalWorkspace snapshot={snapshot} />}
        {mode === 'receipt' && <ReceiptWorkspace snapshot={snapshot} />}
        {mode === 'error' && <ErrorWorkspace snapshot={snapshot} />}
        {mode === 'idle' && <QuietWorkspace snapshot={snapshot} />}
        {mode === 'terminal' && <QuietWorkspace snapshot={snapshot} terminal />}
      </div>
    </section>
  )
}

export function ContextApp({ snapshot }: { snapshot: ContextSnapshot }) {
  const mode = snapshot.decision.mode
  return (
    <main className={`context-workspace context-workspace--${mode}`}>
      <PersistentWorld snapshot={snapshot} />
      <DynamicWorkspace snapshot={snapshot} />
    </main>
  )
}
