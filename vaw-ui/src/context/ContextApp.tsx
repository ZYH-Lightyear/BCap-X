import type { ReactNode } from 'react'

import type { ContextSnapshot } from '../types'

type Seed = ContextSnapshot['catalog']['seeds'][number]
type Tone = 'neutral' | 'blue' | 'green' | 'violet' | 'amber' | 'red'

function fmt(values: number[] | undefined, digits = 4): string {
  if (!values) return '不可用'
  return `[${values.map((value) => value.toFixed(digits)).join(', ')}]`
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
  const imagining = snapshot.world.owner === 'imagination'
  return (
    <section className="ctx-section ctx-world">
      <div className="ctx-world-grid">
        <article className="ctx-main-view">
          <div className="ctx-view-label ctx-view-label--observed"><strong>OBSERVED NOW · AGENTVIEW</strong><span>REAL RGB</span></div>
          <Raster snapshot={snapshot} id={snapshot.world.agentviewRasterId} alt="当前主视角" />
        </article>
        <article className="ctx-near-field-view">
          <div className={`ctx-view-label ${imagining ? 'ctx-view-label--preview' : 'ctx-view-label--observed'}`}>
            <strong>{imagining ? 'CURRENT + PREVIEW · GRIPPER LOCAL' : 'OBSERVED NOW · GRIPPER LOCAL'}</strong>
            <span>{imagining ? 'CURRENT RGB-D + VIRTUAL FK' : 'REAL FK + CURRENT RGB-D'}</span>
          </div>
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

function CandidateCard({ snapshot, item, compact = false }: {
  snapshot: ContextSnapshot
  item: Seed
  compact?: boolean
}) {
  const tone: Tone = item.solveIk === 'returned' ? 'green' : item.solveIk === 'error' ? 'red' : 'amber'
  return (
    <article className={`ctx-candidate${compact ? ' ctx-candidate--compact' : ''}`}>
      <header><strong>{item.id}</strong><Chip tone={tone}>{item.solveIk.toUpperCase()}</Chip></header>
      <div className="ctx-candidate-raster"><Raster snapshot={snapshot} id={item.rasterId} alt={`候选 ${item.id}`} /></div>
    </article>
  )
}

function CandidatesWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const items = snapshot.decision.seedIds
    .map((id) => snapshot.catalog.seeds.find((item) => item.id === id))
    .filter((item): item is Seed => Boolean(item))
  return (
    <div className="ctx-candidates-mode">
      <div className="ctx-candidate-intro"><strong>{items.length} 个 Action Seeds</strong></div>
      <div
        className="ctx-candidate-row"
        style={{ gridTemplateColumns: `repeat(${Math.max(items.length, 1)}, minmax(0, 360px))` }}
      >
        {items.map((item) => <CandidateCard snapshot={snapshot} item={item} key={item.id} />)}
      </div>
    </div>
  )
}

function ProposalWorkspace({ snapshot }: { snapshot: ContextSnapshot }) {
  const action = snapshot.world.action
  const alternatives = snapshot.decision.seedIds
    .map((id) => snapshot.catalog.seeds.find((item) => item.id === id))
    .filter((item): item is Seed => Boolean(item))
  if (!action) return <ErrorWorkspace snapshot={snapshot} message="缺少 ActionTarget" />
  const adjustment = action.latest_edit
  const editing = action.status === 'editing'
  const reviewReason = action.handoff_reason === 'budget_exhausted'
    ? 'BUDGET EXHAUSTED · MAIN REVIEW REQUIRED'
    : 'IMAGINATION COMPLETE · MAIN REVIEW REQUIRED'
  const pose = action.target.pose
  const prediction = action.prediction
  const planReturned = prediction?.solve_ik === 'returned'
  return (
    <div className="ctx-proposal-mode">
      <div className="ctx-proposal-image">
        <Raster snapshot={snapshot} id={snapshot.decision.primaryRasterId} alt="局部动作想象与整臂概览" />
        <div className={`ctx-preview-banner${planReturned ? '' : ' ctx-preview-banner--error'}`}>
          <strong>{planReturned ? 'IMAGINATION' : 'TARGET ONLY'}</strong>
          <span>{planReturned ? 'VIRTUAL PREVIEW · NOT OBSERVED' : 'NO EXECUTABLE FK PREVIEW'}</span>
        </div>
        <div className="ctx-image-legend"><span className="blue">current / source</span><span className="green">motion</span><span className="violet">imagined hand + arm</span></div>
      </div>
      <aside className="ctx-proposal-facts">
        <div className="ctx-proposal-title">
          <div><small>{editing ? 'IMAGINATION AGENT EDITING' : reviewReason}</small><h3>{editing ? 'EDITING · NOT COMMITTABLE' : `${action.action_id} · AWAITING MAIN DECISION`}</h3></div>
          <Chip tone={planReturned ? 'green' : 'red'}>{planReturned ? 'PLAN RETURNED' : `PLAN ${prediction?.solve_ik?.toUpperCase() ?? 'UNAVAILABLE'}`}</Chip>
        </div>
        {snapshot.world.refinementGoal && <div className="ctx-target-pose"><small>REFINEMENT GOAL</small><p>{snapshot.world.refinementGoal}</p></div>}
        <div className={`ctx-plan-status${planReturned ? '' : ' ctx-plan-status--error'}`}>
          <small>MOTION PREDICTION</small>
          <code>
            solve_ik {prediction?.solve_ik ?? 'unavailable'}<br />
            trajectory {prediction?.trajectory_checked ? 'checked' : 'not checked'} · collision {prediction?.collision_checked ? 'checked' : 'not checked'}
          </code>
          {!planReturned && prediction?.detail && <p>{prediction.detail}</p>}
        </div>
        <div className="ctx-target-pose">
          <small>ACTION TARGET</small>
          <code>
            POS {pose ? fmt(pose.position_xyz, 4) : 'inherit current'}<br />
            QUAT {pose ? fmt(pose.quaternion_xyzw, 4) : 'inherit current'}<br />
            GRIP {action.target.gripper ?? 'inherit current'}
          </code>
        </div>
        {adjustment && <div className="ctx-target-pose">
          <small>LOCAL REFINEMENT · {adjustment.frame.toUpperCase()} FRAME</small>
          <code>
            {adjustment.kind === 'delta_move'
              ? `delta_xyz_m ${fmt(adjustment.delta_xyz_m)}`
              : `rotate ${adjustment.axis} ${adjustment.angle_deg?.toFixed(1)} deg`}
          </code>
        </div>}
        {editing && <div className="ctx-refine-cue">
          <small>REFINE PREVIEW</small>
          <code>delta_move([dx,dy,dz], frame) · rotate(axis, angle_deg, frame)</code>
        </div>}
        {!editing && alternatives.length > 0 && <div className="ctx-alternative-rail"><small>Main Agent 可重新选择的 Action Seeds</small><div>{alternatives.map((item) => <CandidateCard snapshot={snapshot} item={item} compact key={item.id} />)}</div></div>}
      </aside>
    </div>
  )
}

function ErrorWorkspace({ snapshot, message }: { snapshot: ContextSnapshot; message?: string }) {
  const error = message ?? snapshot.world.latestError ?? 'unknown error'
  return (
    <div className="ctx-status-mode ctx-status-mode--error">
      <section><small>FUNCTION ERROR</small><h3>RECOVERY</h3><p>{error}</p></section>
      <section><small>CURRENT RGB</small><h3>重新观察并选择下一步</h3></section>
    </div>
  )
}

function QuietWorkspace({ snapshot, terminal = false }: { snapshot: ContextSnapshot; terminal?: boolean }) {
  return (
    <div className="ctx-status-mode ctx-status-mode--quiet">
      <section><small>{terminal ? 'AGENT TERMINAL CLAIM' : 'CURRENT DECISION STATE'}</small><h3>{terminal ? 'Episode ended by Main Agent' : '等待 Main Agent 决策'}</h3></section>
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
        {mode === 'seeds' && <CandidatesWorkspace snapshot={snapshot} />}
        {(mode === 'editing' || mode === 'reviewed') && <ProposalWorkspace snapshot={snapshot} />}
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
