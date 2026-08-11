import type { ReactNode } from 'react'

import type { ContextSnapshot } from '../types'

type Seed = ContextSnapshot['catalog']['seeds'][number]

function fmt(values: number[] | undefined, digits = 3): string {
  if (!values) return 'N/A'
  return `[${values.map((value) => value.toFixed(digits)).join(', ')}]`
}

function Raster({ snapshot, id, alt }: {
  snapshot: ContextSnapshot
  id: string | null
  alt: string
}) {
  const src = id ? snapshot.rasters[id] : null
  return src ? <img src={src} alt={alt} /> : <div className="via-no-raster">NO VISUAL</div>
}

function FactRow({ label, children }: { label: string; children: ReactNode }) {
  return <div className="via-fact-row"><strong>{label}</strong><code>{children}</code></div>
}

function RobotStatePanel({ snapshot }: { snapshot: ContextSnapshot }) {
  const robot = snapshot.world.robot
  const tcp = robot?.tcp_pose ?? robot?.ee_pose
  return (
    <aside className="via-robot-state">
      <FactRow label="GRIP">{robot?.gripper_opening?.toFixed(3) ?? 'N/A'}</FactRow>
      <FactRow label="TCP XYZ">{fmt(tcp?.position_xyz)}</FactRow>
      <FactRow label="JOINTS">{fmt(robot?.joint_positions_rad, 2)}</FactRow>
    </aside>
  )
}

function ObservedLayer({ snapshot }: { snapshot: ContextSnapshot }) {
  return (
    <section className="via-observed">
      <header><h1>OBSERVED NOW · REAL WORLD</h1></header>
      <div className="via-observed-grid">
        <article className="via-raster-card via-agentview">
          <Raster snapshot={snapshot} id={snapshot.world.agentviewRasterId} alt="当前 LIBERO-PRO 主视角" />
          <span>AGENTVIEW · CURRENT RGB</span>
        </article>
        <article className="via-raster-card via-observed-cloud">
          <Raster snapshot={snapshot} id={snapshot.world.observedSceneRasterId} alt="当前稠密 RGB-D 场景" />
          <span>DENSE RGB-D · CURRENT FK</span>
        </article>
        <RobotStatePanel snapshot={snapshot} />
      </div>
    </section>
  )
}

function SeedCard({ snapshot, seed }: { snapshot: ContextSnapshot; seed: Seed }) {
  return (
    <article className="via-seed-card">
      <strong>
        <span>{seed.id}</span>
        <code>APPROACH BASE {fmt(seed.approachVector ?? undefined, 2)}</code>
      </strong>
      <Raster snapshot={snapshot} id={seed.rasterId} alt={`Action Seed ${seed.id}`} />
    </article>
  )
}

function SeedGallery({ snapshot }: { snapshot: ContextSnapshot }) {
  const seeds = snapshot.decision.seedIds
    .map((id) => snapshot.catalog.seeds.find((item) => item.id === id))
    .filter((item): item is Seed => Boolean(item))
  return (
    <div className="via-seed-gallery">
      {seeds.map((seed) => <SeedCard key={seed.id} snapshot={snapshot} seed={seed} />)}
    </div>
  )
}

function GroundingOverlay({ snapshot }: { snapshot: ContextSnapshot }) {
  const region = snapshot.decision.regionIds
    .map((id) => snapshot.catalog.regions.find((item) => item.id === id))
    .find(Boolean)
  const point = snapshot.decision.pointIds
    .map((id) => snapshot.catalog.points.find((item) => item.id === id))
    .find(Boolean)
  const item = point ?? region
  if (!item) return null
  const query = item.query
  const metric = 'position' in item ? `BASE XYZ ${fmt(item.position, 4)}` : `BBOX ${fmt(item.bbox, 1)}`
  return (
    <div className="via-grounding-overlay">
      <Raster snapshot={snapshot} id={item.rasterId} alt="当前 Grounding evidence" />
      <div><strong>{item.id}</strong><p>{query}</p><code>{metric}</code></div>
    </div>
  )
}

function EditOverlay({ snapshot }: { snapshot: ContextSnapshot }) {
  const edit = snapshot.world.action?.latest_edit
  const summary = snapshot.world.action?.edit_summary
  const lastEdit = edit ?? summary?.last_edit
  const target = snapshot.world.action?.target
  const cumulativeMove = summary?.total_translation_base_m
  const cumulativeRotation = summary?.total_rotation_deg
  const hasCumulativeMove = cumulativeMove?.some((value) => Math.abs(value) >= 0.0005)
  const hasCumulativeRotation = cumulativeRotation !== undefined
    && Math.abs(cumulativeRotation) >= 0.05
  const move = hasCumulativeMove
    ? `BASE TOTAL ${fmt(cumulativeMove, 3)} m`
    : edit?.kind === 'delta_move'
      ? `${edit.frame.toUpperCase()}  ${fmt(edit.delta_xyz_m, 3)} m`
      : '—'
  const rotate = hasCumulativeRotation
    ? `BASE AXIS ${fmt(summary?.total_rotation_axis_base, 2)}  ${cumulativeRotation?.toFixed(1)}°`
    : edit?.kind === 'rotate'
      ? `${edit.frame.toUpperCase()}-${edit.axis?.toUpperCase()}  ${edit.angle_deg?.toFixed(1)}°`
      : 'NONE'
  const lastRotate = lastEdit?.kind === 'rotate'
    ? `${lastEdit.frame.toUpperCase()}-${lastEdit.axis?.toUpperCase()}  ${lastEdit.angle_deg?.toFixed(1)}°`
    : '—'
  return (
    <div className="via-edit-overlay">
      <FactRow label="MOVE TOTAL">{move}</FactRow>
      <FactRow label="ROTATE LAST">{lastRotate}</FactRow>
      <FactRow label="ROTATE TOTAL">{rotate}</FactRow>
      {target?.pose && <FactRow label="TARGET XYZ">{fmt(target.pose.position_xyz)}</FactRow>}
    </div>
  )
}

function ModeOverlay({ snapshot }: { snapshot: ContextSnapshot }) {
  const mode = snapshot.decision.mode
  if (mode === 'seeds') return null
  if (mode === 'grounding') return <GroundingOverlay snapshot={snapshot} />
  if (mode === 'error') return <div className="via-error-overlay"><strong>FUNCTION ERROR</strong><p>{snapshot.world.latestError}</p></div>
  if (mode === 'editing' || mode === 'reviewed') return <EditOverlay snapshot={snapshot} />
  return null
}

function PhysicalContinuityInset({ snapshot }: { snapshot: ContextSnapshot }) {
  const action = snapshot.world.lastPhysicalAction
  const currentId = snapshot.world.postCommitCurrentRasterId
  if (!action || !currentId) return null
  const facts: string[] = []
  if (action.requested_arm_delta_base_m) {
    facts.push(`ARM Δ ${fmt(action.requested_arm_delta_base_m)} m`)
  }
  if (action.target_gripper) facts.push(`GRIP CMD ${action.target_gripper.toUpperCase()}`)
  return (
    <aside className="via-continuity-inset">
      <div className="via-continuity-raster">
        <Raster snapshot={snapshot} id={currentId} alt="最近 commit 后的当前真实目标区域" />
        <strong>LAST COMMIT · CURRENT OBSERVED</strong>
      </div>
      <div className="via-continuity-facts">
        <b>SAME OBSERVATION</b>
        <code>{facts.join(' · ') || action.executed_stages.toUpperCase()}</code>
      </div>
    </aside>
  )
}

function decisionHeader(snapshot: ContextSnapshot, seedHeader: string): string {
  switch (snapshot.decision.mode) {
    case 'seeds': return seedHeader
    case 'editing':
    case 'reviewed': return 'IMAGINATION · NOT EXECUTED'
    case 'grounding': return 'CURRENT EVIDENCE · OBSERVED'
    case 'error': return 'RECOVERY CONTEXT · OBSERVED'
    case 'terminal': return 'FINAL OBSERVATION · REAL WORLD'
    default: return 'CURRENT GEOMETRY · OBSERVED'
  }
}

function WaypointPanel({ snapshot }: { snapshot: ContextSnapshot }) {
  const action = snapshot.world.action
  const prediction = action?.prediction
  const planned = prediction?.solve_ik === 'returned'
  const observedGrip = snapshot.world.robot?.gripper_opening
  const targetGrip = action?.target.gripper === 'open'
    ? 1
    : action?.target.gripper === 'closed'
      ? 0
      : observedGrip
  const grip = targetGrip === undefined
    ? 'N/A'
    : `${targetGrip.toFixed(3)} ${action?.target.gripper ? 'TARGET' : 'INHERITED'}`
  const goal = snapshot.world.refinementGoal ?? action?.intent
  const lastPhysical = snapshot.world.lastPhysicalAction
  const lastRealParts: string[] = []
  if (lastPhysical?.requested_arm_delta_base_m) {
    lastRealParts.push(`ARM Δ ${fmt(lastPhysical.requested_arm_delta_base_m)}`)
  }
  if (lastPhysical?.target_gripper) {
    lastRealParts.push(`GRIP ${lastPhysical.target_gripper.toUpperCase()}`)
  }
  if (lastPhysical) lastRealParts.push(lastPhysical.outcome.toUpperCase())
  const lastReal = lastRealParts.length > 0 ? lastRealParts.join(' · ') : null
  const targetRole = action?.target_role
    ?.replaceAll('_', ' ')
    .toUpperCase() ?? 'NONE'
  const sourceGap = action?.source_surface_distance_m
  const sourceGapText = sourceGap === undefined
    ? null
    : `${(sourceGap * 1000).toFixed(0)} mm`
  const verification = snapshot.world.physicalVerification
  return (
    <aside className="via-waypoint-panel">
      <h2>WAYPOINT</h2>
      {lastReal && <FactRow label="LAST REAL">{lastReal}</FactRow>}
      <FactRow label="TARGET ROLE">{targetRole}</FactRow>
      {sourceGapText && <FactRow label="TCP→SOURCE">{sourceGapText}</FactRow>}
      <FactRow label="ARM">{action ? (planned ? 'PLANNED' : prediction?.solve_ik?.toUpperCase() ?? 'TARGET ONLY') : 'NONE'}</FactRow>
      <FactRow label="GRIP TARGET">{grip}</FactRow>
      {verification && (
        <FactRow label="LAST EFFECT">{verification.kind.replace('_', ' ').toUpperCase()} · UNVERIFIED</FactRow>
      )}
      {verification && <FactRow label="NEEDS">{verification.evidenceNeeded}</FactRow>}
      {goal && <p className="via-goal">{goal}</p>}
    </aside>
  )
}

function ImaginationLayer({ snapshot }: { snapshot: ContextSnapshot }) {
  const active = snapshot.world.action !== null
  const selecting = snapshot.decision.mode === 'seeds'
  const hasContactFocus = active && snapshot.world.contactFocusRasterId !== null
  const observedGrip = snapshot.world.robot?.gripper_opening
  const seedHeader = observedGrip === undefined
    ? 'ACTION SEEDS · VIRTUAL OPTIONS'
    : `ACTION SEEDS · GRIP ${observedGrip.toFixed(3)} INHERITED`
  const header = decisionHeader(snapshot, seedHeader)
  const showContinuity = !active
    && snapshot.world.postCommitCurrentRasterId !== null
    && ['idle', 'grounding', 'error'].includes(snapshot.decision.mode)
  return (
    <section className={`via-imagination${active ? ' via-imagination--active' : ''}${selecting ? ' via-imagination--seeds' : ''}`}>
      <header><h1>{header}</h1></header>
      <div className="via-imagination-stage">
        {selecting
          ? <SeedGallery snapshot={snapshot} />
          : hasContactFocus
            ? <div className="via-imagination-visuals">
                <div className="via-imagination-global">
                  <Raster snapshot={snapshot} id={snapshot.world.imaginationSceneRasterId} alt="当前点云上的虚拟 Waypoint" />
                  <ModeOverlay snapshot={snapshot} />
                </div>
                <aside className="via-contact-rail">
                  <div className="via-contact-focus">
                    <Raster snapshot={snapshot} id={snapshot.world.contactFocusRasterId} alt="目标夹爪 jaw-plane 接触视图" />
                  </div>
                  <WaypointPanel snapshot={snapshot} />
                </aside>
              </div>
            : <>
                <Raster snapshot={snapshot} id={snapshot.world.imaginationSceneRasterId} alt="当前点云上的虚拟 Waypoint" />
                <ModeOverlay snapshot={snapshot} />
                {showContinuity && <PhysicalContinuityInset snapshot={snapshot} />}
              </>}
      </div>
    </section>
  )
}

function PostCommitLayer({ snapshot }: { snapshot: ContextSnapshot }) {
  const action = snapshot.world.lastPhysicalAction
  const sourceBefore = snapshot.world.causalSourceBeforeRasterId
  const sourceCurrent = snapshot.world.causalSourceCurrentRasterId
  const hasCausalSource = sourceBefore !== null && sourceCurrent !== null
  const sourceLabel = snapshot.world.causalSourceLabel?.toUpperCase() ?? 'LAST GRASP SOURCE'
  return (
    <section className="via-post-commit">
      <header><h1>POST-COMMIT VERIFY · REAL OBSERVATIONS</h1></header>
      <div className={`via-post-commit-grid${hasCausalSource ? ' via-post-commit-grid-source' : ''}`}>
        {hasCausalSource
          ? <>
              <article className="via-compare-card via-compare-before">
                <Raster snapshot={snapshot} id={sourceBefore} alt="最近抓取对象原位置的执行前真实画面" />
                <strong>SOURCE BEFORE · {sourceLabel}</strong>
              </article>
              <div className="via-causal-arrow" aria-hidden="true">→</div>
              <article className="via-compare-card via-compare-current">
                <Raster snapshot={snapshot} id={sourceCurrent} alt="同一固定图像位置的当前真实画面" />
                <strong>FIXED SOURCE CROP · NOW · OCCLUSION POSSIBLE</strong>
              </article>
              <article className="via-compare-card via-action-area-current">
                <Raster
                  snapshot={snapshot}
                  id={snapshot.world.postCommitCurrentRasterId}
                  alt="执行后动作目标附近当前真实画面"
                />
                <strong>CURRENT ACTION AREA</strong>
              </article>
            </>
          : <>
              <article className="via-compare-card via-compare-before">
                <Raster
                  snapshot={snapshot}
                  id={snapshot.world.postCommitBeforeRasterId}
                  alt="执行前目标附近真实画面"
                />
                <strong>BEFORE COMMIT</strong>
              </article>
              <div className="via-causal-arrow" aria-hidden="true">→</div>
              <article className="via-compare-card via-compare-current">
                <Raster
                  snapshot={snapshot}
                  id={snapshot.world.postCommitCurrentRasterId}
                  alt="执行后目标附近当前真实画面"
                />
                <strong>CURRENT OBSERVED</strong>
              </article>
            </>}
        <aside className="via-post-commit-facts">
          <h2>LAST PHYSICAL ACTION</h2>
          <FactRow label="REQUESTED">{action?.intent ?? 'N/A'}</FactRow>
          <FactRow label="EXECUTED">{action?.executed_stages.toUpperCase() ?? 'N/A'}</FactRow>
          {action?.requested_arm_delta_base_m && (
            <FactRow label="ARM CMD Δ">{fmt(action.requested_arm_delta_base_m)} m</FactRow>
          )}
          {action?.target_gripper && (
            <FactRow label="GRIP CMD">{action.target_gripper.toUpperCase()}</FactRow>
          )}
          {action?.outcome !== 'completed' && <FactRow label="CONTROL ERROR">{action?.outcome.toUpperCase() ?? 'N/A'}</FactRow>}
          {snapshot.world.physicalVerification
            ? <p className="via-verification-card">
                <b>{snapshot.world.physicalVerification.kind.replace('_', ' ').toUpperCase()} EFFECT · UNVERIFIED</b>
                <span>NEEDED · {snapshot.world.physicalVerification.evidenceNeeded}</span>
                <small>{snapshot.world.physicalVerification.ambiguity}</small>
              </p>
            : <p>TASK EFFECT · VERIFY FROM CURRENT IMAGE</p>}
        </aside>
      </div>
    </section>
  )
}

export function ContextApp({ snapshot }: { snapshot: ContextSnapshot }) {
  return (
    <main className="via-canvas">
      <ObservedLayer snapshot={snapshot} />
      {snapshot.decision.mode === 'post_commit'
        ? <PostCommitLayer snapshot={snapshot} />
        : <ImaginationLayer snapshot={snapshot} />}
    </main>
  )
}
