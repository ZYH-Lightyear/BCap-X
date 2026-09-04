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

function ObservedLayer({ snapshot }: { snapshot: ContextSnapshot }) {
  return (
    <section className="via-observed">
      <header><h1>PHYSICAL TRANSITION · REAL WORLD</h1></header>
      <div className="via-observed-grid">
        <article className="via-raster-card via-agentview">
          <Raster snapshot={snapshot} id={snapshot.world.agentviewRasterId} alt="当前 LIBERO-PRO 主视角" />
          <span>{`NOW · CURRENT RGB · r${snapshot.revision}`}</span>
        </article>
        <article
          className="via-raster-card via-auxiliary-view"
          aria-label="陡角度辅助接触视图"
        >
          <Raster snapshot={snapshot} id={snapshot.world.contactAuxiliaryRasterId} alt="陡角度辅助接触视图" />
        </article>
      </div>
    </section>
  )
}

function SeedCard({ snapshot, seed }: { snapshot: ContextSnapshot; seed: Seed }) {
  const family = seed.family ? seed.family.toUpperCase() : null
  return (
    <article className="via-seed-card">
      <strong>
        <span>{seed.id}{family ? ` · ${family}` : ''}</span>
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
  const action = snapshot.world.action
  const edit = snapshot.world.action?.latest_edit
  const summary = snapshot.world.action?.edit_summary
  const lastEdit = edit ?? summary?.last_edit
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
    </div>
  )
}

function ModeOverlay({ snapshot }: { snapshot: ContextSnapshot }) {
  const mode = snapshot.decision.mode
  if (mode === 'seeds') return null
  if (mode === 'grounding' || snapshot.decision.primaryRasterId) {
    return <GroundingOverlay snapshot={snapshot} />
  }
  if (mode === 'editing' || mode === 'proposal') return <EditOverlay snapshot={snapshot} />
  return null
}

function SkillReferenceBoard({ snapshot }: { snapshot: ContextSnapshot }) {
  const board = snapshot.skillReference
  if (!board || board.references.length === 0) return null
  return (
    <aside className="via-skill-references" aria-label="历史技能视觉参考">
      <header>REFERENCE · NOT CURRENT · {board.skillId}</header>
      <div>
        {board.references.map((reference) => (
          <article key={reference.referenceId}>
            <Raster snapshot={snapshot} id={reference.rasterId} alt={`历史技能参考 ${reference.referenceId}`} />
            <p><strong>{reference.state}</strong> · {reference.visualCue}</p>
          </article>
        ))}
      </div>
    </aside>
  )
}

function decisionHeader(snapshot: ContextSnapshot, seedHeader: string): string {
  switch (snapshot.decision.mode) {
    case 'seeds': return seedHeader
    case 'editing': return 'IMAGINATION · NOT EXECUTED'
    case 'proposal': return snapshot.world.action?.status === 'refined'
      ? 'REFINED ACTION · MAIN REVIEW'
      : 'PLANNED ACTION · MAIN REVIEW'
    case 'grounding': return 'CURRENT EVIDENCE · OBSERVED'
    case 'contact': return 'CURRENT CONTACT · REAL WORLD'
    case 'terminal': return 'FINAL OBSERVATION · REAL WORLD'
    default: return 'CURRENT GEOMETRY · OBSERVED'
  }
}

function ImaginationLayer({ snapshot }: { snapshot: ContextSnapshot }) {
  const active = snapshot.world.action !== null
  const selecting = snapshot.decision.mode === 'seeds'
  const hasSkillReference = Boolean(snapshot.skillReference?.references.length)
  const hasContactFocus = (
    snapshot.world.contactFrontRasterId !== null
    && snapshot.world.contactSideRasterId !== null
  )
  const seedHeader = 'ACTION SEEDS · VIRTUAL OPTIONS'
  const header = decisionHeader(snapshot, seedHeader)
  return (
    <section className={`via-imagination${active ? ' via-imagination--active' : ''}${selecting ? ' via-imagination--seeds' : ''}${hasContactFocus ? ' via-imagination--contact' : ''}`}>
      <header><h1>{header}</h1></header>
      <div className={`via-imagination-stage${hasSkillReference ? ' via-imagination-stage--reference' : ''}`}>
        <div className="via-decision-current">
          {selecting
            ? <SeedGallery snapshot={snapshot} />
            : hasContactFocus
              ? <div className="via-imagination-visuals">
                  <div className="via-contact-panel via-contact-panel--front">
                    <Raster snapshot={snapshot} id={snapshot.world.contactFrontRasterId} alt="目标夹爪正面接触视图" />
                  </div>
                  <div className="via-contact-panel via-contact-panel--side">
                    <Raster snapshot={snapshot} id={snapshot.world.contactSideRasterId} alt="目标夹爪侧面接触视图" />
                  </div>
                </div>
              : <Raster snapshot={snapshot} id={snapshot.world.imaginationSceneRasterId} alt="当前点云上的虚拟 Waypoint" />}
          {!selecting && <ModeOverlay snapshot={snapshot} />}
        </div>
        <SkillReferenceBoard snapshot={snapshot} />
      </div>
    </section>
  )
}

export function ContextApp({ snapshot }: { snapshot: ContextSnapshot }) {
  if (snapshot.projection === 'imagination') {
    return (
      <main className="via-canvas via-canvas--focused">
        <ImaginationLayer snapshot={snapshot} />
      </main>
    )
  }
  return (
    <main className="via-canvas">
      <ObservedLayer snapshot={snapshot} />
      <ImaginationLayer snapshot={snapshot} />
    </main>
  )
}
