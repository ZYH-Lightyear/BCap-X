import type { ReactNode } from 'react'

import type { CandidateSnapshot, IkStatus, WorkspaceSnapshot } from './types'

export const EMPTY: WorkspaceSnapshot = {
  schemaVersion: 1,
  renderId: 'empty',
  viewport: { width: 1024, height: 576 },
  header: {
    task: '等待 Visual Action Workspace 快照',
    revision: 0,
    view: 'agentview',
    source: 'physical RGB',
    selectedId: null,
  },
  scene: { image: null, objectCount: 0, candidateCount: 0 },
  focus: null,
  self: {
    wristImage: null,
    currentImage: null,
    gripperState: 'unknown',
    opening: null,
    revision: 0,
    jointsObserved: false,
  },
  intent: {
    selectedId: null,
    kind: null,
    nextImage: null,
    ik: 'unchecked',
    trajectory: 'not_checked',
    collision: 'not_checked',
  },
  candidates: [],
  receipt: null,
}

export function waitForPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(async () => {
        await document.fonts.ready
        const pending = Array.from(document.images)
          .filter((image) => !image.complete)
          .map(
            (image) =>
              new Promise<void>((done) => {
                image.addEventListener('load', () => done(), { once: true })
                image.addEventListener('error', () => done(), { once: true })
              }),
          )
        await Promise.all(pending)
        resolve()
      })
    })
  })
}

function StatusChip({
  tone,
  children,
}: {
  tone: 'blue' | 'green' | 'violet' | 'amber' | 'red' | 'neutral'
  children: ReactNode
}) {
  return <span className={`status status--${tone}`}>{children}</span>
}

function IkChip({ status }: { status: IkStatus }) {
  if (status === 'pass') return <StatusChip tone="green">IK 通过</StatusChip>
  if (status === 'fail') return <StatusChip tone="red">IK 失败</StatusChip>
  return <StatusChip tone="amber">IK 未检查</StatusChip>
}

function EmptyImage({ label }: { label: string }) {
  return (
    <div className="empty-image">
      <span>{label}</span>
    </div>
  )
}

function CandidateCard({ candidate }: { candidate: CandidateSnapshot }) {
  const tone = candidate.stale ? 'stale' : candidate.selected ? 'selected' : 'ready'
  const kind = candidate.kind === 'grasp'
    ? '抓取'
    : candidate.kind === 'waypoint'
      ? '路点'
      : candidate.kind === 'place'
        ? '放置'
        : candidate.kind
  return (
    <article className={`candidate-card candidate-card--${tone}`}>
      <div className="candidate-card__header">
        <strong>{candidate.id}</strong>
        <span>{kind}</span>
        {candidate.objectId && <span className="candidate-card__object">{candidate.objectId}</span>}
      </div>
      {candidate.image ? (
        <img src={candidate.image} alt={`${candidate.id} isolated candidate evidence`} />
      ) : (
        <EmptyImage label="NO VISUAL" />
      )}
      <div className="candidate-card__status">
        {candidate.selected && <StatusChip tone="green">已选择</StatusChip>}
        {candidate.stale ? <StatusChip tone="red">已过期</StatusChip> : <IkChip status={candidate.ik} />}
      </div>
    </article>
  )
}

function AppHeader({ snapshot }: { snapshot: WorkspaceSnapshot }) {
  return (
    <header className="topbar">
      <div className="topbar__brand">
        <span className="brand-mark" aria-hidden="true" />
        <strong>VAW</strong>
      </div>
      <div className="topbar__task">
        <span>任务</span>
        <strong>{snapshot.header.task}</strong>
      </div>
      <div className="topbar__facts">
        <StatusChip tone="blue">版本 R{snapshot.header.revision}</StatusChip>
        <StatusChip tone="neutral">{snapshot.header.view === 'agentview' ? '主视角' : snapshot.header.view.toUpperCase()}</StatusChip>
        <StatusChip tone="neutral">{snapshot.header.source === 'physical RGB' ? '物理 RGB' : snapshot.header.source.toUpperCase()}</StatusChip>
        <StatusChip tone={snapshot.header.selectedId ? 'green' : 'neutral'}>
          {snapshot.header.selectedId ? `已选择 ${snapshot.header.selectedId}` : '未选择'}
        </StatusChip>
      </div>
    </header>
  )
}

function ScenePanel({ snapshot }: { snapshot: WorkspaceSnapshot }) {
  return (
    <section className="scene-panel" aria-label="Global scene evidence">
      {snapshot.scene.image ? (
        <img src={snapshot.scene.image} alt="全局工作空间场景" />
      ) : (
        <EmptyImage label="等待场景证据" />
      )}
      <div className="scene-panel__legend">
        <span><i className="legend-dot legend-dot--blue" />当前机器人</span>
        <span><i className="legend-dot legend-dot--violet" />候选 TCP</span>
        <span><i className="legend-dot legend-dot--green" />已选 TCP</span>
      </div>
      <div className="scene-panel__count">
        {snapshot.scene.objectCount} 个对象 · {snapshot.scene.candidateCount} 个候选
      </div>
    </section>
  )
}

function FocusPanel({ snapshot }: { snapshot: WorkspaceSnapshot }) {
  const focus = snapshot.focus
  return (
    <section className={`card focus-card ${focus?.stale ? 'card--danger' : ''}`}>
      <div className="card__title">
        <span>焦点 · 显式 INSPECT</span>
        {focus && <StatusChip tone={focus.stale ? 'red' : 'blue'}>{focus.stale ? '已过期' : `证据 R${focus.revision}`}</StatusChip>}
      </div>
      {focus ? (
        <>
          <div className="focus-card__identity">
            <strong>{focus.objectId}</strong>
            <span>{focus.name}</span>
            <div>
              {focus.hasMask && <StatusChip tone="blue">掩码</StatusChip>}
              {focus.hasObb && <StatusChip tone="violet">OBB</StatusChip>}
            </div>
          </div>
          <img src={focus.image} alt={`Focused evidence for ${focus.objectId}`} />
        </>
      ) : (
        <div className="focus-empty">
          <strong>尚未请求焦点</strong>
          <span>调用 inspect(object_id) 后显示局部裁剪、轮廓和 OBB 证据。</span>
        </div>
      )}
    </section>
  )
}

function SelfPanel({ snapshot }: { snapshot: WorkspaceSnapshot }) {
  const opening = Math.round((snapshot.self.opening ?? 0) * 100)
  return (
    <section className="card self-card">
      <div className="card__title">
        <span>自身状态 · 观测 R{snapshot.self.revision}</span>
        <StatusChip tone={snapshot.self.gripperState === 'unknown' ? 'neutral' : 'blue'}>
          夹爪 {snapshot.self.gripperState === 'open' ? '张开' : snapshot.self.gripperState === 'closed' ? '闭合' : snapshot.self.gripperState === 'partial' ? '部分闭合' : '未知'}
        </StatusChip>
      </div>
      <div className="self-card__body">
        {snapshot.self.wristImage ? <img src={snapshot.self.wristImage} alt="腕部 RGB" /> : <EmptyImage label="无腕部图像" />}
        {snapshot.self.currentImage ? <img src={snapshot.self.currentImage} alt="当前夹爪证据" /> : <EmptyImage label="无 FK" />}
        <div className="aperture">
          <span>开合度</span>
          <div><i style={{ width: `${opening}%` }} /></div>
          <small>{snapshot.self.jointsObserved ? '关节已观测' : '关节未知'}</small>
        </div>
      </div>
    </section>
  )
}

function IntentPanel({ snapshot }: { snapshot: WorkspaceSnapshot }) {
  return (
    <section className="card intent-card">
      <div className="card__title">
        <span>意图 · 当前 → 下一步</span>
        <IkChip status={snapshot.intent.ik} />
      </div>
      <div className="intent-card__body">
        <div className="intent-frame intent-frame--now">
          <span>当前</span>
          {snapshot.self.currentImage ? <img src={snapshot.self.currentImage} alt="Current gripper" /> : <EmptyImage label="—" />}
        </div>
        <div className="intent-arrow" aria-hidden="true">→</div>
        <div className="intent-frame intent-frame--next">
          <span>下一步 {snapshot.intent.selectedId ?? '—'}</span>
          {snapshot.intent.nextImage ? <img src={snapshot.intent.nextImage} alt="候选下一夹爪位姿" /> : <EmptyImage label="无目标" />}
        </div>
        <div className="intent-checks">
          <StatusChip tone="amber">路径未检查</StatusChip>
          <StatusChip tone="amber">碰撞未检查</StatusChip>
        </div>
      </div>
    </section>
  )
}

function EvidenceRail({ snapshot }: { snapshot: WorkspaceSnapshot }) {
  const slots = Array.from({ length: 5 }, (_, index) => snapshot.candidates[index] ?? null)
  const receipt = snapshot.receipt
  return (
    <section className="evidence-row">
      <div className="candidate-rail" aria-label="Candidate selector evidence">
        {slots.map((candidate, index) =>
          candidate ? <CandidateCard candidate={candidate} key={candidate.id} /> : (
            <article className="candidate-card candidate-card--empty" key={`empty-${index}`}>
              <div className="candidate-card__header"><strong>—</strong><span>空位</span></div>
              <EmptyImage label="无候选" />
              <div className="candidate-card__status"><StatusChip tone="neutral">未绑定</StatusChip></div>
            </article>
          ),
        )}
      </div>
      <article className={`receipt-card ${receipt?.unpredictedFailure ? 'receipt-card--danger' : ''}`}>
        <div className="card__title">
          <span>最近物理执行回执</span>
          {receipt && <StatusChip tone={receipt.unpredictedFailure ? 'red' : 'green'}>{receipt.id}</StatusChip>}
        </div>
        {receipt ? (
          <div className="receipt-card__body">
            <strong>{receipt.op.toUpperCase()}</strong>
            <span>{receipt.candidateId ? `目标 ${receipt.candidateId}` : '无候选目标'}</span>
            <div>
              <StatusChip tone={receipt.hasDiscrepancy ? 'amber' : 'green'}>
                {receipt.hasDiscrepancy ? '已记录偏差' : '未发现偏差'}
              </StatusChip>
              {receipt.unpredictedFailure && <StatusChip tone="red">非预期失败</StatusChip>}
            </div>
            <small>精确数值保留在 state_summary 中。</small>
          </div>
        ) : (
          <div className="receipt-empty">
            <strong>尚无物理操作</strong>
            <span>执行 commit、移动或夹爪控制后才会生成回执。</span>
          </div>
        )}
      </article>
    </section>
  )
}

export function App({ snapshot = EMPTY }: { snapshot?: WorkspaceSnapshot }) {
  return (
    <main className="workspace">
      <AppHeader snapshot={snapshot} />
      <div className="workspace__middle">
        <ScenePanel snapshot={snapshot} />
        <aside className="context-stack">
          <FocusPanel snapshot={snapshot} />
          <SelfPanel snapshot={snapshot} />
          <IntentPanel snapshot={snapshot} />
        </aside>
      </div>
      <EvidenceRail snapshot={snapshot} />
    </main>
  )
}
