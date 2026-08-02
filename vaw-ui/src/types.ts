export type IkStatus = 'unchecked' | 'pass' | 'fail'

export interface CandidateSnapshot {
  id: string
  kind: string
  objectId: string | null
  selected: boolean
  stale: boolean
  ik: IkStatus
  image: string | null
}

export interface WorkspaceSnapshot {
  schemaVersion: 1
  renderId: string
  viewport: { width: 1024; height: 576 }
  header: {
    task: string
    revision: number
    view: string
    source: string
    selectedId: string | null
  }
  scene: {
    image: string | null
    objectCount: number
    candidateCount: number
  }
  focus: {
    objectId: string
    name: string
    image: string
    stale: boolean
    hasMask: boolean
    hasObb: boolean
    revision: number
  } | null
  self: {
    wristImage: string | null
    currentImage: string | null
    gripperState: 'open' | 'closed' | 'partial' | 'unknown'
    opening: number | null
    revision: number
    jointsObserved: boolean
  }
  intent: {
    selectedId: string | null
    kind: string | null
    nextImage: string | null
    ik: IkStatus
    trajectory: 'not_checked'
    collision: 'not_checked'
  }
  candidates: CandidateSnapshot[]
  receipt: {
    id: string
    op: string
    candidateId: string | null
    unpredictedFailure: boolean
    hasDiscrepancy: boolean
  } | null
}

declare global {
  interface Window {
    __VAW_RENDER__?: (snapshot: WorkspaceSnapshot) => Promise<void>
  }
}
