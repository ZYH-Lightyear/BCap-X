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

export interface ContextSnapshot {
  schemaVersion: 3
  schema: 'vaw-context-v2'
  renderId: string
  revision: number
  viewport: { width: 1440; height: 1080 }
  rasterIds: string[]
  rasters: Record<string, string>
  world: {
    taskPrompt: string
    agentviewRasterId: string
    wristRasterId: string | null
    robot: {
      source_revision: number
      ee_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      tcp_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      joint_positions_rad?: number[]
      gripper_opening?: number
    } | null
    activeAction: {
      action_id: string
      kind: string
      source_ref?: string
      source_revision: number
      target_pose: { position_xyz: number[]; quaternion_xyzw: number[] }
      prediction: {
        solve_ik: 'returned' | 'error' | 'unavailable'
        trajectory_checked: boolean
        collision_checked: boolean
        detail?: string
      }
      adjustment?: {
        kind: 'delta_move' | 'rotate'
        frame: 'base' | 'tool'
        reference_pose: { position_xyz: number[]; quaternion_xyzw: number[] }
        parent_action_id?: string
        delta_xyz_m?: number[]
        axis?: 'x' | 'y' | 'z'
        angle_deg?: number
      }
    } | null
    latestEvent: {
      function_name: string
      arguments: Record<string, unknown>
      result: Record<string, unknown>
      revision_before: number
      revision_after: number
      ok: boolean
      action_id?: string
    } | null
    lastReceipt: Record<string, unknown> | null
  }
  catalog: {
    regions: Array<{
      id: string
      query: string
      bbox: number[]
      sourceRevision: number
      rasterId: string
      withinRegionId?: string
    }>
    points: Array<{
      id: string
      query: string
      pixel: number[]
      position: number[]
      sourceRevision: number
      rasterId: string
      withinRegionId?: string
    }>
    candidates: Array<{
      id: string
      kind: string
      sourceRef?: string
      targetPose: {
        position_xyz: number[]
        quaternion_xyzw: number[]
      }
      deltaFromAnchor: number[] | null
      approachVector: number[] | null
      solveIk: 'returned' | 'error' | 'unavailable'
      sourceRevision: number
      rasterId: string
    }>
  }
  decision: {
    mode: 'idle' | 'grounding' | 'candidates' | 'proposal' | 'receipt' | 'error' | 'terminal'
    regionIds: string[]
    pointIds: string[]
    candidateIds: string[]
    actionId: string | null
    primaryRasterId: string | null
  }
}

export type VawSnapshot = WorkspaceSnapshot | ContextSnapshot

declare global {
  interface Window {
    __VAW_RENDER__?: (snapshot: VawSnapshot) => Promise<void>
  }
}
