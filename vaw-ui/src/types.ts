export interface ContextSnapshot {
  schemaVersion: 11
  schema: 'vaw-context-v10-via-dense'
  renderId: string
  revision: number
  viewport: { width: 1920; height: 1080 }
  rasterIds: string[]
  rasters: Record<string, string>
  world: {
    agentviewRasterId: string
    observedSceneRasterId: string
    imaginationSceneRasterId: string
    owner: 'main' | 'imagination'
    refinementGoal: string | null
    latestError: string | null
    robot: {
      ee_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      tcp_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      joint_positions_rad?: number[]
      gripper_opening?: number
    } | null
    action: {
      status: 'editing' | 'review'
      action_id?: string
      handoff_reason?: 'completed' | 'budget_exhausted'
      target: {
        pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
        gripper?: 'open' | 'closed'
      }
      prediction?: {
        solve_ik: 'returned' | 'error' | 'unavailable'
        trajectory_checked: boolean
        collision_checked: boolean
        detail?: string
      }
      latest_edit?: {
        kind: 'delta_move' | 'rotate'
        frame: 'base' | 'tool'
        delta_xyz_m?: number[]
        axis?: 'x' | 'y' | 'z'
        angle_deg?: number
      }
    } | null
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
    seeds: Array<{
      id: string
      targetPose: { position_xyz: number[]; quaternion_xyzw: number[] }
      deltaFromAnchor: number[] | null
      approachVector: number[] | null
      solveIk: 'returned' | 'error' | 'unavailable' | 'mismatch'
      sourceRevision: number
      rasterId: string
    }>
  }
  decision: {
    mode: 'idle' | 'grounding' | 'seeds' | 'editing' | 'reviewed' | 'error' | 'terminal'
    regionIds: string[]
    pointIds: string[]
    seedIds: string[]
    actionId: string | null
    primaryRasterId: string | null
  }
}

declare global {
  interface Window {
    __VAW_RENDER__?: (snapshot: ContextSnapshot) => Promise<void>
  }
}
