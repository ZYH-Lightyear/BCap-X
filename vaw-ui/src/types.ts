export interface VisualEditSummary {
  kind: 'delta_move' | 'rotate'
  frame: 'base' | 'tool'
  delta_xyz_m?: number[]
  axis?: 'x' | 'y' | 'z'
  angle_deg?: number
}

export interface ContextSnapshot {
  schemaVersion: 46
  schema: 'vaw-context-v46-oblique-contact'
  projection: 'main' | 'imagination'
  renderId: string
  revision: number
  viewport: { width: 2048; height: 1280 }
  rasterIds: string[]
  rasters: Record<string, string>
  world: {
    agentviewRasterId: string
    observedSceneRasterId: string
    imaginationSceneRasterId: string
    contactFrontRasterId: string | null
    contactSideRasterId: string | null
    contactSideElevationDeg: number
    refinementGoal: string | null
    robot: {
      ee_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      tcp_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      joint_positions_rad?: number[]
      gripper_opening?: number
    } | null
    action: {
      status: 'refining' | 'planned' | 'refined'
      action_id?: string
      intent?: string
      target_role: 'grasp_contact' | 'point_pose' | 'relative_pose'
      target: {
        pose: { position_xyz: number[]; quaternion_xyzw: number[] }
      }
      prediction?: {
        solve_ik: 'returned' | 'error' | 'unavailable'
        trajectory_checked: boolean
        collision_checked: boolean
        detail?: string
      }
      latest_edit?: VisualEditSummary
      rotation_gizmo_frame?: 'base' | 'tool'
      rotation_gizmo_axis?: 'x' | 'y' | 'z'
      edit_summary?: {
        total_translation_base_m?: number[]
        total_rotation_axis_base?: number[]
        total_rotation_deg?: number
        previous_edit?: VisualEditSummary
        last_edit?: VisualEditSummary
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
      status?: 'occluded'
    }>
    points: Array<{
      id: string
      query: string
      pixel: number[]
      position: number[]
      sourceRevision: number
      rasterId: string
      withinRegionId?: string
      status?: 'occluded'
    }>
    seeds: Array<{
      id: string
      targetPose: { position_xyz: number[]; quaternion_xyzw: number[] }
      deltaFromAnchor: number[] | null
      approachVector: number[] | null
      solveIk: 'returned' | 'error' | 'unavailable' | 'mismatch'
      sourceRevision: number
      rasterId: string
      family?: 'top' | 'pca' | 'cgn'
    }>
  }
  decision: {
    mode: 'idle' | 'grounding' | 'seeds' | 'editing' | 'proposal' | 'contact' | 'terminal'
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
