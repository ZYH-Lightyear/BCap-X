export interface VisualEditSummary {
  kind: 'delta_move' | 'rotate'
  frame: 'base' | 'tool'
  delta_xyz_m?: number[]
  axis?: 'x' | 'y' | 'z'
  angle_deg?: number
}

export interface ContextSnapshot {
  schemaVersion: 32
  schema: 'vaw-context-v31-separated-control-guides'
  renderId: string
  revision: number
  viewport: { width: 2048; height: 1280 }
  rasterIds: string[]
  rasters: Record<string, string>
  world: {
    agentviewRasterId: string
    observedSceneRasterId: string
    imaginationSceneRasterId: string
    contactFocusRasterId: string | null
    owner: 'main' | 'imagination'
    refinementGoal: string | null
    latestError: string | null
    lastPhysicalAction: {
      intent: string
      executed_stages: 'arm' | 'gripper'
      outcome: 'completed' | 'arm_failed' | 'gripper_failed'
      target_gripper?: 'open' | 'closed'
      requested_arm_delta_base_m?: number[]
    } | null
    postCommitBeforeRasterId: string | null
    postCommitCurrentRasterId: string | null
    causalSourceBeforeRasterId: string | null
    causalSourceCurrentRasterId: string | null
    causalSourceLabel: string | null
    physicalVerification: {
      status: 'unverified'
      kind: 'closure' | 'release' | 'arm_motion'
      evidenceNeeded: string
      ambiguity: string
    } | null
    robot: {
      ee_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      tcp_pose?: { position_xyz: number[]; quaternion_xyzw: number[] }
      joint_positions_rad?: number[]
      gripper_opening?: number
    } | null
    action: {
      status: 'editing' | 'review'
      action_id?: string
      intent?: string
      target_role: 'grasp_contact' | 'point_pose' | 'relative_pose' | 'gripper_only'
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
      latest_edit?: VisualEditSummary
      rotation_gizmo_frame?: 'base' | 'tool'
      rotation_gizmo_axis?: 'x' | 'y' | 'z'
      edit_summary?: {
        target_gripper?: 'open' | 'closed'
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
    mode: 'idle' | 'grounding' | 'seeds' | 'editing' | 'reviewed' | 'post_commit' | 'error' | 'terminal'
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
