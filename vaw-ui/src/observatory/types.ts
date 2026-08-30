export interface InteractionEvent {
  turn: number
  kind: 'call' | 'action'
  function: string
  arguments: Record<string, unknown>
  outcome: string
  revision_before: number
  revision_after: number
}

export interface ContextView {
  schema: string
  turn: number
  task: string
  revision: number
  canvas: string
  live_references: Record<string, unknown>
  embodied_state_card: {
    tcp_pose?: { position_xyz: number[]; quaternion_xyzw: number[] } | null
    gripper_opening?: number | null
    last_action?: Record<string, unknown> | null
    last_gripper_action?: Record<string, unknown> | null
    active_action?: Record<string, unknown> | null
  }
  interaction_memory_before: InteractionEvent[]
  interaction_memory_prompt: string[]
  protocol_feedback?: string | null
}

export interface ActionStream {
  path: string
  frames: number
  fps: number
  duration_s: number
}

export interface EpisodeVideo {
  path: string
  frames?: number
  fps?: number
  duration_s?: number
}

export interface ActionSegment {
  segment_id: string
  turn: number
  function: string
  arguments: Record<string, unknown>
  outcome: string
  revision_before: number
  revision_after: number
  status: string
  poster?: string | null
  streams: Record<string, ActionStream>
}

export interface TurnView {
  turn: number
  status: string
  context_available: boolean
  context: ContextView | null
  canvas: string | null
  decision: {
    basis?: string | null
    function?: string | null
    arguments?: Record<string, unknown> | null
    effect_kind?: 'call' | 'action' | null
    result?: Record<string, unknown> | null
    error?: string | null
    advisory?: string | null
    revision_before?: number | null
    revision_after?: number | null
  }
  interaction_event?: InteractionEvent | null
  action_segment?: ActionSegment | null
  model_io_available?: boolean
  imagination?: ImaginationSummary | null
  event_seq: number
}

export interface ImaginationSummary {
  session_id: string
  instruction?: string | null
  status: string
  action_id?: string | null
  reason?: string | null
  turn_count: number
}

export interface ImaginationTurnView {
  turn: number
  canvas?: string | null
  decision_basis?: string | null
  function_call?: { name?: string; arguments?: Record<string, unknown> } | null
  function_result?: Record<string, unknown> | null
  runtime_diagnostics?: Record<string, unknown> | null
  model_io_available?: boolean
}

export interface ImaginationSessionView {
  schema: 'vaw-observatory-imagination-v1'
  session_id: string
  instruction?: string | null
  status: string
  action_id?: string | null
  reason?: string | null
  turns: ImaginationTurnView[]
}

export interface ModelIOAttempt {
  attempt: number
  request: Record<string, unknown> | null
  response: Record<string, unknown> | null
}

export interface ModelIOView {
  schema: 'vaw-observatory-model-io-v1'
  turn: number
  legacy: boolean
  attempts: ModelIOAttempt[]
}

export interface RunSummary {
  run_id: string
  run_name: string
  display_name: string
  relative_path: string
  collection: string
  task?: string | null
  suite?: string | null
  task_id?: number | null
  seed?: number | null
  model?: string | null
  turns: number
  revision?: number | null
  status: string
  live: boolean
  env_success?: boolean | null
  job_id?: string | null
  launcher_status?: string | null
  launcher_log?: string | null
  latest_event_seq: number
  elapsed_s?: number | null
}

export interface ObservatorySnapshot {
  schema: 'vaw-observatory-snapshot-v1'
  run: RunSummary & { profile?: Record<string, unknown> }
  turns: TurnView[]
  action_segments: ActionSegment[]
  episode_videos: Record<string, EpisodeVideo>
  latest_event_seq: number
  closed: boolean
}

export interface ControlConfig {
  control_enabled: boolean
  workspace: string
  default_collection: string
  default_model: string
  default_suite: string
  collections: string[]
  model_suggestions: string[]
  suite_suggestions: string[]
  max_concurrent: number
}

export interface LaunchFormState {
  run_name: string
  suite: string
  task_id: number
  seed: number
  model: string
  imagination_model: string
  protocol: 'native' | 'text'
  motion_backend: 'pyroki' | 'curobo'
  max_turns: number
  max_time_s: number
  max_physical_ops: number
  collection: string
}
