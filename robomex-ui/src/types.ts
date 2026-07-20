export interface RunSummary {
  path: string
  name: string
  mtime: number
  task: string
  planner_status: string
  env_success: boolean | null
  n_subgoals: number
  authoring_strategy: string
}

export interface SubgoalIndex {
  index: number
  goal: string
  postcondition: string
  authoring_status: string
  verification_status: string
  success: boolean | null
  motion_attempted: boolean | null
  note: string
  inner_turns: number
  loaded_skill_ids: string[]
  has_swarm: boolean
}

export interface RunDetail {
  path: string
  summary: Record<string, unknown>
  subgoals: SubgoalIndex[]
  media: string[]
}

export interface GraphNode {
  id: string
  role: string
  specialist_skill: string
  objective: string
  status: string
  verifier: boolean
  changes_world: boolean
}

export interface GraphEdge {
  source: string
  target: string
  on: string
}

export interface AgentInstance {
  dir: string
  node_id: string
  role: string
  status: string
  attempt: number | null
  turn_count: number
  has_result: boolean
  media: string[]
}

export interface ManagerTurn {
  turn: number
  request_path: string
  response_path: string
  response_preview: string
}

export interface MediaItem {
  path: string
  kind: string
  url: string
}

export interface SwarmView {
  run_dir: string
  subgoal_index: number
  task_skill: string
  entry: string
  success_node: string
  outcome_status: string
  verification: string
  note: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  agents: AgentInstance[]
  manager_turns: ManagerTurn[]
  media: MediaItem[]
  artifact_keys: string[]
}

export interface TurnIndex {
  turn: number
  has_code: boolean
  has_out: boolean
  has_llm: boolean
  code_path: string
  out_path: string
  llm_request_path: string
  llm_response_path: string
  llm_response_txt_path: string
}

export interface AgentDetail {
  run_dir: string
  subgoal_index: number
  agent_dir: string
  node_id: string
  role: string
  status: string
  request: Record<string, unknown>
  result: Record<string, unknown>
  turns: TurnIndex[]
  media: MediaItem[]
}

export interface TurnDetail {
  run_dir: string
  subgoal_index: number
  agent_dir: string
  turn: number
  code: string
  out: string
  code_path: string
  out_path: string
  llm_request_path: string
  llm_response_path: string
  llm_response_txt_path: string
  response_preview: string
}

export interface LiveEvent {
  id?: string
  ts?: string
  event: string
  message?: string
  subgoal_index?: number
  agent_label?: string
  agent_role?: string
  node_id?: string
  turn?: number
  action?: string
  status?: string
  ok?: boolean
  goal?: string
  code?: string
  raw?: string
  [key: string]: unknown
}
