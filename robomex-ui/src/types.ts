export interface RunSummary {
  run_id: string
  task: string
  profile: string
  status: string
  started_at: number
  updated_at: number
  intent_count: number
}

export interface Artifact {
  artifact_id: string
  kind: string
  path: string
  mime: string
  digest: string
  producer: string
  intent_id: string
  world_revision?: number
  candidate_id?: string
}

export interface AgentSummary {
  agent_run_id: string
  role_id: string
  kind: string
  objective: string
  model: string
  skills: string[]
  capabilities: string[]
  depends_on: string[]
  status: string
  failure_kind: string
  detail: string
  candidate_count: number
  duration_s?: number
}

export interface GraphNode {
  id: string
  label: string
  kind: string
  model: string
  skills: string[]
  status: string
  agent_run_id: string
}

export interface GraphEdge {
  source: string
  target: string
}

export interface Candidate {
  candidate_id: string
  role_id: string
  kind: string
  notes: string
  candidate_digest: string
  intent_id: string
  config_id: string
  gates: {
    checks?: Record<string, string>
    reasons?: Record<string, string>
  }
}

export interface SwarmConfigView {
  config_id: string
  manager: {
    request: unknown
    response: string
    swarm_config: Record<string, unknown>
  }
  nodes: GraphNode[]
  edges: GraphEdge[]
  candidates: Candidate[]
  selection: Record<string, unknown>
  execution: {
    admission: Record<string, unknown>
    receipt: Record<string, unknown>
    observation: Record<string, unknown>
  }
}

export interface IntentView {
  intent_id: string
  instruction: string
  expected_effect: string
  observation_revision?: number
  status: string
  planner: {
    request: unknown
    response: string
    decision: Record<string, unknown>
  }
  configs: SwarmConfigView[]
}

export interface ActivityEvent {
  event_seq: number
  ts: number
  stage: string
  event: string
  intent_id: string
  config_id: string
  candidate_id: string
  summary: string
  status: string
  details: Record<string, unknown>
}

export interface Snapshot {
  schema_version: string
  run: Record<string, unknown>
  live: boolean
  intents: IntentView[]
  agents: Record<string, AgentSummary>
  candidates: Candidate[]
  artifacts: Artifact[]
  latest_observations: Artifact[]
  activity: ActivityEvent[]
  last_event_seq: number
}

export interface AgentDetail {
  agent_run_id: string
  assignment: Record<string, unknown>
  world_view: Record<string, unknown>
  upstream: Record<string, unknown>
  llm_request: unknown
  llm_response: string
  code: string
  stdout: string
  stderr: string
  result: Record<string, unknown>
  artifacts: Artifact[]
  log: string
}
