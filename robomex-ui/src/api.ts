import type {
  AgentDetail,
  RunDetail,
  RunSummary,
  SwarmView,
  TurnDetail,
} from './types'

async function getJson<T>(url: string): Promise<T> {
  const resp = await fetch(url)
  if (!resp.ok) {
    throw new Error(await resp.text())
  }
  return resp.json() as Promise<T>
}

export function listRuns(root: string) {
  return getJson<{ root: string; runs: RunSummary[] }>(
    `/api/v1/runs?root=${encodeURIComponent(root)}`,
  )
}

export function fetchRun(dir: string) {
  return getJson<RunDetail>(`/api/v1/runs/by-path?dir=${encodeURIComponent(dir)}`)
}

export function fetchSwarm(dir: string, subgoal: number) {
  return getJson<SwarmView>(
    `/api/v1/runs/by-path/swarm?dir=${encodeURIComponent(dir)}&subgoal=${subgoal}`,
  )
}

export function fetchAgent(dir: string, subgoal: number, agent: string) {
  return getJson<AgentDetail>(
    `/api/v1/runs/by-path/agent?dir=${encodeURIComponent(dir)}&subgoal=${subgoal}&agent=${encodeURIComponent(agent)}`,
  )
}

export function fetchTurn(dir: string, subgoal: number, agent: string, turn: number) {
  return getJson<TurnDetail>(
    `/api/v1/runs/by-path/turn?dir=${encodeURIComponent(dir)}&subgoal=${subgoal}&agent=${encodeURIComponent(agent)}&turn=${turn}`,
  )
}

export function fetchLlm(dir: string, path: string, view: 'text' | 'meta' | 'full' = 'text') {
  return getJson<{ path: string; view: string; size: number; content: unknown }>(
    `/api/v1/runs/by-path/llm?dir=${encodeURIComponent(dir)}&path=${encodeURIComponent(path)}&view=${view}`,
  )
}

export function liveUrl(dir: string, fromStart = false) {
  return `/api/v1/runs/by-path/live?dir=${encodeURIComponent(dir)}&from_start=${fromStart ? 'true' : 'false'}`
}
