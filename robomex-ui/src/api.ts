import type { AgentDetail, RunSummary, Snapshot } from './types'

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<T>
}

export function listRuns() {
  return getJson<{ root: string; runs: RunSummary[] }>('/api/v2/runs')
}

export function fetchSnapshot(runId: string) {
  return getJson<Snapshot>(`/api/v2/runs/${encodeURIComponent(runId)}/snapshot`)
}

export function fetchAgent(runId: string, agentRunId: string) {
  return getJson<AgentDetail>(
    `/api/v2/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(agentRunId)}`,
  )
}

export function artifactUrl(runId: string, artifactId: string) {
  return `/api/v2/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`
}

export function streamUrl(runId: string, afterSeq: number) {
  return `/api/v2/runs/${encodeURIComponent(runId)}/stream?after_seq=${afterSeq}`
}
