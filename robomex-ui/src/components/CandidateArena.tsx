import { artifactUrl } from '../api'
import type { Artifact, SwarmConfigView } from '../types'
import { pretty } from '../App'

export function CandidateArena({
  runId,
  config,
  artifacts,
}: {
  runId: string
  config?: SwarmConfigView
  artifacts: Artifact[]
}) {
  const selected = String(config?.selection.selected_candidate_id || '')
  const media = artifacts.filter(
    (artifact) => artifact.candidate_id || artifact.intent_id === config?.manager.swarm_config.intent_id,
  )
  return (
    <div className="arena panel">
      <div className="panel-title">
        <span>Candidate Arena &amp; Action Evidence</span>
        <small>immutable proposals · read-only imagination</small>
      </div>
      <div className="candidate-strip">
        {config?.candidates.length ? config.candidates.map((candidate) => (
          <article key={candidate.candidate_id} className={candidate.candidate_id === selected ? 'selected' : ''}>
            <header><b>{candidate.kind}</b><em>{candidate.candidate_id === selected ? 'SELECTED' : 'CANDIDATE'}</em></header>
            <code>{candidate.candidate_id}</code>
            <p>{candidate.notes || 'No agent note.'}</p>
            <div className="gate-list">
              {Object.entries(candidate.gates.checks || {}).map(([gate, status]) => (
                <span key={gate} className={String(status)}>{gate}: {status}</span>
              ))}
            </div>
            <div className="candidate-media">
              {media.filter((item) => !item.candidate_id || item.candidate_id === candidate.candidate_id).slice(0, 3).map((item) =>
                item.mime.startsWith('image/') ? (
                  <img key={item.artifact_id} src={artifactUrl(runId, item.artifact_id)} alt={item.kind} />
                ) : item.mime.startsWith('video/') ? (
                  <video key={item.artifact_id} src={artifactUrl(runId, item.artifact_id)} controls />
                ) : null,
              )}
            </div>
          </article>
        )) : <div className="empty">No candidate has reached the arena.</div>}
        <article className="decision-card">
          <header><b>Selector</b></header>
          <pre>{pretty(config?.selection || {})}</pre>
        </article>
        <article className="execution-card">
          <header><b>Physical action</b></header>
          <label>Admission</label><pre>{pretty(config?.execution.admission || {})}</pre>
          <label>Receipt</label><pre>{pretty(config?.execution.receipt || {})}</pre>
        </article>
      </div>
    </div>
  )
}
