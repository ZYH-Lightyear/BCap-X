import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import { useMemo, useState } from 'react'
import { artifactUrl } from '../api'
import type { AgentDetail, AgentSummary } from '../types'
import { pretty } from '../App'

hljs.registerLanguage('python', python)

type Tab = 'overview' | 'code' | 'result' | 'console' | 'llm'

export function AgentInspector({
  summary,
  detail,
  runId,
  onArtifact,
}: {
  summary?: AgentSummary
  detail: AgentDetail | null
  runId: string
  onArtifact: (artifactId: string) => void
}) {
  const [tab, setTab] = useState<Tab>('overview')
  const highlighted = useMemo(
    () => hljs.highlight(detail?.code || '', { language: 'python' }).value,
    [detail?.code],
  )
  if (!summary) {
    return <aside className="inspector panel empty">Select an Agent node to inspect its work.</aside>
  }
  return (
    <aside className="inspector panel">
      <header>
        <div>
          <span>{summary.kind}</span>
          <h2>{summary.role_id}</h2>
        </div>
        <em className={`status ${summary.status}`}>{summary.status}</em>
      </header>
      <p className="agent-objective">{summary.objective}</p>
      <div className="tabs">
        {(['overview', 'code', 'result', 'console', 'llm'] as Tab[]).map((name) => (
          <button key={name} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>
            {name}
          </button>
        ))}
      </div>
      <div className="tab-body">
        {tab === 'overview' && (
          <div className="facts">
            <Fact label="Model" value={summary.model} />
            <Fact label="Duration" value={`${summary.duration_s ?? '—'} s`} />
            <Fact label="Skills" value={summary.skills.join(', ') || 'none'} />
            <Fact label="Depends on" value={summary.depends_on.join(', ') || 'none'} />
            <Fact label="Failure" value={summary.failure_kind || '—'} />
            <details><summary>Assignment</summary><pre>{pretty(detail?.assignment || {})}</pre></details>
            <details><summary>Agent World view</summary><pre>{pretty(detail?.world_view || {})}</pre></details>
            <details><summary>Upstream input</summary><pre>{pretty(detail?.upstream || {})}</pre></details>
          </div>
        )}
        {tab === 'code' && (
          <div className="code-pane">
            <div className="code-toolbar">
              <span>code.py · saved before execution</span>
              <div>
                <button onClick={() => navigator.clipboard.writeText(detail?.code || '')}>Copy</button>
                <button onClick={() => downloadCode(summary.role_id, detail?.code || '')}>Download</button>
              </div>
            </div>
            <pre><code dangerouslySetInnerHTML={{ __html: highlighted }} /></pre>
          </div>
        )}
        {tab === 'result' && (
          <>
            <pre>{pretty(detail?.result || {})}</pre>
            <div className="artifact-grid">
              {detail?.artifacts.map((artifact) => (
                <button key={artifact.artifact_id} onClick={() => onArtifact(artifact.artifact_id)}>
                  {artifact.mime.startsWith('image/') && (
                    <img src={artifactUrl(runId, artifact.artifact_id)} alt="" />
                  )}
                  <strong>{artifact.kind}</strong>
                  <small>{artifact.path.split('/').slice(-1)[0]}</small>
                </button>
              ))}
            </div>
          </>
        )}
        {tab === 'console' && (
          <div className="console-pair">
            <label>stdout</label><pre>{detail?.stdout || '∅'}</pre>
            <label>stderr</label><pre className="stderr">{detail?.stderr || '∅'}</pre>
            <label>agent.log</label><pre>{detail?.log || '∅'}</pre>
          </div>
        )}
        {tab === 'llm' && (
          <>
            <label>Request</label><pre>{pretty(detail?.llm_request || [])}</pre>
            <label>Raw response</label><pre>{detail?.llm_response || '∅'}</pre>
          </>
        )}
      </div>
    </aside>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>
}

function downloadCode(roleId: string, code: string) {
  const url = URL.createObjectURL(new Blob([code], { type: 'text/x-python' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${roleId}.py`
  anchor.click()
  URL.revokeObjectURL(url)
}
