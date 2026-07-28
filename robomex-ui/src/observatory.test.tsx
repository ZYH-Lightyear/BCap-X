import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AgentInspector } from './components/AgentInspector'
import { CandidateArena } from './components/CandidateArena'
import { SwarmGraph } from './components/SwarmGraph'
import type { AgentDetail, AgentSummary, SwarmConfigView } from './types'

describe('Swarm Observatory', () => {
  it('renders the actual graph surface and its status legend', () => {
    const html = renderToStaticMarkup(
      <SwarmGraph
        nodes={[{
          id: 'motion-a',
          label: 'motion-a',
          kind: 'motion',
          model: 'small',
          skills: ['plan-motion'],
          status: 'failed',
          agent_run_id: 'intent-1::cfg-0::motion-a',
        }]}
        edges={[]}
        selectedAgent=""
        onSelect={() => undefined}
      />,
    )
    expect(html).toContain('graph-legend')
    expect(html).toContain('failed')
  })

  it('shows failed Agent code, error and upstream evidence', () => {
    const summary: AgentSummary = {
      agent_run_id: 'intent-1::cfg-0::motion-a',
      role_id: 'motion-a',
      kind: 'motion',
      objective: 'plan motion',
      model: 'small',
      skills: ['plan-motion'],
      capabilities: ['ik'],
      depends_on: ['grounding'],
      status: 'failed',
      failure_kind: 'code_error',
      detail: 'boom',
      candidate_count: 0,
    }
    const detail: AgentDetail = {
      agent_run_id: summary.agent_run_id,
      assignment: {},
      world_view: {},
      upstream: { grounding: { center: [1, 2, 3] } },
      llm_request: [],
      llm_response: '```python',
      code: "raise RuntimeError('boom')",
      stdout: '',
      stderr: 'RuntimeError: boom',
      result: { succeeded: false },
      artifacts: [],
      log: 'failed',
    }
    const html = renderToStaticMarkup(
      <AgentInspector
        summary={summary}
        detail={detail}
        runId="run-1"
        onArtifact={() => undefined}
      />,
    )
    expect(html).toContain('motion-a')
    expect(html).toContain('code_error')
    expect(html).toContain('Upstream input')
  })

  it('compares candidates with selector and physical execution evidence', () => {
    const config: SwarmConfigView = {
      config_id: 'cfg-0',
      manager: { request: [], response: '', swarm_config: { intent_id: 'intent-1' } },
      nodes: [],
      edges: [],
      candidates: [{
        candidate_id: 'candidate-a',
        role_id: 'motion-a',
        kind: 'motion',
        notes: 'safe arc',
        candidate_digest: 'sha256:x',
        intent_id: 'intent-1',
        config_id: 'cfg-0',
        gates: { checks: { joint_limits: 'pass' } },
      }],
      selection: {
        verdict: 'selected',
        selected_candidate_id: 'candidate-a',
        reason: 'best overlay',
      },
      execution: {
        admission: { admitted: true },
        receipt: { status: 'completed' },
        observation: {},
      },
    }
    const html = renderToStaticMarkup(
      <CandidateArena runId="run-1" config={config} artifacts={[]} />,
    )
    expect(html).toContain('SELECTED')
    expect(html).toContain('best overlay')
    expect(html).toContain('completed')
  })
})
