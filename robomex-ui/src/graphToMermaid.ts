import type { GraphEdge, GraphNode } from './types'

/** Role → subgraph bucket (pipeline stages). */
const ROLE_STAGE: Record<string, string> = {
  grounding: 'grounding',
  geometry_analyzer: 'geometry',
  affordance: 'affordance',
  motion_planner: 'motion',
  action_executor: 'action',
  verifier: 'verify',
}

const STAGE_ORDER = [
  'grounding',
  'geometry',
  'affordance',
  'motion',
  'action',
  'verify',
  'other',
]

function mermaidId(raw: string): string {
  const id = raw.replace(/[^a-zA-Z0-9_]/g, '_')
  return /^[0-9]/.test(id) ? `n_${id}` : id
}

function escapeLabel(text: string): string {
  return text.replace(/"/g, "'").replace(/[<>]/g, '').replace(/\n/g, ' ')
}

function stageFor(node: GraphNode): string {
  return ROLE_STAGE[node.role] || 'other'
}

function statusClass(status: string): string {
  const s = status.toLowerCase()
  if (['succeeded', 'success', 'passed', 'pass', 'ok', 'finished'].includes(s)) return 'ok'
  if (['failed', 'fail', 'error', 'exhausted'].includes(s)) return 'fail'
  if (['uncertain', 'running', 'started'].includes(s)) return 'warn'
  return ''
}

function kindClass(node: GraphNode): string {
  if (node.verifier) return 'verifier'
  if (node.changes_world) return 'action'
  return 'script'
}

function isFailureOn(on: string): boolean {
  const s = on.toLowerCase()
  return s.includes('fail') || s === 'failed' || s === 'error' || s === 'abort'
}

/**
 * Convert a RoboMEx subgoal graph into Mermaid flowchart text.
 * Visual language: stage subgraphs, solid success edges, dotted failure edges,
 * blue action / green perception / amber verifier nodes.
 */
export function graphToMermaid(
  nodes: GraphNode[],
  edges: GraphEdge[],
  opts?: { entry?: string; successNode?: string },
): string {
  const lines: string[] = [
    '%%{init: {"theme":"base","themeVariables":{"clusterBkg":"#f5f7fa","clusterBorder":"#90a4ae","titleColor":"#455a64"},"flowchart":{"curve":"basis","nodeSpacing":28,"rankSpacing":50}}}%%',
    'flowchart LR',
    '  classDef script fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20,stroke-width:1.5px',
    '  classDef action fill:#e3f2fd,stroke:#1565c0,color:#0d47a1,stroke-width:1.5px',
    '  classDef verifier fill:#fff8e1,stroke:#f9a825,color:#5d4037,stroke-width:1.5px',
    '  classDef ok fill:#c8e6c9,stroke:#1b5e20,color:#1b5e20,stroke-width:2.5px',
    '  classDef fail fill:#ffcdd2,stroke:#b71c1c,color:#b71c1c,stroke-width:2.5px',
    '  classDef warn fill:#ffe0b2,stroke:#ef6c00,color:#e65100,stroke-width:2px',
    '  classDef terminalOk fill:#a5d6a7,stroke:#1b5e20,color:#1b5e20,stroke-width:3px',
    '  classDef terminalFail fill:#ef9a9a,stroke:#b71c1c,color:#b71c1c,stroke-width:3px',
    '  classDef start fill:#eceff1,stroke:#546e7a,color:#263238,stroke-width:1.5px',
  ]

  const byStage = new Map<string, GraphNode[]>()
  for (const stage of STAGE_ORDER) byStage.set(stage, [])
  for (const node of nodes) {
    ;(byStage.get(stageFor(node)) || byStage.get('other')!).push(node)
  }

  const stageIds: string[] = []
  for (const stage of STAGE_ORDER) {
    const bucket = byStage.get(stage) || []
    if (!bucket.length) continue
    const sgId = mermaidId(`sg_${stage}`)
    stageIds.push(sgId)
    lines.push(`  subgraph ${sgId}["${stage}"]`)
    for (const node of bucket) {
      const id = mermaidId(node.id)
      const role = escapeLabel(node.role || node.specialist_skill || 'node')
      const status = escapeLabel(node.status || 'pending')
      lines.push(`    ${id}["${escapeLabel(node.id)}<br/>(${role})<br/>${status}"]`)
    }
    lines.push('  end')
  }
  for (const sgId of stageIds) {
    lines.push(`  style ${sgId} fill:#f5f7fa,stroke:#90a4ae,color:#455a64`)
  }

  lines.push('  DONE(["done (success)"])')
  lines.push('  ABORT(["abort (failure)"])')
  lines.push('  class DONE terminalOk')
  lines.push('  class ABORT terminalFail')

  const nodeIds = new Set(nodes.map((n) => n.id))
  const hasOutgoingSuccess = new Set(
    edges.filter((e) => !isFailureOn(e.on || 'success')).map((e) => e.source),
  )
  const hasFailureOut = new Set(
    edges.filter((e) => isFailureOn(e.on || '')).map((e) => e.source),
  )

  if (opts?.entry && nodeIds.has(opts.entry)) {
    lines.push('  START([start])')
    lines.push('  class START start')
    lines.push(`  START --> ${mermaidId(opts.entry)}`)
  }

  const edgeStyleIndexes: Array<{ index: number; fail: boolean }> = []
  let edgeIndex = 0

  for (const edge of edges) {
    if (!nodeIds.has(edge.source)) continue
    const src = mermaidId(edge.source)
    const dst = nodeIds.has(edge.target) ? mermaidId(edge.target) : mermaidId(edge.target)
    const on = edge.on || 'success'
    const label = escapeLabel(on)
    if (isFailureOn(on)) {
      lines.push(`  ${src} -.->|${label}| ${dst}`)
      edgeStyleIndexes.push({ index: edgeIndex, fail: true })
    } else {
      lines.push(`  ${src} -->|${label}| ${dst}`)
      edgeStyleIndexes.push({ index: edgeIndex, fail: false })
    }
    edgeIndex += 1
  }

  const successNode = opts?.successNode
  if (successNode && nodeIds.has(successNode) && !hasOutgoingSuccess.has(successNode)) {
    lines.push(`  ${mermaidId(successNode)} -->|succeeded| DONE`)
    edgeStyleIndexes.push({ index: edgeIndex, fail: false })
    edgeIndex += 1
  } else {
    for (const node of nodes) {
      if (!hasOutgoingSuccess.has(node.id) && node.verifier) {
        lines.push(`  ${mermaidId(node.id)} -->|succeeded| DONE`)
        edgeStyleIndexes.push({ index: edgeIndex, fail: false })
        edgeIndex += 1
        break
      }
    }
  }

  for (const node of nodes) {
    if (node.status.toLowerCase() === 'failed' && !hasFailureOut.has(node.id)) {
      lines.push(`  ${mermaidId(node.id)} -.->|failed| ABORT`)
      edgeStyleIndexes.push({ index: edgeIndex, fail: true })
      edgeIndex += 1
    }
  }

  // Also count START edge if present
  let styleOffset = opts?.entry && nodeIds.has(opts.entry) ? 1 : 0
  for (const item of edgeStyleIndexes) {
    const idx = item.index + styleOffset
    if (item.fail) {
      lines.push(`  linkStyle ${idx} stroke:#c62828,stroke-width:2.5px`)
    } else {
      lines.push(`  linkStyle ${idx} stroke:#2e7d32,stroke-width:2.5px`)
    }
  }
  if (styleOffset === 1) {
    lines.push('  linkStyle 0 stroke:#546e7a,stroke-width:1.5px')
  }

  for (const node of nodes) {
    const id = mermaidId(node.id)
    const st = statusClass(node.status)
    lines.push(`  class ${id} ${st || kindClass(node)}`)
  }

  return lines.join('\n')
}
