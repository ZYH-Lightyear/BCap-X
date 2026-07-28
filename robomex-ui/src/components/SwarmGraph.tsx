import cytoscape, { type Core, type ElementDefinition, type StylesheetStyle } from 'cytoscape'
import { useEffect, useRef } from 'react'
import type { GraphEdge, GraphNode } from '../types'

export function SwarmGraph({
  nodes,
  edges,
  selectedAgent,
  onSelect,
}: {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedAgent: string
  onSelect: (agentRunId: string) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const graph = useRef<Core | null>(null)

  useEffect(() => {
    if (!host.current) return
    const elements: ElementDefinition[] = [
      ...nodes.map((node) => ({
        data: {
          ...node,
          label: `${node.kind.toUpperCase()}\n${node.label}\n${node.model}`,
        },
        classes: `${node.status} ${node.agent_run_id === selectedAgent ? 'selected' : ''}`,
      })),
      ...edges.map((edge, index) => ({
        data: { id: `edge-${index}`, source: edge.source, target: edge.target },
      })),
    ]
    graph.current?.destroy()
    const cy = cytoscape({
      container: host.current,
      elements,
      style: styles,
      layout: {
        name: 'breadthfirst',
        directed: true,
        padding: 28,
        spacingFactor: 1.25,
        avoidOverlap: true,
      },
      minZoom: 0.35,
      maxZoom: 2,
    })
    cy.on('tap', 'node', (event) => onSelect(String(event.target.data('agent_run_id'))))
    cy.on('tap', 'edge', (event) => {
      const target = cy.getElementById(String(event.target.data('target')))
      onSelect(String(target.data('agent_run_id')))
    })
    graph.current = cy
    return () => cy.destroy()
  }, [nodes, edges, selectedAgent, onSelect])

  return (
    <div className="cy-wrap">
      {nodes.length ? <div ref={host} className="cy-graph" /> : <Empty />}
      <div className="graph-legend">
        <span><i className="pending" /> pending</span>
        <span><i className="running" /> running</span>
        <span><i className="succeeded" /> succeeded</span>
        <span><i className="failed" /> failed</span>
      </div>
    </div>
  )
}

function Empty() {
  return <div className="empty">Manager has not authored a SwarmConfig yet.</div>
}

const styles: StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': '#182127',
      'border-color': '#60717b',
      'border-width': 1,
      color: '#e8eff1',
      label: 'data(label)',
      'font-family': 'Inter, ui-sans-serif',
      'font-size': 9,
      'font-weight': 600,
      'text-wrap': 'wrap',
      'text-max-width': '126px',
      'text-valign': 'center',
      'text-halign': 'center',
      shape: 'round-rectangle',
      width: 154,
      height: 70,
    },
  },
  { selector: 'node.running', style: { 'border-color': '#e9a84b', 'border-width': 3 } },
  { selector: 'node.succeeded', style: { 'border-color': '#51b793', 'background-color': '#172b29' } },
  { selector: 'node.failed', style: { 'border-color': '#e46b62', 'background-color': '#321d1d' } },
  { selector: 'node.selected', style: { 'overlay-color': '#65b7d4', 'overlay-opacity': 0.18, 'overlay-padding': 8 } },
  {
    selector: 'edge',
    style: {
      width: 1.5,
      'line-color': '#52636c',
      'target-arrow-color': '#78909b',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
    },
  },
]
