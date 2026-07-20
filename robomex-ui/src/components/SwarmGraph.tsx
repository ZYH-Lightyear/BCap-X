import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import mermaid from 'mermaid'
import { graphToMermaid } from '../graphToMermaid'
import type { AgentInstance, GraphEdge, GraphNode } from '../types'
import { statusTone } from '../status'

mermaid.initialize({
  startOnLoad: false,
  securityLevel: 'loose',
  // `neutral` paints subgraph clusters black (B&W theme). Use base + light clusters.
  theme: 'base',
  themeVariables: {
    darkMode: false,
    background: '#ffffff',
    primaryTextColor: '#263238',
    lineColor: '#607d8b',
    clusterBkg: '#f5f7fa',
    clusterBorder: '#90a4ae',
    titleColor: '#455a64',
    edgeLabelBackground: '#ffffff',
  },
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
    padding: 12,
  },
})

type Props = {
  nodes: GraphNode[]
  edges: GraphEdge[]
  agents: AgentInstance[]
  selectedAgent: string
  onSelectAgent: (dir: string) => void
  entry?: string
  successNode?: string
  outcomeStatus?: string
  note?: string
}

type ViewState = {
  scale: number
  x: number
  y: number
}

const MIN_SCALE = 0.35
const MAX_SCALE = 4

function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale))
}

/** Mermaid SVG reuses marker/clipPath ids; duplicating the markup blanks the second copy. */
function uniquifySvgIds(svg: string, prefix: string): string {
  const ids = new Set<string>()
  for (const match of svg.matchAll(/\bid="([^"]+)"/g)) {
    ids.add(match[1])
  }
  // Longer ids first so short prefixes (e.g. "a" vs "a1") cannot corrupt neighbors.
  const ordered = [...ids].sort((a, b) => b.length - a.length)
  let out = svg
  for (const id of ordered) {
    const safe = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    out = out
      .replace(new RegExp(`id="${safe}"`, 'g'), `id="${prefix}${id}"`)
      .replace(new RegExp(`url\\(#${safe}\\)`, 'g'), `url(#${prefix}${id})`)
      .replace(new RegExp(`href="#${safe}"`, 'g'), `href="#${prefix}${id}"`)
      .replace(new RegExp(`xlink:href="#${safe}"`, 'g'), `xlink:href="#${prefix}${id}"`)
  }
  return out
}

function parseSvgLength(raw: string | null): number {
  if (!raw) return 0
  // Mermaid often emits width="100%"; that is useless inside a max-content pan stage.
  if (raw.includes('%')) return 0
  const n = Number.parseFloat(raw)
  return Number.isFinite(n) && n > 0 ? n : 0
}

/** Keep a real pixel size so zoom/pan parent (width:max-content) does not collapse to 0. */
function applySvgLayout(svgEl: SVGSVGElement): void {
  svgEl.style.maxWidth = 'none'
  svgEl.style.display = 'block'
  const viewBox = svgEl.getAttribute('viewBox')
  const vb = viewBox?.trim().split(/[\s,]+/).map(Number)
  const vbW = vb && vb.length === 4 && Number.isFinite(vb[2]) ? vb[2] : 0
  const vbH = vb && vb.length === 4 && Number.isFinite(vb[3]) ? vb[3] : 0
  const width = parseSvgLength(svgEl.getAttribute('width')) || vbW || 800
  const height = parseSvgLength(svgEl.getAttribute('height')) || vbH || 400
  svgEl.setAttribute('width', String(width))
  svgEl.setAttribute('height', String(height))
  svgEl.style.width = `${width}px`
  svgEl.style.height = `${height}px`
}

function GraphCanvas({
  svg,
  instanceKey,
  nodes,
  agentByNode,
  onSelectAgent,
  className,
  initialScale = 1,
}: {
  svg: string
  instanceKey: string
  nodes: GraphNode[]
  agentByNode: Map<string, AgentInstance[]>
  onSelectAgent: (dir: string) => void
  className?: string
  initialScale?: number
}) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<ViewState>({ scale: initialScale, x: 16, y: 16 })
  const dragRef = useRef<{
    active: boolean
    startX: number
    startY: number
    originX: number
    originY: number
  }>({ active: false, startX: 0, startY: 0, originX: 0, originY: 0 })

  const bindNodeClicks = useCallback(
    (root: HTMLElement) => {
      root.querySelectorAll('.node').forEach((el) => {
        const nodeEl = el as HTMLElement
        nodeEl.style.cursor = 'pointer'
        nodeEl.onclick = () => {
          const rawId = nodeEl.id
            ?.replace(new RegExp(`^${instanceKey}`), '')
            .replace(/^flowchart-/, '')
            .replace(/-\d+$/, '') || ''
          const match =
            nodes.find((n) => n.id === rawId) ||
            nodes.find((n) => n.id.replace(/[^a-zA-Z0-9_]/g, '_') === rawId)
          if (!match) return
          const instances = agentByNode.get(match.id) || []
          if (instances.length) onSelectAgent(instances[instances.length - 1].dir)
        }
      })
    },
    [agentByNode, instanceKey, nodes, onSelectAgent],
  )

  useEffect(() => {
    const stage = stageRef.current
    if (!stage || !svg) return
    stage.innerHTML = uniquifySvgIds(svg, `${instanceKey}_`)
    const svgEl = stage.querySelector('svg')
    if (svgEl) applySvgLayout(svgEl)
    bindNodeClicks(stage)
  }, [svg, instanceKey, bindNodeClicks])

  useEffect(() => {
    setView({ scale: initialScale, x: 16, y: 16 })
  }, [svg, initialScale, instanceKey])

  const onWheel = (event: React.WheelEvent) => {
    event.preventDefault()
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect) return
    const cursorX = event.clientX - rect.left
    const cursorY = event.clientY - rect.top
    const factor = event.deltaY > 0 ? 0.9 : 1.1
    setView((prev) => {
      const nextScale = clampScale(prev.scale * factor)
      const ratio = nextScale / prev.scale
      return {
        scale: nextScale,
        x: cursorX - (cursorX - prev.x) * ratio,
        y: cursorY - (cursorY - prev.y) * ratio,
      }
    })
  }

  const onPointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return
    const target = event.target as HTMLElement
    if (target.closest('.node')) return
    dragRef.current = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
      originX: view.x,
      originY: view.y,
    }
    ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
  }

  const onPointerMove = (event: React.PointerEvent) => {
    if (!dragRef.current.active) return
    const dx = event.clientX - dragRef.current.startX
    const dy = event.clientY - dragRef.current.startY
    setView((prev) => ({
      ...prev,
      x: dragRef.current.originX + dx,
      y: dragRef.current.originY + dy,
    }))
  }

  const onPointerUp = (event: React.PointerEvent) => {
    dragRef.current.active = false
    try {
      ;(event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId)
    } catch {
      // ignore
    }
  }

  const zoomBy = (factor: number) => {
    const rect = viewportRef.current?.getBoundingClientRect()
    const cx = rect ? rect.width / 2 : 0
    const cy = rect ? rect.height / 2 : 0
    setView((prev) => {
      const nextScale = clampScale(prev.scale * factor)
      const ratio = nextScale / prev.scale
      return {
        scale: nextScale,
        x: cx - (cx - prev.x) * ratio,
        y: cy - (cy - prev.y) * ratio,
      }
    })
  }

  const resetView = () => setView({ scale: initialScale, x: 16, y: 16 })

  return (
    <div className={`relative min-h-0 ${className || ''}`}>
      <div className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded border border-ink-300 bg-white/95 p-1 shadow-sm">
        <button
          type="button"
          onClick={() => zoomBy(1 / 1.2)}
          className="h-7 w-7 rounded border border-ink-200 text-sm hover:bg-ink-50"
          title="Zoom out"
        >
          −
        </button>
        <span className="min-w-[3.2rem] text-center font-mono text-[11px] text-ink-600">
          {Math.round(view.scale * 100)}%
        </span>
        <button
          type="button"
          onClick={() => zoomBy(1.2)}
          className="h-7 w-7 rounded border border-ink-200 text-sm hover:bg-ink-50"
          title="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={resetView}
          className="rounded border border-ink-200 px-2 py-1 text-[11px] hover:bg-ink-50"
          title="Reset view"
        >
          Reset
        </button>
      </div>
      <div
        ref={viewportRef}
        className="h-full min-h-[200px] w-full cursor-grab overflow-hidden bg-white active:cursor-grabbing"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div
          ref={stageRef}
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            transformOrigin: '0 0',
            width: 'max-content',
          }}
        />
      </div>
      <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-ink-900/70 px-2 py-1 text-[10px] text-white">
        scroll 缩放 · 拖拽平移 · 点击节点选中 agent
      </div>
    </div>
  )
}

export function SwarmGraph({
  nodes,
  edges,
  agents,
  selectedAgent,
  onSelectAgent,
  entry,
  successNode,
  outcomeStatus,
  note,
}: Props) {
  const [renderError, setRenderError] = useState<string | null>(null)
  const [showSource, setShowSource] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const [svg, setSvg] = useState('')

  const agentByNode = useMemo(() => {
    const map = new Map<string, AgentInstance[]>()
    for (const agent of agents) {
      const list = map.get(agent.node_id) || []
      list.push(agent)
      map.set(agent.node_id, list)
    }
    return map
  }, [agents])

  const source = useMemo(
    () => graphToMermaid(nodes, edges, { entry, successNode }),
    [nodes, edges, entry, successNode],
  )

  useEffect(() => {
    let cancelled = false
    if (!nodes.length) {
      setSvg('')
      return
    }
    ;(async () => {
      try {
        const id = `swarm_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
        const { svg: rendered } = await mermaid.render(id, source)
        if (cancelled) return
        setSvg(rendered)
        setRenderError(null)
      } catch (err) {
        if (!cancelled) {
          setSvg('')
          setRenderError(err instanceof Error ? err.message : String(err))
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [source, nodes.length])

  useEffect(() => {
    if (!fullscreen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setFullscreen(false)
    }
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = prev
      window.removeEventListener('keydown', onKey)
    }
  }, [fullscreen])

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded border border-ink-200 bg-white px-2 py-1">entry={entry || '—'}</span>
        <span className={`rounded border px-2 py-1 ${statusTone(outcomeStatus)}`}>
          outcome={outcomeStatus || '—'}
        </span>
        {note && <span className="max-w-xl truncate text-ink-500">{note}</span>}
        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setShowSource((v) => !v)}
            className="rounded border border-ink-300 bg-white px-2 py-1 text-[11px] hover:bg-ink-50"
          >
            {showSource ? 'Hide Mermaid' : 'Show Mermaid'}
          </button>
          <button
            onClick={() => setFullscreen(true)}
            disabled={!svg}
            className="rounded border border-ink-800 bg-ink-900 px-2 py-1 text-[11px] text-white hover:bg-ink-700 disabled:opacity-40"
          >
            全屏放大
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 text-[11px] text-ink-600">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm border border-[#1565c0] bg-[#e3f2fd]" /> action
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm border border-[#2e7d32] bg-[#e8f5e9]" /> perception/script
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm border border-[#f9a825] bg-[#fff8e1]" /> verifier
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-0.5 w-4 bg-[#2e7d32]" /> success
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-0.5 w-4 border-t-2 border-dashed border-[#c62828]" /> failure
        </span>
      </div>

      {nodes.length === 0 ? (
        <div className="rounded border border-dashed border-ink-200 p-6 text-sm text-ink-500">
          No graph nodes for this subgoal.
        </div>
      ) : renderError ? (
        <div className="rounded border border-red-200 bg-red-50 p-3 text-xs text-signal-fail">
          Mermaid render failed: {renderError}
        </div>
      ) : (
        <div className="h-[280px] overflow-hidden rounded border border-ink-200">
          {/* Unmount preview while fullscreen so SVG ids never collide. */}
          {!fullscreen && svg ? (
            <GraphCanvas
              svg={svg}
              instanceKey="preview"
              nodes={nodes}
              agentByNode={agentByNode}
              onSelectAgent={onSelectAgent}
              className="h-full"
              initialScale={0.85}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-ink-500">
              {fullscreen ? 'Fullscreen open…' : 'Rendering…'}
            </div>
          )}
        </div>
      )}

      {showSource && (
        <pre className="max-h-48 overflow-auto rounded border border-ink-200 bg-ink-50 p-2 font-mono text-[11px] text-ink-800">
          {source}
        </pre>
      )}

      <div className="rounded border border-ink-200 bg-white p-2">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
          Agent instances (click graph node or row)
        </div>
        <div className="flex flex-wrap gap-1.5">
          {agents.map((agent) => (
            <button
              key={agent.dir}
              onClick={() => onSelectAgent(agent.dir)}
              className={`rounded border px-2 py-1 font-mono text-[11px] ${
                selectedAgent === agent.dir
                  ? 'border-ink-800 bg-ink-900 text-white'
                  : `border-ink-200 bg-ink-50 hover:border-ink-500 ${statusTone(agent.status)}`
              }`}
            >
              {agent.dir}
              <span className="ml-1 opacity-70">t={agent.turn_count}</span>
            </button>
          ))}
          {agents.length === 0 && (
            <span className="text-[11px] text-ink-500">No agent instances yet.</span>
          )}
        </div>
      </div>

      {fullscreen && svg && (
        <div className="fixed inset-0 z-[80] flex flex-col bg-ink-900/50">
          <div className="flex flex-shrink-0 items-center justify-between border-b border-ink-300 bg-white px-4 py-3">
            <div>
              <div className="text-sm font-semibold">Swarm Graph · Fullscreen</div>
              <div className="text-[11px] text-ink-500">
                entry={entry || '—'} · outcome={outcomeStatus || '—'} · Esc 退出
              </div>
            </div>
            <button
              onClick={() => setFullscreen(false)}
              className="rounded border border-ink-300 bg-white px-3 py-1.5 text-xs hover:bg-ink-50"
            >
              关闭
            </button>
          </div>
          <div className="min-h-0 flex-1 bg-ink-100 p-3">
            <div className="h-full min-h-0 overflow-hidden rounded border border-ink-300 bg-white shadow-sm">
              <GraphCanvas
                svg={svg}
                instanceKey="fullscreen"
                nodes={nodes}
                agentByNode={agentByNode}
                onSelectAgent={onSelectAgent}
                className="h-full"
                initialScale={1.1}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
