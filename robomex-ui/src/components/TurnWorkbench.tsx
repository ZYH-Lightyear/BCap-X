import { useEffect, useMemo, useRef } from 'react'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import 'highlight.js/styles/github.css'
import type { AgentDetail, TurnDetail, TurnIndex } from '../types'
import { statusTone } from '../status'

hljs.registerLanguage('python', python)

type Props = {
  agent: AgentDetail | null
  turns: TurnIndex[]
  selectedTurn: number | null
  turnDetail: TurnDetail | null
  onSelectTurn: (turn: number) => void
  onOpenLlm: (path: string) => void
}

export function TurnWorkbench({
  agent,
  turns,
  selectedTurn,
  turnDetail,
  onSelectTurn,
  onOpenLlm,
}: Props) {
  const codeRef = useRef<HTMLElement>(null)
  const highlighted = useMemo(() => {
    if (!turnDetail?.code) return ''
    return hljs.highlight(turnDetail.code, { language: 'python' }).value
  }, [turnDetail?.code])

  useEffect(() => {
    if (codeRef.current) {
      codeRef.current.innerHTML = highlighted || '<span class="text-ink-400">(no code)</span>'
    }
  }, [highlighted])

  if (!agent) {
    return (
      <div className="flex h-full items-center justify-center rounded border border-dashed border-ink-200 text-sm text-ink-500">
        Select an agent instance to inspect turns.
      </div>
    )
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[200px_minmax(0,1fr)] gap-3">
      <div className="min-h-0 overflow-y-auto rounded border border-ink-200 bg-white">
        <div className="border-b border-ink-200 px-3 py-2 text-xs font-semibold uppercase tracking-wide">
          Turns
        </div>
        <div className="space-y-1 p-2">
          {turns.map((turn) => (
            <button
              key={turn.turn}
              onClick={() => onSelectTurn(turn.turn)}
              className={`w-full rounded border px-2 py-1.5 text-left text-xs ${
                selectedTurn === turn.turn
                  ? 'border-ink-800 bg-ink-900 text-white'
                  : 'border-ink-200 bg-ink-50 hover:border-ink-400'
              }`}
            >
              <div className="font-mono">turn_{String(turn.turn).padStart(2, '0')}</div>
              <div className="mt-0.5 text-[10px] opacity-70">
                {[turn.has_code && 'code', turn.has_out && 'out', turn.has_llm && 'llm']
                  .filter(Boolean)
                  .join(' · ') || 'empty'}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 space-y-3 overflow-y-auto">
        <div className={`rounded border p-3 ${statusTone(agent.status)}`}>
          <div className="flex items-center justify-between gap-2">
            <div>
              <div className="font-mono text-sm font-semibold">{agent.agent_dir}</div>
              <div className="text-xs opacity-80">
                {agent.node_id} · {agent.role || 'role?'}
              </div>
            </div>
            <div className="text-xs">{agent.status || '—'}</div>
          </div>
        </div>

        <section className="rounded border border-ink-200 bg-white">
          <div className="border-b border-ink-200 px-3 py-2 text-xs font-semibold uppercase tracking-wide">
            Code
          </div>
          <pre className="max-h-72 overflow-auto p-3 text-xs">
            <code ref={codeRef} className="language-python hljs" />
          </pre>
        </section>

        <section className="rounded border border-ink-200 bg-white">
          <div className="border-b border-ink-200 px-3 py-2 text-xs font-semibold uppercase tracking-wide">
            Output
          </div>
          <pre className="max-h-56 overflow-auto whitespace-pre-wrap p-3 font-mono text-xs text-ink-800">
            {turnDetail?.out || '(empty)'}
          </pre>
        </section>

        {(turnDetail?.llm_request_path || turnDetail?.llm_response_path || turnDetail?.llm_response_txt_path) && (
          <section className="rounded border border-ink-200 bg-white p-3">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide">LLM I/O</div>
            <div className="flex flex-wrap gap-2">
              {turnDetail.llm_request_path && (
                <button
                  onClick={() => onOpenLlm(turnDetail.llm_request_path)}
                  className="rounded border border-ink-300 px-2 py-1 text-xs hover:bg-ink-50"
                >
                  request
                </button>
              )}
              {(turnDetail.llm_response_txt_path || turnDetail.llm_response_path) && (
                <button
                  onClick={() =>
                    onOpenLlm(turnDetail.llm_response_txt_path || turnDetail.llm_response_path)
                  }
                  className="rounded border border-ink-300 px-2 py-1 text-xs hover:bg-ink-50"
                >
                  response
                </button>
              )}
            </div>
            {turnDetail.response_preview && (
              <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-ink-50 p-2 font-mono text-[11px]">
                {turnDetail.response_preview}
              </pre>
            )}
          </section>
        )}
      </div>
    </div>
  )
}
