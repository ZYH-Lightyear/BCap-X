import { useState } from 'react';
import type { ExecutionLineData } from '../types/messages';

interface ExecutionTimelinePanelProps {
  lines: ExecutionLineData[];
  isExecuting?: boolean;
}

function phaseClass(phase: ExecutionLineData['phase']): string {
  if (phase === 'exception') return 'border-red-500/60 bg-red-950/20';
  if (phase === 'start') return 'border-accent bg-accent/10';
  if (phase === 'update') return 'border-accent/70 bg-accent/5';
  return 'border-surface-border bg-surface-sunken/60';
}

export function ExecutionTimelinePanel({ lines, isExecuting }: ExecutionTimelinePanelProps) {
  const [expanded, setExpanded] = useState(true);
  const currentLine = [...lines].reverse().find((line) => line.phase === 'start' || line.phase === 'update');

  if (lines.length === 0) return null;

  return (
    <div className="mt-2 bg-surface-raised/50 rounded-md border border-surface-border overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-surface-overlay transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg className={`w-4 h-4 text-accent transition-transform ${expanded ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
          <span className="text-sm font-medium text-accent">Line Timeline</span>
          <span className="text-xs text-text-tertiary">({lines.length} line{lines.length !== 1 ? 's' : ''})</span>
          {isExecuting && (
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          )}
        </div>
        {currentLine && (
          <span className="text-xs text-text-tertiary truncate max-w-[40%]">
            line {currentLine.lineno}
          </span>
        )}
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-1 space-y-1 max-h-80 overflow-y-auto">
          {lines.map((line) => {
            const frameRange = line.frameEnd > line.frameStart
              ? `${line.frameStart}-${line.frameEnd}`
              : `${line.frameStart}`;
            return (
              <div key={line.lineIndex} className={`border-l-2 pl-3 py-2 rounded-r-md ${phaseClass(line.phase)}`}>
                <div className="flex items-center gap-2 text-xs mb-1">
                  <span className="font-mono text-accent">L{line.lineno}</span>
                  <span className="text-text-tertiary">{line.phase}</span>
                  <span className="text-text-muted">frames {frameRange}</span>
                </div>
                <pre className="font-mono text-xs text-text-secondary whitespace-pre-wrap break-words">
                  {line.source || '(blank line)'}
                </pre>
                {(line.stdoutDelta || line.stderrDelta || line.exceptionMessage) && (
                  <div className="mt-2 space-y-1">
                    {line.stdoutDelta && (
                      <pre className="text-xs bg-surface-sunken border border-surface-border rounded p-2 text-text-secondary whitespace-pre-wrap">
                        {line.stdoutDelta}
                      </pre>
                    )}
                    {line.stderrDelta && (
                      <pre className="text-xs bg-red-950/30 border border-red-800/20 rounded p-2 text-red-400 whitespace-pre-wrap">
                        {line.stderrDelta}
                      </pre>
                    )}
                    {line.exceptionMessage && (
                      <div className="text-xs text-red-400">
                        {line.exceptionType}: {line.exceptionMessage}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
