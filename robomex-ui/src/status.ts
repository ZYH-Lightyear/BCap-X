export function statusTone(status: string | null | undefined): string {
  const s = String(status || '').toLowerCase()
  if (['succeeded', 'success', 'passed', 'pass', 'ok', 'finished', 'done'].includes(s)) {
    return 'border-emerald-300 bg-emerald-50 text-emerald-900'
  }
  if (['failed', 'fail', 'error', 'exhausted'].includes(s)) {
    return 'border-red-300 bg-red-50 text-red-900'
  }
  if (['uncertain', 'running', 'started', 'warn', 'warning'].includes(s)) {
    return 'border-amber-300 bg-amber-50 text-amber-900'
  }
  return 'border-ink-200 bg-white text-ink-700'
}

export function shortName(path: string): string {
  const parts = path.split('/')
  return parts.length > 3 ? `${parts.slice(0, 1).join('/')}/…/${parts.slice(-2).join('/')}` : path
}
