"""Live trace viewer for VAW context-agent runs.

Reads a run directory (``steps.jsonl`` plus ``context_*.png``) while the episode
is still being written and serves a single-page timeline over stdlib HTTP.  The
point is not to pretty-print the log: it is to surface the derived signals that
the Canvas itself does not carry, above all the gap between a commanded
``delta_move`` and the displacement the robot actually achieved.  A run can
spend a third of its budget re-planning around a physically blocked axis while
every function result still reports success.

    python -m vaw.scripts.serve_trace --trace-dir vaw/out/<suite> --port 8123

A parent directory becomes the Board (``/api/board``).  Clicking a row opens
the Inspector (``#run=<name>``), which polls ``/api/trace?after=`` so a live
episode does not resend the whole JSONL every two seconds.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from vaw.context_runtime.trace_read import (
    _identity_from_name,
    compile_board,
    compile_run,
    discover_runs,
    summarize_run,
)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_relative(root: Path, relative: str) -> Path | None:
    """Resolve a request path inside ``root``, refusing traversal."""

    parts = [part for part in unquote(relative).split("/") if part]
    if not parts or any(not _SAFE_NAME.match(part) or part == ".." for part in parts):
        return None
    candidate = (root / Path(*parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class _Handler(BaseHTTPRequestHandler):
    root: Path
    single_run: bool

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - quiet by default
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _run_dir(self, query: dict[str, list[str]]) -> Path | None:
        if self.single_run:
            return self.root
        names = query.get("run")
        if not names:
            available = discover_runs(self.root)
            return self.root / available[0] if available else None
        return _safe_relative(self.root, names[0])

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in {"/", "/index.html"}:
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/runs":
            self._send_json(
                {
                    "singleRun": self.single_run,
                    "runs": [self.root.name] if self.single_run else discover_runs(self.root),
                }
            )
            return
        if parsed.path == "/api/board":
            self._send_json(compile_board(self.root, single_run=self.single_run))
            return
        if parsed.path == "/api/trace":
            run_dir = self._run_dir(query)
            if run_dir is None or not (run_dir / "steps.jsonl").is_file():
                self._send_json({"error": "run not found"}, status=404)
                return
            after = None
            raw_after = query.get("after")
            if raw_after:
                try:
                    after = int(raw_after[0])
                except ValueError:
                    self._send_json({"error": "after must be an integer"}, status=400)
                    return
            self._send_json(compile_run(run_dir, after=after))
            return
        if parsed.path.startswith("/img/"):
            run_dir = self._run_dir(query)
            if run_dir is None:
                self._send_json({"error": "run not found"}, status=404)
                return
            target = _safe_relative(run_dir, parsed.path[len("/img/") :])
            if target is None or not target.is_file():
                self._send_json({"error": "image not found"}, status=404)
                return
            guessed, _ = mimetypes.guess_type(target.name)
            self._send(200, target.read_bytes(), guessed or "application/octet-stream")
            return
        self._send_json({"error": "not found"}, status=404)


_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VAW Board</title>
<style>
:root{--bg:#0b1220;--panel:#121b2e;--line:#233149;--ink:#e6edf7;--dim:#8ea3c0;
--ok:#34d399;--warn:#fbbf24;--bad:#f87171;--accent:#60a5fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;flex-wrap:wrap;
background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
header h1{font-size:13px;margin:0;letter-spacing:.08em;color:var(--accent)}
select,button{background:#0e1729;color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:5px 9px;font:inherit}
button{cursor:pointer}
.stats{margin-left:auto;display:flex;gap:14px;color:var(--dim)}
.stats b{color:var(--ink);font-weight:600}
.stats b.bad{color:var(--bad)} .stats b.ok{color:var(--ok)}
#board{padding:16px;display:none}
#board.show{display:block}
#inspector{display:none;grid-template-columns:340px 1fr;height:calc(100vh - 49px)}
#inspector.show{display:grid}
#list{overflow-y:auto;border-right:1px solid var(--line);background:#0d1524}
#detail{overflow-y:auto;padding:16px 20px}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 10px;border-bottom:1px solid #18233a;text-align:left;vertical-align:top}
th{color:var(--dim);font-size:10px;letter-spacing:.08em;font-weight:600}
tr.run{cursor:pointer}
tr.run:hover{background:#152036}
.prompt{color:#cfe0f7;max-width:360px}
.last{color:var(--dim);max-width:420px}
.turn{padding:8px 12px;border-bottom:1px solid #18233a;cursor:pointer;display:flex;
gap:8px;align-items:baseline}
.turn:hover{background:#152036}
.turn.sel{background:#1b2b48;box-shadow:inset 3px 0 0 var(--accent)}
.turn .n{color:var(--dim);min-width:30px}
.turn .fn{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.turn.sub{padding-left:30px;background:#0a1120}
.turn.sub .fn{color:#c4b5fd}
.tag{font-size:10px;padding:1px 6px;border-radius:4px;letter-spacing:.04em;margin-right:4px}
.tag.live{background:#064e3b;color:var(--ok)}
.tag.done{background:#1e293b;color:var(--dim)}
.tag.ok{background:#064e3b;color:var(--ok)}
.tag.fail{background:#4c1d1d;color:var(--bad)}
.tag.blocked{background:#4c1d1d;color:var(--bad)}
.tag.err{background:#4c1d1d;color:var(--bad)}
.tag.adv{background:#463209;color:var(--warn)}
.tag.claim{background:#1e3a5f;color:var(--accent)}
.phase{display:flex;gap:6px;margin-bottom:12px}
.phase span{flex:1;text-align:center;padding:4px 0;border:1px solid var(--line);
border-radius:6px;color:var(--dim);font-size:11px;letter-spacing:.08em}
.phase span.on{border-color:var(--accent);color:var(--ink);background:#1b2b48}
.frames{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.frames figure{margin:0}
.frames img{width:100%;border:1px solid var(--line);border-radius:8px;display:block}
.frames figcaption{color:var(--dim);font-size:10px;letter-spacing:.08em;margin-top:4px}
section{margin-bottom:16px}
h2{font-size:11px;letter-spacing:.12em;color:var(--dim);margin:0 0 6px;font-weight:600}
pre{background:#0d1524;border:1px solid var(--line);border-radius:8px;padding:10px 12px;
margin:0;white-space:pre-wrap;word-break:break-word;overflow-x:auto}
.basis{border-left:3px solid var(--accent);padding-left:10px;color:#cfe0f7}
.advisory{border-left:3px solid var(--warn);padding-left:10px;color:#f4dfa8}
.error{border-left:3px solid var(--bad);padding-left:10px;color:#f7b8b8}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.kv div{background:#0d1524;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kv span{display:block;color:var(--dim);font-size:10px;letter-spacing:.08em}
.kv b{font-weight:600}
.kv b.bad{color:var(--bad)} .kv b.ok{color:var(--ok)}
.empty{color:var(--dim);padding:40px;text-align:center}
.filters{display:flex;gap:8px;align-items:center}
</style>
</head>
<body>
<header>
  <h1 id="title">VAW BOARD</h1>
  <button id="back" hidden>board</button>
  <div class="filters" id="board-filters">
    <select id="filter-status">
      <option value="all">all runs</option>
      <option value="live">live</option>
      <option value="ok">env success</option>
      <option value="fail">env fail</option>
      <option value="imagination">used imagination</option>
    </select>
  </div>
  <label id="follow-wrap" style="color:var(--dim)" hidden>
    <input type="checkbox" id="follow" checked/> follow
  </label>
  <button id="reload">reload</button>
  <div class="stats" id="stats"></div>
</header>
<div id="board"><div class="empty">loading…</div></div>
<div id="inspector">
  <div id="list"></div>
  <div id="detail"><div class="empty">select a turn</div></div>
</div>
<script>
const $ = (id) => document.getElementById(id);
const PHASES = ['detect', 'grasp', 'carry', 'place'];
let boardRows = [];
let rows = [];
let selected = null;
let runName = null;
let cursor = -1;
let inspectorOn = false;

const cm = (m) => (m == null ? '–' : (m * 100).toFixed(1) + ' cm');
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const pretty = (v) => typeof v === 'string' ? v : JSON.stringify(v, null, 2);
const hashRun = () => new URLSearchParams(location.hash.replace(/^#/, '')).get('run');

function setView(name) {
  inspectorOn = name === 'inspector';
  $('board').classList.toggle('show', !inspectorOn);
  $('inspector').classList.toggle('show', inspectorOn);
  $('board-filters').hidden = inspectorOn;
  $('follow-wrap').hidden = !inspectorOn;
  $('back').hidden = !inspectorOn;
  $('title').textContent = inspectorOn ? 'VAW TRACE' : 'VAW BOARD';
}

function flatten(data) {
  const out = [];
  const subs = data.subagents || [];
  let cursorSub = 0;
  for (const turn of data.turns) {
    out.push({ ...turn, sub: false });
    if (turn.function === 'call_imagination' && cursorSub < subs.length) {
      for (const inner of subs[cursorSub].turns) out.push({ ...inner, sub: true });
      cursorSub += 1;
    }
  }
  for (; cursorSub < subs.length; cursorSub += 1) {
    for (const inner of subs[cursorSub].turns) out.push({ ...inner, sub: true });
  }
  return out;
}

function rowKey(t) {
  return (t.sub ? 'imagination' : 'main') + ':' + (t.index ?? t.turn);
}

function mergeTurns(incoming) {
  const map = new Map(rows.map((t) => [rowKey(t), t]));
  for (const t of incoming) map.set(rowKey(t), t);
  rows = [...map.values()];
}

function phaseOf(fn) {
  if (!fn) return 'detect';
  if (['detection_and_sam', 'locate_point'].includes(fn)) return 'detect';
  if (['propose_grasps', 'select', 'close_gripper'].includes(fn)) return 'grasp';
  if (['propose_pose', 'open_gripper'].includes(fn)) return 'place';
  if (fn === 'commit') return 'grasp';
  return null;
}

function currentPhase() {
  let phase = 'detect';
  let closed = false;
  for (const t of rows) {
    if (t.sub) continue;
    if (t.function === 'close_gripper') { closed = true; phase = 'carry'; continue; }
    if (closed && (t.function === 'propose_pose' || t.function === 'open_gripper' || t.function === 'locate_point')) {
      phase = 'place';
      continue;
    }
    const next = phaseOf(t.function);
    if (!closed && next) phase = next;
    if (closed && t.function === 'delta_move' && phase !== 'place') phase = 'carry';
  }
  return phase;
}

function outcome(row) {
  if (row.live) return '<span class="tag live">LIVE</span>';
  if (row.envSuccess === true) return '<span class="tag ok">ENV OK</span>';
  if (row.envSuccess === false) return '<span class="tag fail">ENV FAIL</span>';
  return '<span class="tag done">DONE</span>';
}

function toolLine(tools) {
  return Object.entries(tools || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, v]) => k + '×' + v)
    .join('  ');
}

function filteredBoard() {
  const mode = $('filter-status').value;
  return boardRows.filter((row) => {
    if (mode === 'live') return row.live;
    if (mode === 'ok') return row.envSuccess === true;
    if (mode === 'fail') return row.envSuccess === false;
    if (mode === 'imagination') return row.imaginationCalls > 0;
    return true;
  });
}

function renderBoard() {
  const rowsView = filteredBoard();
  const live = boardRows.filter((r) => r.live).length;
  const ok = boardRows.filter((r) => r.envSuccess === true).length;
  $('stats').innerHTML = `<span>runs <b>${boardRows.length}</b></span>
    <span>live <b>${live}</b></span>
    <span>ok <b class="ok">${ok}</b></span>`;
  if (!rowsView.length) {
    $('board').innerHTML = '<div class="empty">no runs match the filter</div>';
    return;
  }
  $('board').innerHTML = `<table><thead><tr>
    <th>STATUS</th><th>TASK</th><th>ID</th><th>TURNS</th><th>IMAG</th>
    <th>BLOCKED</th><th>LAST CALL</th><th>TOKENS</th></tr></thead><tbody>
    ${rowsView.map((r) => `<tr class="run" data-run="${esc(r.name)}">
      <td>${outcome(r)}${r.claimedSuccess ? ' <span class="tag claim">CLAIMED</span>' : ''}</td>
      <td><div class="prompt">${esc(r.taskPrompt || r.name)}</div>
        <div class="last">${esc([r.suite, r.model].filter(Boolean).join(' · '))}</div></td>
      <td>t${r.taskId ?? '–'} s${r.seed ?? '–'}</td>
      <td>${r.turns}</td>
      <td>${r.imaginationCalls}</td>
      <td>${r.blockedMoves ? '<b class="bad">' + r.blockedMoves + '</b>' : '0'}</td>
      <td><div>${esc(r.lastFunction || '—')}</div>
        <div class="last">${esc((r.lastBasis || '').slice(0, 140))}</div>
        <div class="last">${esc(toolLine(r.tools))}</div></td>
      <td>${r.tokens ?? '–'}</td>
    </tr>`).join('')}</tbody></table>`;
  for (const node of document.querySelectorAll('tr.run')) {
    node.onclick = () => openRun(node.dataset.run);
  }
}

function renderList() {
  $('list').innerHTML = rows.map((t, i) => {
    const tags = [];
    if (t.motion && t.motion.blocked) tags.push('<span class="tag blocked">BLOCKED</span>');
    if (t.error) tags.push('<span class="tag err">ERR</span>');
    else if (t.advisory) tags.push('<span class="tag adv">ADV</span>');
    return `<div class="turn ${t.sub ? 'sub' : ''} ${i === selected ? 'sel' : ''}" data-i="${i}">
      <span class="n">${t.sub ? '↳' : ''}${t.turn ?? ''}</span>
      <span class="fn">${esc(t.function || '—')}</span>${tags.join('')}</div>`;
  }).join('');
  for (const node of document.querySelectorAll('.turn')) {
    node.onclick = () => { selected = +node.dataset.i; $('follow').checked = false; renderList(); renderDetail(); };
  }
}

function imgUrl(image) {
  return '/img/' + image + '?run=' + encodeURIComponent(runName);
}

function renderDetail() {
  const t = rows[selected];
  if (!t) { $('detail').innerHTML = '<div class="empty">no turn selected</div>'; return; }
  const prev = [...rows.slice(0, selected)].reverse().find((x) => x.image && !x.sub && !t.sub)
    || [...rows.slice(0, selected)].reverse().find((x) => x.image);
  const m = t.motion;
  const cells = [];
  if (t.tcp) cells.push(`<div><span>TCP BASE XYZ</span><b>${t.tcp.map((v) => v.toFixed(4)).join('  ')}</b></div>`);
  if (t.grip != null) cells.push(`<div><span>GRIP</span><b>${t.grip.toFixed(3)}</b></div>`);
  if (m) {
    cells.push(`<div><span>COMMANDED</span><b>${cm(m.commandedM)}</b></div>`);
    if (m.previewOnly) {
      cells.push(`<div><span>PHYSICAL TCP</span><b>unchanged · preview only</b></div>`);
    } else {
      cells.push(`<div><span>ACTUALLY MOVED</span><b class="${m.blocked ? 'bad' : 'ok'}">${cm(m.achievedM)}</b></div>`);
    }
  }
  const phase = currentPhase();
  const parts = [`<div class="phase">${PHASES.map((p) => `<span class="${p === phase ? 'on' : ''}">${p}</span>`).join('')}</div>`];
  if (t.image) {
    parts.push(`<section class="frames">
      <figure>${prev && prev.image ? `<img src="${imgUrl(prev.image)}" alt="previous"/>` : '<div class="empty">no previous frame</div>'}
        <figcaption>PREVIOUS</figcaption></figure>
      <figure><img src="${imgUrl(t.image)}" alt="current"/>
        <figcaption>CURRENT · T${t.turn ?? ''}</figcaption></figure>
    </section>`);
  }
  if (m && m.blocked) parts.push(`<section><h2>BLOCKED MOTION</h2><pre class="error">commanded ${cm(m.commandedM)} but the robot moved ${cm(m.achievedM)}. The function still reported success.</pre></section>`);
  if (cells.length) parts.push(`<section><h2>PHYSICAL STATE</h2><div class="kv">${cells.join('')}</div></section>`);
  parts.push(`<section><h2>CALL</h2><pre>${esc(t.function || '—')}(${esc(t.arguments ? JSON.stringify(t.arguments) : '')})</pre></section>`);
  if (t.basis) parts.push(`<section><h2>DECISION BASIS</h2><pre class="basis">${esc(t.basis)}</pre></section>`);
  if (t.thought && t.thought !== t.basis) parts.push(`<section><h2>THOUGHT</h2><pre>${esc(t.thought)}</pre></section>`);
  if (t.error) parts.push(`<section><h2>ERROR</h2><pre class="error">${esc(t.error)}</pre></section>`);
  if (t.advisory) parts.push(`<section><h2>ADVISORY</h2><pre class="advisory">${esc(t.advisory)}</pre></section>`);
  if (t.result != null) parts.push(`<section><h2>RESULT</h2><pre>${esc(pretty(t.result))}</pre></section>`);
  $('detail').innerHTML = parts.join('');
}

async function pollBoard() {
  const data = await (await fetch('/api/board')).json();
  boardRows = data.runs || [];
  if (!inspectorOn) renderBoard();
}

async function pollTrace() {
  if (!runName) return;
  const params = new URLSearchParams({ run: runName });
  if (cursor >= 0) params.set('after', String(cursor));
  const data = await (await fetch('/api/trace?' + params.toString())).json();
  if (data.error) return;
  runName = data.run;
  const incoming = flatten(data);
  const atTail = selected === null || selected === rows.length - 1;
  if (cursor < 0) rows = incoming;
  else mergeTurns(incoming);
  cursor = data.cursor ?? cursor;
  if ($('follow').checked && atTail) selected = rows.length - 1;
  if (selected === null || selected >= rows.length) selected = rows.length - 1;
  const s = data.summary || {};
  $('stats').innerHTML = `<span>${esc(s.taskPrompt || runName)}</span>
    <span>turns <b>${s.turnCount}</b></span>
    <span>blocked <b class="${s.blockedMoves ? 'bad' : ''}">${s.blockedMoves}</b></span>
    <span>errors <b>${s.errors}</b></span>
    <span>env <b class="${s.envSuccess ? 'ok' : (s.envSuccess === false ? 'bad' : '')}">${s.envSuccess == null ? '–' : s.envSuccess}</b></span>`;
  renderList();
  renderDetail();
  if ($('follow').checked) {
    const node = document.querySelector('.turn.sel');
    if (node) node.scrollIntoView({ block: 'nearest' });
  }
}

function openRun(name) {
  runName = name;
  rows = [];
  selected = null;
  cursor = -1;
  location.hash = 'run=' + encodeURIComponent(name);
  setView('inspector');
  pollTrace();
}

function showBoard() {
  runName = null;
  cursor = -1;
  location.hash = '';
  setView('board');
  renderBoard();
}

async function tick() {
  try {
    if (inspectorOn) await pollTrace();
    else await pollBoard();
  } catch (err) { /* mid-write */ }
}

async function boot() {
  $('filter-status').onchange = renderBoard;
  $('back').onclick = showBoard;
  $('reload').onclick = tick;
  window.onhashchange = () => {
    const name = hashRun();
    if (name) openRun(name);
    else showBoard();
  };
  await pollBoard();
  const initial = hashRun();
  if (initial) openRun(initial);
  else setView('board');
  setInterval(tick, 2000);
}
boot();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        required=True,
        type=Path,
        help="a run directory, or a parent directory of run directories",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    root = args.trace_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"trace dir not found: {root}")
    single_run = (root / "steps.jsonl").is_file()
    if not single_run and not discover_runs(root):
        raise SystemExit(f"no runs with steps.jsonl under: {root}")

    handler = type("_BoundHandler", (_Handler,), {"root": root, "single_run": single_run})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    scope = "run" if single_run else f"{len(discover_runs(root))} runs"
    print(f"VAW trace viewer: http://{args.host}:{args.port}  ({scope} under {root})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
