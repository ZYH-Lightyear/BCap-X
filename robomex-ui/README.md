# RoboMEx Swarm Observatory

独立、常驻、只读的 Agent Swarm 可观测界面。它以 ActionIntent 为主导航，
展示 Planner、Manager、真实 Swarm DAG、Agent 代码/结果、Candidate Arena、
Selector 和物理执行证据，并通过 SSE 尾随 `trace/events.jsonl`。

## 推荐：跟 RoboMEx tmux 一键启动

```bash
scripts/serve_up.sh
# 或先停再启
scripts/serve_down.sh && scripts/serve_up.sh
```

会多开一个 tmux window `trace`，监听 `0.0.0.0:8300`（可用 `TRACE_PORT` / `TRACE_ROOT` 覆盖）。
若 `robomex-ui/dist` 不存在，启动时会自动 `npm install && npm run build`。

在你自己的浏览器打开（**不要用本机 localhost**）：

```text
http://<服务器IP>:8300
```

云平台需放行安全组端口 **8300**。若不能直连，用 SSH 隧道：

```bash
ssh -L 8300:127.0.0.1:8300 <user>@<服务器>
# 然后打开 http://localhost:8300
```

## 手动单端口启动

```bash
cd robomex-ui && npm run build
cd .. && robomex ui --host 0.0.0.0 --port 8300 --root outputs/robomex_libero_live
```

## 可选：Vite 热更新开发（:5174）

```bash
robomex ui --host 0.0.0.0 --port 8300 --root outputs/robomex_libero_live
cd robomex-ui && npm run dev
# 浏览器: http://<服务器IP>:5174
```

## 说明

- 后端：`robomex/web`（FastAPI v2，默认 `0.0.0.0:8300`）
- 前端：本目录；`npm run build` 产物由 API 挂载在 `/`
- 数据源：`RunObservabilityStore` 生成的 v2 run bundle
- 不修改 Session / Swarm 执行逻辑，不提供物理控制，不生成静态 HTML report
