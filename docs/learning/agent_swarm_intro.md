# Agent Swarm: Concepts, Patterns, and How It Applies to RoboMEx

Agent Swarm is an engineering idea for building systems where multiple agents
coordinate to solve a task that is too broad, uncertain, or failure-prone for one
monolithic agent. In the recent LLM-agent literature, the term overlaps with
multi-agent systems, agent teams, agent-to-agent collaboration, and multi-agent
orchestration. The exact naming differs across frameworks, but the core question
is stable: instead of asking one agent to perceive, plan, code, verify, remember,
and execute everything, can we assign narrower responsibilities to multiple
agents and coordinate their outputs into a more reliable system?

This document explains the concept from first principles, then maps it to the
RoboMEx direction: skill-grounded robot coding agents with specialist SubAgents
for grounding, affordance, verification, and possibly motion planning.

## 1. Why Agent Swarm Exists

A single LLM agent is convenient because all context, reasoning, and tool use sit
inside one loop. But single-agent systems often fail in predictable ways:

- They overload one context window with unrelated concerns.
- They mix high-level planning with low-level execution details.
- They hallucinate missing evidence instead of asking a specialist process to
  gather it.
- They repeat failed strategies because there is no independent critic.
- They are hard to debug: when a run fails, it is unclear whether the root cause
  was perception, planning, affordance selection, motion execution, or
  verification.

Multi-agent systems try to reduce these problems by introducing role separation.
For example, AutoGen frames LLM applications as conversations among customizable
agents that can combine LLMs, tools, code execution, and human inputs. CAMEL
studies role-playing agents that cooperate through structured dialogue. AgentVerse
explores groups of agents that can collaborate and dynamically adjust their
composition. These systems differ in details, but they share one thesis: complex
tasks benefit from modular, interacting reasoning processes.

The word "swarm" adds two implications beyond ordinary "multi-agent":

1. The system may contain several specialized agents that can be invoked
   dynamically rather than as a fixed linear pipeline.
2. Coordination itself becomes part of the system design. You are not only
   designing prompts; you are designing who talks to whom, what evidence is
   exchanged, when an agent should stop, and how conflicting outputs are handled.

In practice, an Agent Swarm can be very small. A useful swarm might be only three
agents: a planner, an executor, and a verifier. The important point is not the
number of agents; it is whether responsibility boundaries produce better
reliability, interpretability, or reuse.

## 2. Agent, Tool, Skill, Workflow: Do Not Confuse Them

Before designing a swarm, separate four concepts.

An agent is an autonomous reasoning loop. It observes context, decides what to do
next, calls tools or emits messages, receives feedback, and eventually stops.
In LLM systems, an agent usually means an LLM plus a policy loop, a system prompt,
tools, memory, and a stopping rule.

A tool is a callable capability. Examples include `run_python`, `query_vlm`,
`solve_ik`, `segment_sam3_text_prompt`, or `goto_pose`. A tool should have a
clear input/output contract. It does not decide long-term strategy by itself.

A skill is reusable procedural knowledge. In Claude-style or Qwen-Code-style
systems, a skill is often a directory containing `SKILL.md`, scripts, references,
and assets. A skill is not necessarily executable by itself. It may tell the agent
when to use a method, what assumptions hold, what failure modes to watch, and
which helper script to run.

A workflow is a fixed or semi-fixed procedure. For example, "segment object,
compute OBB, choose top-down grasp, execute, verify." Workflows are easy to
control but less flexible. Agents are more flexible but less predictable. Many
production systems combine them: agents choose among workflows, and workflows
call tools.

The design mistake is to call everything an agent. If a component always performs
the same deterministic steps, it is probably a workflow or tool. If it reasons
about when and how to use multiple skills/tools under uncertainty, it is closer
to an agent.

## 3. Common Agent Swarm Architectures

### 3.1 Supervisor-Worker

One supervisor receives the task, decomposes it, delegates subtasks to workers,
and integrates results. This is the most common practical pattern. The supervisor
may be called Planner, Manager, Orchestrator, or Act Agent.

Benefits:

- Easy to debug because all decisions route through one place.
- Clear authority: only the supervisor can commit final actions.
- Works well when subtasks are specialized.

Risks:

- The supervisor becomes a bottleneck.
- Workers may be underused if the supervisor does not know when to delegate.
- If the supervisor is wrong, the whole system follows the wrong decomposition.

RoboMEx currently fits this pattern: Act is the central executor, while a
generic task-first SubAgent can provide focused localization, affordance,
placement, or state-checking analysis.

### 3.2 Peer-to-Peer Debate

Several agents discuss a problem and critique each other. This is common in
reasoning tasks, code review, and decision making. One agent proposes, another
criticizes, a third arbitrates.

Benefits:

- Can reduce shallow reasoning.
- Useful when the answer can be evaluated from text or structured evidence.
- Good for design review and failure analysis.

Risks:

- Expensive: every debate round costs tokens and time.
- Agents may converge on a plausible but wrong shared story.
- Without external ground truth, debate can amplify hallucination.

For robotics, pure debate is not enough. A verifier should inspect images, masks,
state, videos, or simulator reward, not just argue.

### 3.3 Blackboard / Shared Memory

Agents read and write to a shared evidence store. Instead of passing long chat
messages, each specialist publishes artifacts: masks, bounding boxes, grasp
candidates, overlays, verification judgments, and failure notes.

Benefits:

- Strong fit for multimodal robotics.
- Reduces repeated perception work.
- Makes debugging easier because intermediate artifacts are visible.

Risks:

- Requires naming discipline.
- Old evidence can become stale after the robot moves.
- Agents need rules for when evidence must be refreshed.

RoboMEx's `EVIDENCE` dictionary and output artifacts are a blackboard-style
design. This is promising and should become more central.

### 3.4 Hierarchical Agents

High-level agents plan over abstract goals; low-level agents handle concrete
operations. For example, a task planner decides "pick bowl then place on tray,"
while an affordance agent decides "top-down rim grasp at this point," and a
motion agent turns that into joint commands.

Benefits:

- Natural fit for robot tasks.
- Separates semantic planning from geometry and control.
- Allows different time scales: task planning changes slowly, perception changes
  every observation.

Risks:

- Interface design is hard.
- Too much hierarchy creates latency and error propagation.
- If each layer emits vague natural language, the system becomes untestable.

The practical rule is: high-level communication can be natural language, but
physical evidence should become structured data as soon as possible.

### 3.5 Dynamic Swarm

The system chooses which agents to instantiate based on the task. A simple can
pick may only need Act. A bowl task may call Grounding and Affordance. A drawer
task may call Grounding, Handle Affordance, Motion Critic, and Verifier.

Benefits:

- Avoids paying the cost of all agents on every task.
- Lets the system scale with task difficulty.
- Supports future extension.

Risks:

- Requires good delegation triggers.
- Harder to benchmark because traces vary by episode.
- Agent composition can become unstable without limits.

This is likely the right long-term direction for RoboMEx, but the first version
should stay small: Act + Grounding + Affordance + optional Verifier.

## 4. What Agent Swarm Is Good For

Agent Swarm is useful when a task has separable sources of uncertainty. In
robotics, these are obvious:

- Perception uncertainty: which object is the target?
- Grounding uncertainty: where is it in pixels and 3D?
- Affordance uncertainty: where can the robot grasp, push, pull, or place?
- Motion uncertainty: is the pose reachable and collision-safe?
- State uncertainty: did the object actually move, lift, open, or land inside the
  target?
- Recovery uncertainty: should the system retry, re-observe, switch strategy, or
  give up?

A single agent can reason about all of these, but it tends to blur them. A swarm
lets each uncertainty type have a specialist process and a specialist evidence
format.

However, Agent Swarm is not automatically better. It is often worse when:

- The task is simple and one agent can solve it cheaply.
- SubAgents only rephrase the same context without adding evidence.
- There is no structured output contract.
- Agents are allowed to execute conflicting actions.
- The orchestration policy creates long conversations before every action.

The engineering goal is not "more agents." The goal is better decomposition
under uncertainty.

## 5. Coordination Protocols

A swarm needs a protocol. This protocol answers five questions.

First, who has authority? In robot systems, only one component should execute
physical actions. If several agents can independently move the robot, collisions,
undoing behavior, and accidental release become likely. For RoboMEx, Act should
remain the only executor.

Second, what context can each delegated task observe? A localization task may
need RGB, depth, camera intrinsics, SAM/VLM tools, and output paths. An
affordance task may need segmented points, masks, geometry helpers, and IK
checks. A state-checking task may need before/after images and videos. Giving
every delegated worker every execution tool increases risk.

Third, what output must be structured? Natural language is fine for explanation,
but robot decisions need data: `bbox`, `mask_path`, `points_key`, `candidate_pose`,
`quat`, `score`, `ik_ok`, `overlay_path`, `verification_state`. If a SubAgent
returns prose only, Act must parse vague text and may hallucinate details.

Fourth, when does an agent stop? In AutoGen-like systems, agents stop via
conversation termination conditions. In tool-call systems, they stop by emitting
`finish`. In robotics, a SubAgent should stop when it has produced usable
evidence or a clear failure reason, not when it has "solved the whole task."

Fifth, how is failure handled? A good SubAgent result includes not only success
data but also why it may be unreliable: low mask score, stale observation, IK
failure, occlusion, ambiguous object identity, or no safe candidate. This is
where swarm systems become more robust than monolithic agents: failures become
typed and local.

## 6. Skills as the Memory of the Swarm

Skills are the long-term memory of an Agent Swarm. Without skills, SubAgents are
just role prompts. With skills, they become specialists that can reuse accumulated
physical knowledge.

For RoboMEx, the four skill categories are a strong foundation:

- `perception`: how to observe, ground, segment, and estimate state.
- `affordance`: how to propose actionable points, poses, grasps, pulls, or
  placements.
- `motion`: how to execute movement primitives and maintain safety.
- `task`: how to compose lower-level capabilities into manipulation workflows.

The distinction matters. A bowl grasp affordance skill should not open the
gripper. It should propose candidates, show overlays, and explain geometry. A
motion skill can execute. A task skill can tell Act how to compose perception,
affordance, motion, and verification.

This separation is how you avoid the common multi-agent failure where every
component tries to solve the entire task.

The newer "Swarm Skills" idea pushes this further: not only individual skills,
but also coordination patterns can be packaged and reused. For example, a
"BowlPickSwarm" skill could specify roles, sequence, outputs, stop conditions,
and evaluation criteria. That is conceptually close to where RoboMEx could go
after the basic SubAgent architecture is stable.

## 7. Applying Agent Swarm to RoboMEx

RoboMEx should not become a free-for-all swarm. It should be a controlled,
robot-safe swarm:

```text
Planner
  -> gives subgoal to Act

Act Agent
  -> reads task skills
  -> decides whether to call SubAgents
  -> owns all robot execution
  -> returns finish to Planner

Generic CodingAgentSubAgent(task, inputs)
  -> chooses relevant skills itself
  -> uses perception / affordance / motion / task guidance as needed
  -> returns compact evidence, candidates, state judgments, and artifacts
```

For the current project stage, avoid binding SubAgent profiles too early. State
checking can be added as another delegated task, but it should not become a
mandatory hard-coded loop. This avoids the earlier Act-Verifier rigidity while
preserving the useful idea of independent checking.

The key rule is:

Act decides when to call SubAgents; SubAgents do not execute motion; Skills give
domain knowledge; artifacts make the result inspectable.

## 8. Example: Bowl Grasp

A bowl task demonstrates why Agent Swarm helps.

A single agent might see "pick up the bowl," segment it, choose a grasp point,
move, close, lift, and declare success. If it fails, the trace may not explain
whether the problem was wrong object identity, bad mask, bad rim geometry,
infeasible IK, or false verification.

With a swarm:

1. Act loads `pick_object`.
2. Act calls a SubAgent: "localize the black bowl between the plate and ramekin."
3. The SubAgent returns mask, points, bbox, and overlay.
4. Act calls a SubAgent: "propose top-down rim candidates for this bowl."
5. The SubAgent loads `grasp_open_bowl`, runs the helper script, saves an overlay
   showing grasp point, approach axis, jaw axis, and IK status.
6. Act executes one candidate.
7. Act asks a SubAgent or VLM state check: "is the bowl visibly held after lift?"
8. If not, Act chooses another candidate or asks Affordance for a different
   strategy.

The benefit is not just better success. It is better diagnosis. If the overlay
shows points outside the bowl, the issue is affordance geometry. If the mask
selects the wrong object, the issue is grounding. If the grasp succeeds but the
agent opens the gripper while going home, the issue is motion safety policy.

This is exactly the kind of clarity that a swarm can provide.

## 9. Measuring Whether the Swarm Helps

Agent Swarm adds cost, so evaluation must include more than success rate.

Important metrics:

- Success rate: final task completion.
- First-attempt success: whether the first planned execution works.
- Recovery success: whether the system can recover after a failed attempt.
- LLM calls: total calls to planner, Act, and SubAgents.
- VLM calls: grounding/verification cost.
- Wall-clock latency: total episode time.
- Action blocks: number of executed robot code blocks.
- Tool failures: parsing errors, IK failures, segmentation failures.
- Evidence quality: whether artifacts contain usable masks, overlays, and
  structured candidates.
- Delegation rate: how often Act calls each SubAgent.
- SubAgent utility: success improvement per added call.

For RoboMEx, a clean ablation would be:

1. Single Act Agent + skills.
2. Act + one delegated localization task.
3. Act + localization + affordance delegation.
4. Act + localization + affordance + state-checking delegation.

Then report success, latency, token/cost, and failure categories. The method is
strong only if the added agents improve hard tasks enough to justify their cost.

## 10. Practical Design Rules

Use fewer agents than you think. Start with one generic task-first SubAgent and
add new delegated task patterns only when a repeated failure type cannot be
cleanly solved by skills alone.

Keep execution authority centralized. In robot settings, this is non-negotiable.
SubAgents can inspect, compute, score, and recommend. Act executes.

Make SubAgent outputs structured but task-specific. Use a compact JSON envelope
for the claim, uncertainty, and artifact paths; include physical evidence such as
coordinates, frame conventions, scores, and visualization paths only when the
delegated task needs them.

Prefer evidence over debate. A VLM conversation about whether a grasp is good is
less useful than an overlay with point, jaw axis, approach axis, mask, and IK
status.

Let skills carry physical knowledge. Prompts should not repeatedly explain bowl
rim geometry, cylinder side grasps, drawer pulls, or placement clearance. Put
that into skills and scripts.

Log raw request/response and artifacts. Multi-agent systems are harder to debug
without a trace UI. Every SubAgent call should leave enough evidence to answer:
what was asked, what was observed, what was returned, and why Act trusted it.

Do not force delegation. A swarm is useful because it can scale with difficulty.
If every task calls every agent, latency explodes and simple tasks become worse.

## 11. Risks and Failure Modes

Agent Swarm introduces new problems.

Coordination overhead: multiple agents can spend many turns negotiating without
moving the task forward.

Responsibility diffusion: each agent assumes another agent will verify or fix
the issue.

Stale evidence: a grounding result may become invalid after the robot moves.

Conflicting recommendations: two agents may propose incompatible strategies.

Prompt drift: agents may ignore their role and attempt full task execution.

Cost explosion: specialist calls, VLM calls, and retries can dominate runtime.

Safety hazards: if multiple agents can execute actions, one may undo another's
work. The "go home then open gripper" failure is a small example of why authority
and safety gates matter.

The solution is not to remove agents. The solution is to define narrow roles,
structured outputs, execution gates, and clear stop conditions.

## 12. How to Think About RoboMEx as a Method

The method can be stated as:

RoboMEx converts embodied manipulation from a monolithic Code-as-Policy loop into
a skill-grounded swarm where a central Act Agent dynamically delegates uncertain
perception and affordance reasoning to specialist CodingAgentSubAgents, then
executes grounded motion using structured evidence and inspectable artifacts.

This method claims three advantages:

1. Generalization: new task types can be handled by adding skills and specialists
   without rewriting the whole agent loop.
2. Success rate: hard tasks improve because grounding, affordance selection, and
   verification are separated and evidence-driven.
3. Cost control: the system does not call every specialist every time; delegation
   is conditional and artifacts/scripts reduce repeated reasoning.

The strongest version of the project is not "we built many agents." It is:

We identified the physical uncertainty boundaries in robot manipulation and
designed a controlled multi-agent architecture where each boundary has a
specialist evidence producer, while execution remains centralized and safe.

That is a clean research story and a practical engineering direction.

## References

- AutoGen paper: https://arxiv.org/abs/2308.08155
- AgentVerse paper: https://arxiv.org/abs/2308.10848
- CAMEL paper: https://arxiv.org/abs/2303.17760
- Multi-Agent Collaboration Mechanisms survey: https://arxiv.org/abs/2501.06322
- AutoGen Studio paper: https://arxiv.org/abs/2408.15247
- Swarm Skills paper: https://arxiv.org/abs/2605.10052
- LangGraph multi-agent documentation: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
- OpenAI Swarm repository: https://github.com/openai/swarm
- CrewAI repository: https://github.com/crewAIInc/crewAI
- RATs local reference: `third_party/RATs`
