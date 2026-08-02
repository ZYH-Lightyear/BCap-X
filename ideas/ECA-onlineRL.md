# ECA-onlineRL vs CaP-RL

Embodied Compositional Agent online RL：多轮参数化 skill tool-call + 阶段 schema 过程奖励 + 终局 binary。下表与结论只保留实现相关事实，并与 CaP-RL 对照。

---

## 1. MDP

| | **CaP-RL** | **ECA-onlineRL（多轮 tool-use）** |
|--|------------|----------------------------------|
| LM 决策次数 | **1 次** | **多次**（每 skill 一步） |
| 状态 *s* | 固定 prompt（任务 + API docstring）+ seed | 任务描述 + 历史 calls + **每步 skill 执行后观测/verifier**（可选 VDM 文本） |
| 动作 *a* | 整段 Python 程序（≤~1k token） | 短结构化 tool-call：`skill` + JSON args（可加短 think）；受 phase schema 约束 |
| 转移 | 生成结束后 `env.reset` + 一次 `exec(code)` | **交织**：generate call → 执行 skill → obs/r 写回 → 再 generate |
| 奖励 *r* | 整段仿真后标量（见下式） | 每步 verifier + 终局 binary（见下式） |
| 终止 | EOS / max length；仿真超时（~90s） | `finish` / 任务成功 / 非法超限 / `max_assistant_turns` |
| LM 侧 MDP 形态 | **单步 bandit**（仿真多步压进一次 `step`） | **多步 MDP**（机器人控制仍在 skill 内部） |

**CaP-RL 奖励整形**

$$
\text{score} =
\begin{cases}
\max(R,\,0.1) & \text{if sandbox\_ok} \\
\max(R,\,0) & \text{otherwise}
\end{cases}
$$

**ECA 轨迹奖励**

$$
R_{\text{traj}} = R + \lambda \sum_{t=1}^{T} r_t - c \cdot \mathbf{1}_{\text{illegal}}
$$

其中终局 $R \in \{0,1\}$，$r_t$ 为第 $t$ 步硬 verifier 过程奖励，$\lambda$ 宜小，$c$ 为非法转移惩罚。

Skill Schema（阶段先验，非闭集纯分类）：

`perceive(3D) → propose → approach → manipulate(pre) → commit → manipulate(post) → coordinate`

- 用于合法转移 mask、阶段门控奖励、失败诊断。
- 动作须为 **skill × 参数**，不是仅 skill ID。

---

## 2. 算法 / 更新

| | **CaP-RL** | **ECA-onlineRL** |
|--|------------|------------------|
| 算法 | GRPO，无 critic | GRPO（长程可用 TGRPO：步级+轨迹级 advantage） |
| 组定义 | 同 prompt 的 $G$ 条完整程序 | 同 `(task, seed)` 的 $G$ 条**多轮轨迹** |
| Advantage | outcome $R$ 组内归一化，广播到整段 response | 轨迹回报组内相对化；**仅 assistant token** 有 loss |
| KL | `use_kl_loss=True`（$\beta \approx 0.02$），`use_kl_in_reward=False` | 同：KL 进 loss；多轮更易漂移，$\beta$ 或 curriculum 可略加强 |
| 在线/离线 | prompt parquet **离线**；生成与仿真打分 **在线 on-policy** | `(task,seed)` 可离线；**rollout 中环境必须在线**（主路径）；可先 SFT 热身 |
| 过程奖励约束 | 无（仅 outcome） | $\lambda$ 小；硬断言 verifier；防刷分（如反复 perceive） |

**GRPO outcome advantage**（组内相对化；CaP 用标量 $R$，ECA 用 $R_{\text{traj}}$）

$$
A_i = \frac{R_i - \mathrm{mean}(\{R_j\}_{j \in \mathrm{group}})}{\mathrm{std}(\{R_j\}_{j \in \mathrm{group}}) + \varepsilon}
$$

再按 `response_mask`（ECA：仅 assistant 段）广播到 token；$A_i$ 进入 clipped policy gradient。

**策略损失（两边共用形态）**

$$
L = L_{\mathrm{PG}}^{\mathrm{clip}} + \beta \, \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})
$$

训练循环（ECA）：

```text
(task, seed) → 采 G 条多轮轨迹
  [think → tool-call → skill exec → obs, r_t] × T → R_terminal
→ group-normalize advantage → clipped PG + β KL_ref → update_actor
```

Token mask：tool / obs 写入 messages，但 `loss_mask = 0`；只对 assistant 生成段回传梯度。

---

## 3. 环境与 Skill 执行

| | **CaP-RL** | **ECA-onlineRL** |
|--|------------|------------------|
| Env API | `reset` + 一次 `step(code_str)` | `reset` + 反复 `apply(skill, args)` / `step(skill_call)` |
| 状态持久 | 单次 exec 内 globals | **跨轮持久**：世界/夹爪/持物必须保留 |
| 超时 | 整段 ~90s | 每 skill 短超时 + 总步数上限 |
| 并行 | 生成与 reward 易 pipeline；`GROUP_SIZE` × workers | 每 traj 绑定独立 env（Ray actor / 池）；禁止未隔离共用仿真 |
| 训练感知 | 常用 S1 特权 API 稳奖励 | 训练可用 privileged verifier；部署换视觉 skill |
| 实现锚点 | `verl_agent_reward/capx_franka_reward.py` + `env.step(code)` | VeRL `multi_turn` + tool executor → Cap-X / robomex skills |

---

## 4. Prompt / Tool Schema

| | **CaP-RL** | **ECA-onlineRL** |
|--|------------|------------------|
| System | 诱导一次写全程序；API docstring + 示例 | 任务目标 + **Skill Schema 阶段与合法转移**；禁止每步贴完整 Python 教程 |
| Tools | 隐式：sandbox 内 API 函数 | 显式 OpenAI-style tools：每 skill 的 JSON schema（参数类型、必填） |
| 每步上下文 | 无（单轮） | 上一执行结果：stdout、verifier、短 scene 摘要 |
| 输出形态 | 完整 Python 代码块 | 短 think（阶段/对象/理由）+ 结构化 tool-call |
| 策略规模（事实） | 7B coder（文献/脚本默认） | 文献主流 **7B–32B**；不宜用 397B 做逐步短 call |

短输出不必然浪费：算力多在 **prefill（读上下文决策）** 与 **环境**；浪费发生在「超大模型 + 无 think 的纯 skill ID 分类」。

---

## 5. 训练课程

| 阶段 | **CaP-RL** | **ECA-onlineRL** |
|------|------------|------------------|
| 冷启动 | 底座 coder 已会写代码，直接 GRPO | **SFT 热身**：合法 call 轨迹（可由 oracle / CaP 轨迹蒸馏） |
| 主训 | 单任务 GRPO，~50 epoch；$G \approx 15$ | 先短视界（2–4 skill）→ 再加长；先 verifier 为主 → 再加大终局权重 |
| 多任务 | 脚本级 **每任务独立训** | **必须 mixed multi-task**，否则过拟合单任务阶段顺序 |
| 评测拆分 | 同任务 sim / real | ID 任务 / 组合未见 / 需新参数 / 需新 skill（后者应失败或扩库） |

---

## 6. 部署

| | **CaP-RL** | **ECA-onlineRL** |
|--|------------|------------------|
| 推理环 | vLLM 出一段代码 → sandbox **一次**执行 | Agent loop：生成 call → 执行 skill → 观测回灌 → 再问模型 |
| 延迟 | 一次长生成 + 一次长执行 | 多次短生成 + 多次短执行；可早停/纠错，总延迟可能更高 |
| 未见任务 | 靠代码组合 API / 控制流 | 靠 **skill 组合 + 填参**；库外能力需扩 tool 或保留稀有 `emit_code` |
| 真机 | 共享感知/控制 API → sim-to-real 间隙小（论文设定） | 同：智能在组合层，低层 skill 跨 sim/real 固定实现 |

---

## 7. 算力

| | **CaP-RL** | **ECA-onlineRL** |
|--|------------|------------------|
| 交互次数（同 batch、$G$） | $B \times G$ 次「长生成 + 1 次仿真」 | $B \times G \times T$ 次「短生成 + skill」≈ **×$T$** |
| Decode | 长，贵 | 短，相对便宜 |
| Prefill | 通常一次 | 每轮累积上下文，随 $T$ 涨（需截断/摘要） |
| 工程瓶颈 | 仿真吞吐、FSDP merge | **env 调度、状态隔离、轨迹超时杀进程** |
| 策略模型 | 7B 级可训 | 7B（或 14B）VLM/LLM；超大模型仅作教师/偶发 escalate |

粗算：取 $B=256,\ G=8,\ T=6$，则

$$
\frac{N_{\mathrm{ECA}}}{N_{\mathrm{CaP}}} \approx T = 6
$$

即 ECA 交互次数约为 CaP-RL 的约 6 倍；token 总量未必 6 倍（每次 decode 更短）。

---

## 8. 强调的钩子（论文/实现必须钉死）

不要宣称「首个机器人 skill-level GRPO」（REVER/RoboFarseer 已占位）。可辩护差异与可发钩子：

1. **动作空间钩子（vs CaP-RL）**  
   从 **单轮 code-bandit** 改为 **在线交织的参数化 skill tool-call MDP**，用过程信用分配换可诊断、可组合的多任务迁移。

2. **奖励钩子（vs REVER）**  
   奖励来自 **在线物理/硬 verifier 执行结果**，不是离线 GT skill 序列匹配；过程项 $\lambda$ 小、终局 binary 主导，并做 no-hacking 消融。

3. **Schema 钩子（须做成算法/归纳偏置，而非名词列表）**  
   `perceive → … → coordinate` 用于 **phase-gated mask / 阶段门控 advantage（或 phase-gated GRPO）**；消融证明：有 schema 门控才有 hold-out 组合泛化，而不是手写阶段名本身。

4. **实证钉子（顶会门槛）**  
   多任务 hold-out 上，在匹配算力下系统对比：**ECA vs CaP-RL vs REVER-style plan RL vs frozen SayCan**；拆开「组合未见 / 需新参数 / 需新 skill」失败模式。

5. **可选升格钩子**  
   学习或演化 schema / 与可执行 skill 实现共进化；或稀有 `emit_code` 作 escape hatch——主路径仍是短 tool-call。

**一句话定位**：ECA-onlineRL = Cap-X 上的 **Skill-RLVR**——在线交织、schema 门控、参数化 tool-use；卖的是 **相对 code-as-policy RL 的迁移与信用分配边界**，不是「又一个 GRPO」。
