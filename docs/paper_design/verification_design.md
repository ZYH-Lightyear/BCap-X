# 验证子系统:实现框架设计

## 原则
不破坏两个 agentic 概念:**skill = 文档**(只声明意图与判据,不写死流程)、**coding agent = 自主写代码**。验证因此不是写死的管线,而是一个独立的**验证 Coding Agent** 在证据上自主判断。

## 三层解耦:Spec / Evidence / Judge
1. **Spec(规格)= `ref/verify.md`**:纯文档,声明"什么算达成"。面向 VLM/人,**不含取证代码**。
2. **Evidence(证据)= 执行期捕获 + 持久沙箱状态**。**证据是 skill 类型相关的,不是统一 before/after**:
   - 框架自动抓子目标净帧,以约定名暴露:`OBS_BEFORE` / `OBS_AFTER`(并存 PNG);
   - skill 把关键结构化中间量发布到持久的 `EVIDENCE` 字典(如 `EVIDENCE['target_box']`、`EVIDENCE['gripper_width']`);沙箱 globals 跨 block 持久,验证器直接读。
3. **Judge(裁决)= `VerifyCodeAgent`**:子目标结束后**只跑一次**。读 `VerifierContext`(子目标 + 脱敏 op-trace + verify.md 配方 + 证据约定名),自主写代码、用沙箱已有的 `query_vlm` 判断,输出裸 JSON verdict。

## 粒度:逐块"存",整目标"判一次"
- 块级正确性 → 沙箱 `ok`/`stderr` + skill 内 `assert`(廉价自纠错,驱动内层循环);
- 子目标级达成 → 验证器一次性语义视觉裁决,喂 planner。
中间块(分割、算位姿)没有可视觉判断的独立成败,故不逐块判;但逐块存的证据让那唯一一次裁决既能看净 before/after,也能翻看时间线。

## 结果如何用
`verdict ∈ {passed, failed, uncertain}` + `reason`:
- `passed → SubGoalResult.success=True`;`failed/uncertain → False`。
- **`reason` 永远进 planner history**,反应式 planner 据此重试 / 改写描述 / 推进。
- 验证器是子目标成败的**唯一权威**;内层 `FINISH` 仅结束内层循环,不等于成功。

## 删除与简化(让 agentic 不被代码污染)
- **删 `TaskSignalVerifier`**:env 信号占位,对 success 无贡献。
- **删 `vlm_judge` 原语 + `VERIFY_PRIMITIVES`/`_setup` seeding**:验证器直接用命名空间里已有的 `query_vlm` 自己写判断代码。
- **`verify.py` 不作要求**:降级为 EvolveAgent 未来可选蒸馏。
结果:每个 skill 只交付**文档**(SKILL.md + ref/verify.md),验证器只交付"判断意图"的能力。

## `verify.md`:与 SKILL.md 对称的验证配方
verify.md **不是死 rubric,而是像 SKILL.md 一样的建议性配方**:含 rubric + **示例代码**,告诉验证器"看哪类证据、怎么标注、怎么问 VLM"。验证器读它、改写它、自己写代码——agentic 概念两端对称。证据**按类型而非统一模板**组织,且**只引用约定名**(`OBS_*`、`EVIDENCE[...]`),不写绝对路径。

每个 verify.md 含:Goal / Pass criteria(可视觉判定)/ 证据 + 示例代码 / Fail signals。

**铁律:标注不能污染被判物体的像素。** 允许画**外框 bounding box** 或把 **bbox 坐标作为文本**喂 VLM;**禁止**把 mask 用颜色**填涂**在物体上(破坏 VLM 对物体本体的识别)。

示例——observe `segment_object`(单帧判定,不用 before/after):
```python
box = EVIDENCE["target_box"]                       # 分割时已发布
annotated = draw_box(OBS_AFTER.copy(), box)        # 只画外框,不填涂
v = query_vlm(f"框内物体是否为 '{object_name}'?只回 JSON{{verdict,confidence,reason}}",
              images=[annotated])
```
示例——action `pick_object`(净 before/after 判定):
```python
v = query_vlm(f"对比两图:'{object_name}' 是否已被夹爪抓起并离开桌面?只回 JSON{{...}}",
              images=[OBS_BEFORE, OBS_AFTER])
```

## 证据存储形式
- 逐块原始帧:`subgoal_NN/turn_01.before.png` / `turn_01.after.png`(与 `turn_01.py` 同级,人/agent 易对照)——调试产物。
- 验证器运行期输入:持久 globals 里的 `OBS_BEFORE` / `OBS_AFTER` + `EVIDENCE` 字典(零磁盘往返)。

## 接口改动清单
- 框架在子目标始/末暴露 `OBS_BEFORE`/`OBS_AFTER`(变量 + PNG);种空 `EVIDENCE = {}`;提供 `draw_box`/可选 `render_diff`。
- `EvidenceCollector`:逐块存 `turn_NN.before/after.png` 到 sub-goal 目录。
- `session.py`:子目标后组 `VerifierContext` → 跑验证器 → verdict 映射成 `success` + `reason` → 落盘。
- `VerifyCodeAgent`:去掉 `_setup` seeding;system prompt 含"标注不污染像素"铁律 + "用 `query_vlm` 自行取证、FINISH 输出裸 JSON"。
- 三个 `verify.md`:按上面配方重写(segment 单帧标框、pick/grasp 净 before/after)。
