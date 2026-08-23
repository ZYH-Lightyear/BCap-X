# Canvas 可观察性审计矩阵(B0)

> 目的:在 Canvas 契约冻结(M1.6 出口)之前,系统性回答一个问题——
> **Agent 在每个决策时刻需要判定的自由度,是否在至少一个视图中"位移 1cm 产生可分辨的像素变化"?**
> 红格(不可分辨)直接驱动 B1–B4 的实现优先级;矩阵填满且无红格(或红格有明确豁免理由)
> 是 Canvas 冻结的前提。
>
> 状态:骨架 + 已知红格(2026-08-20);待 trace 逐帧标注补全。

## 1. 矩阵维度

- **任务族**(LIBERO-PRO 按目标形态聚类):
  - F1 开口容器放置(basket / bin / pot)
  - F2 平面放置(plate / stove / 指定桌区)
  - F3 插入-上架(book-shelf / 竖直插入)
  - F4 抽屉-铰链(drawer / cabinet 开合)
- **决策时刻**:
  - T1 pre-grasp 选点 · T2 grasp 后确认 · T3 carry 途中 ·
    T4 align(XY)· T5 descend(Z)· T6 release 前
- **被控自由度**:该时刻正在判定的量(XY 横移 / Z 高度 / yaw / 开合 / 附着状态)。

判据统一为:**1cm(或 5°)的变化,在至少一个当前可用视图中产生人眼可分辨的像素差**。
可用视图 = AGENTVIEW、OPPOSITE VIEW、CONTACT FRONT/SIDE(v46 起携带载荷时 SIDE 为 55° 斜俯视)、(v42 起)plumb line 标注。

## 2. 当前矩阵(F1 开口容器放置,依据 m16q/m16r/m16l trace)

| 时刻 | 被控自由度 | AGENTVIEW | OPPOSITE | CONTACT F/S | 结论 |
| --- | --- | --- | --- | --- | --- |
| T1 pre-grasp | 抓点 XY | 可分辨 | 可分辨 | 可分辨 | 绿 |
| T2 grasp 确认 | 附着/开度 | 部分(遮挡) | 部分 | 可分辨(GRIP+接触) | 黄 |
| T3 carry | 载荷姿态 | 可分辨 | 可分辨 | **载荷与目标不同框** | 黄 |
| T4 align | 载荷-开口 XY | **深度歧义,1cm 不可分辨** | 视角依赖 | **不同框,无法对齐判断** | **红 → B1/B2** |
| T5 descend | 离面/离沿 Z | 斜视角压缩,难判 | 同左 | 载荷底沿可见性不稳定 | **红 → B1** |
| T6 release 前 | 越沿判定 | 不可判(m16r 根因) | 部分 | 遮挡 | **红 → B1/B3** |

**v42 plumb line 对红格的覆盖**(2026-08-20 已实现):

- T4:落点→目标中心的 dXY 箭头 + cm 标注,把"XY 对齐"从深度猜测变成读数 → 预期转绿,
  待 m16r 同任务复跑确认。
- T5:H 标注给出载荷底面离支撑面的显式高度 → 预期转绿。
- T6:落点足迹多边形落进开口轮廓内=越沿的必要条件可视化;充分条件仍需互补视角
  (B3 近顶视图补充)。

## 3. 已知红格(其他任务族,来源:trace 观察 + 任务几何分析)

| 任务族 | 红格 | 驱动的 Canvas 项 |
| --- | --- | --- |
| F1 | T4/T5/T6(见上表) | B1(已实现)/ B2 / B3 |
| F3 书上架 | T4:左右视角被架体遮挡,书-缝隙关系不可判 | B4 任务轴向视图 |
| F3 | T5:插入深度在正视角无像素差 | B2 双主体取景 |
| F4 抽屉 | T2/T6:把手抓持与开合行程,agentview 近似切向 | 待评估(可能仅需 B2) |
| F2 平面放置 | 初判无红格(平面目标大、遮挡少) | 待空跑确认 |

## 4. 填格方法(补全剩余格子的确定性流程)

1. 从 `vaw/out/libero_pro_object_eval/` 现有 trace(m16q/m16r/m16l/m16m/m16n)
   按决策时刻切帧;每格标注:可用视图 × 1cm 位移的像素差是否可分辨(人工判定,双人交叉)。
2. F2/F4 无现成 trace 的格子:每族挑 1 个任务空跑采样(无 LLM,脚本推动到指定相位截图)。
3. 每格记录:绿/黄/红 + 证据帧路径 + 一句理由。红格必须映射到一个 B 项或一条显式豁免。

## 5. m16t v43 全套件扫描的实测红格(2026-08-20,task 0–9 各 1 seed)

结果:4/10 成功(task 0/1/6/9),6 个失败全部是 `max_turns` 耗尽——没有崩溃、没有
错误终止,预算被下面四类可观测性缺口吃掉。按实测代价排序:

| # | 缺口 | 实测代价 | 证据 | 修复项 |
| --- | --- | --- | --- | --- |
| G1 | **seed 可执行性在目录中不可见**:SeedSpec 只有 `solveIk`(常年 `returned`),完整规划验证(轨迹/碰撞)要 select 之后才在 action.executable 暴露;prompt 又要求"同一组 seed 先试不同候选",形成合法死循环 | task 8:**30/32 轮全是 select**,一次抓取未尝试,整局报废;task 7:5 轮 | task8 steps.jsonl;task7 turn3–5/9–10 | **B6**:seed 目录直接标 `executable`(数据已在 `SeedArtifacts.preview_plan.prediction` 里);全组不可执行时给 advisory |
| G2 | **plumb H 语义在容器内误读**:H=载荷底面到"正下方第一个表面"的距离;载荷进入篮口后该表面变成篮底/内衬,H 恒>0;模型把 H>0 读作"仍在开口平面上方",永远不满足释放判据 | task 4:瓶子实际已入篮(两 Contact 视图下半身没入篮沿后方),仍连降 10 轮不释放,耗尽预算;m16s-s2 crashed run 同因 | task4 turn20–32,context_0030.png | **B7**:对开口容器目标渲染"沿口平面相对深度"(载荷底面在沿口平面上/下 X cm),或 H 交点落于容器内部时改标 "H→篮底" |
| G3 | **附着假设自我证实**:碰撞假设体积刚性随 TCP,plumb 从假设底面下垂;抓取实际失败时,H 读数(假设底面→未被抓走的真实物体顶面)反而给出"已抓起、离面 2.2cm"的合理小数,确认了错误信念。而世界改变校验其实一直在复验源 region 未动——机器可判的矛盾没有被呈现 | task 7:空手抬升 6 轮(18cm),turn 29 仍信"已夹持";另 task 5 的 5 次 close/open 循环同属附着验证缺口 | task7 context_0028.png、turn23–29 | **B8**:抬升后源 region 仍在原位复验通过 × 附着假设声称随动 → 矛盾 advisory("attachment contradicted: source region unchanged after N cm lift") |
| G4 | **被顶住的物理动作标记为 completed**:命令 -3cm 下降、`position_error_m≈0.03`(实际零位移),outcome 仍是 `completed`;同时 GRIP 0.475→0.445(瓶被顶得在指间滑移)。"载荷已被支撑,应释放"的最强信号完全静默 | task 4 turn 29–30:瓶已坐进篮内被夹爪下压,正确动作是 open_gripper,agent 却去重新 detection | task4 turn28–30 result | **R1**(runtime,非渲染):`position_error ≥ 0.8×‖命令‖` → outcome=blocked + 显式消息"手臂未移动,疑似接触,载荷可能已被支撑";GRIP 下降趋势并入同一 advisory |
| G5 | 3cm/轴 + 一轮一步的运输税:长距离纯抬升/下降靠 delta_move 链,每步一整轮 | task 3:20/32 轮是 delta_move;task 7:6 轮连抬 | task3/7 steps.jsonl | prompt 层:>6cm 的位移应 propose_pose+commit;或允许纯 +Z 恢复动作放宽步长 |
|  | 对照组:成功的 task 1/6/9 恰好都没踩中 G1–G4(seed 首选即 executable、释放时 H 判读未遇容器内歧义) | | | |

v43 本身工作正常的证据:A2 盲降 advisory 全部触发且被响应(task4 turn31 转 grounding)、
A3 路由在 task2/5 触发 imagination、partial 交回在 crashed run 正确交回、anchor dXY
读数准确(task4 turn24 dXY 0.7cm 与足迹一致)。失败不再来自这些机制,而来自上表缺口。

**2026-08-21 补充(task4 context_0025 复盘 → v44 B3 落地)**:task4 的横向盲区比 G2
更基本——turn 25–30 期间两个 Contact 视角均近水平,落点足迹与篮沿的内外关系在投影中
退化成一条线,thought 反复写"plumb 交点在开口内部"而实际足迹压在沿口(命令 -3cm、H 卡在
1.5cm 不动即 G4 证据);OPPOSITE VIEW 全程距离过远,未提供任何横向证据。v44 已实现 B3:
携带载荷时右上面板切换为 LANDING TOP-DOWN(落点正上方 live 俯视 + 足迹/dXY/±X±Y 罗盘),
schema `vaw-context-v44-landing-topdown`。原 T5/T6 红格中"横向对齐不可判"的部分由 B3
接管;G2(H 语义)与 G4(受阻运动)仍待 B7/R1。

**2026-08-21 v44 失效复盘 → v45(RVT-2 虚拟视图引擎)**:m16u task4 `context_0010.png` /
`context_0016.png` 显示 v44 物理俯视相机被机械腕与载荷完全挡住,俯视面板无场景信息。
根因是"在机器人正上方放一台 MuJoCo 相机"——携带相位腕部必在光路上。v45 改为点云重投影
(RVT arXiv:2306.14896, RVT-2 arXiv:2406.08545):传感器 RGB-D + URDF FK 剔除机器人,
准正交针孔渲染成标准 camera dict,平时 GLOBAL、携带时 LANDING ZOOM。

**2026-08-21 v45 失效复盘 → v46(斜俯视 Contact SIDE)**:m16v task4 `context_0001.png`
显示虚拟俯视是带空洞的碎点图,coverage 长期不过闸而回退 OPPOSITE VIEW。根因不在引擎
而在输入:RVT-2 的视点自由建立在多相机稠密点云上,本 setup 只有 agentview + wrist 两台,
桌面后方、筐内与物体顶面从未被采样,FK 剔除机器人只会把这些区域变成洞。**只要物理相机
不增加,任何合成俯视都无法给 VLM 可解析的画面。**

v46 因此放弃合成视图,把视点自由度放回本来就具备的地方:Contact 相机本就可以摆在任意
位姿,`_render_rgbd_camera` 本就接受 `elevation_m`(OPPOSITE 一直在用 0.38),选相机时
本就在 4 组方位正负号里做可见性搜索——只是被一条"不得让视角变斜"的设计约束冻在水平面。
真正的病因是**两个近水平面板只能就高度互相印证,横向对齐不可观测**,于是 agent 只会调 Z。

- `ContactCameraRequest.side_elevation_deg`:SIDE 沿锁定方位轴抬升,FRONT 保持水平;
- 携带载荷时取 55°;抓取相位保持 0°(指间净空只有从水平才量得准);
- 抬升沿球面进行(半径 = framing distance),斜面板与水平面板同一变焦;
- 仰角切换时清掉 session 方位锁,让可见性搜索在新仰角下重新避开手臂;
- 面板标题标出 `OBLIQUE 55° DOWN`,MOVE BASE 罗盘按实际相机位姿投影,符号仍对应 delta_move;
- 提示词明确:横向读斜面板,高度读水平面板与 H;斜面板的画面上下混合了高度与进深。

**待验证**:m16w task4/task7 复跑,看 T4/T5 是否出现 XY 修正(v43–v45 全程只有 Z)。
若 55° 仍不足以分辨沿口内外,下一步是加物理相机而不是再合成视图。

## 6. 冻结判据(M1.6 出口)

- 矩阵全部格子有标注;
- 无未映射红格;
- B1–B4 实现后,原红格复测转绿(用同一批证据帧 + 新渲染重放);
- 之后 Canvas 渲染进入契约层冻结,RSI 进化不再触碰渲染代码(见 M1_6 计划 §6)。
