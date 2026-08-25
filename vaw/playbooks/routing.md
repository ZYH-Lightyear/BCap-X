# Routing

<!-- vaw:slot select -->
select/propose_pose 创建 planned 空间 Action 并缓存规划；当 Preview 与当前视觉证据足以判断动作安全
合理时，可以直接 commit。commit 执行当前空间 Action 的可执行缓存计划，planned 与 refined 均可，
Live References.action_proposal.executable 是能否 commit 的权威标志；false 时 commit 必然拒绝且不改变
世界，必须改用其他 seed、修改目标或 reject。同一组 seed 仍有效时，先尝试其中几何方向实质不同的
候选，不要重新 detection/propose 生成等价集合。TOP、PCA、CGN 只是候选来源，不存在固定优先级；
尤其对贴近支撑面的薄/扁物体，必须检查两指是否有支撑面净空，不能因为 top-down 看起来简单就默认选择。
局部几何不确定、需要连续观察或旋转时，用
call_imagination(action_id, instruction) 委派 Imagination，而不是重复盲目物理尝试。以下情形默认
先委派而不是直接 commit 或反复物理微调：容器放置、插入或上架类目标；载荷或机械臂遮挡 Contact View
使对齐不可判；载荷与目标沿口的间隙与载荷自身尺度相当；连续两次物理动作未改善同一对齐问题。
这些是路由建议而非门控——若多视角证据已一致且 Preview 清晰合理，仍可直接 commit；
它也不是 commit 的前置条件，refined 后仍须由你检查 Preview 再决定 commit。返回 status=partial
（reason=turn_limit）表示预算耗尽但已交回成果：每步编辑都通过了规划校验，Action 停在最后一次已验证的
编辑上，微调本身就有价值；审查 Preview 后可直接 commit、带更聚焦的 instruction 继续委派（从已
推进的状态继续）或 reject。返回 status=failed 时 ActionProposal 回滚到进入前的目标，仍可 commit 或
reject；失败原因和本 revision 的尝试记录留在 Live References：subagent_error 是内部错误，可原样
重试一次；geometry_unresolved 或 plan_unavailable 说明该目标不可解，不得用等价 instruction 再次
委派。只有回滚 Action 的 executable=true 才可直接 commit；否则应 reject、选择其他候选或引入实质
不同的几何目标。

<!-- /vaw:slot -->

<!-- vaw:slot amber -->
琥珀色 carried-volume 是可选的附着几何假设，不是所有抓取路径都会提供。存在时可用它做物体—容器
对齐；不存在时，不得给 Imagination 下达依赖“未来物体投影/落点”的不可观察停止条件，而应改为让
目标夹爪对齐到开口上方并保留安全净空，随后执行到高位并从新的真实 Canvas 闭环。

<!-- /vaw:slot -->
