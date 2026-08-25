# Recover

<!-- vaw:slot grounding -->
detection_and_sam 返回当前 observation 中身份和二维位置的权威 region；不得用自己的分类否定其 query，
但它不证明接触、抓持、支撑或包含。若 locate_point 的 query 属于一个已有 region，必须传
within_region_id，禁止脱离该 region 重新搜索相似物体。region、point、seed 和 action ID 只在 Live
References 中有效。每次物理动作后，region/point 会自动对照新画面复验：仍列出的条目即画面未变化，
可直接继续引用，不需要重新 detection/locate；标记 status=occluded 的条目暂被机械臂遮挡、几何未能
重新确认，使用前应结合当前 Canvas 判断；已发生变化的条目会被自动移除，Function Event 的
world change check 会说明哪些被移除、哪些仍然有效。seed 与 ActionProposal 仍是 revision-local。

<!-- /vaw:slot -->

<!-- vaw:slot restore -->
若一次
闭合后物体没有随动、机械臂遮挡目标或接触区已不可判断，可连续使用 base +Z 的
delta_move 小步抬升来恢复净空和可观察性，再决定如何重试；不要把重新 detection 当作机械恢复动作。
若刚执行的 move_to intent 是接近某个操作对象，当前优先问题是“局部几何是否支持下一个物理动作，
或是否需要抬升/退让恢复可观察性”。只要目标仍清楚可见于当前 Contact View、身份没有歧义且修正方向
可由当前坐标提示判断，就应直接进行局部修正或推进下一物理动作；
尤其当对象仍位于张开夹爪正下方、只是存在竖直净空时，应直接沿 Contact 卡片所示 BASE 方向小步接近，
不要仅因蓝轮廓已经消失而重新 detection；
只有目标离开局部视野、身份不确定或必须重新生成抓取方向时，才重新调用 detection_and_sam 和
propose_grasps。

<!-- /vaw:slot -->
