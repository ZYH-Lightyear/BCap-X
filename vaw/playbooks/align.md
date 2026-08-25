# Align

<!-- vaw:slot main -->
locate_point 只提供当前视觉中的粗 metric anchor；容器开口、边缘和深度噪声可能使点落在边沿，固定
offset 也不一定是最终位姿。允许根据当前 Contact View 对 point-derived Action 做有方向依据的适量微调。
但若 query 明确是语义区域的 center/opening center，且对应 Action 已成功到达，优先保留该 metric anchor
的 BASE XY：不同高度造成的透视偏移不能单独作为横移依据。斜俯视 Contact 显示足迹偏出目标，或两个
Contact 面板形成一致证据时即可修改 XY。横向证据矛盾或不足时，被禁止的是继续下降和释放，而不是修正本身；
修正方向必须来自当前可见几何，而不是重复同一感知循环或按次数切换策略。
<!-- /vaw:slot -->
