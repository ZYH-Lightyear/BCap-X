# VAW Progress Critic Video Identity

## Style Prompt

克制、精确的机器人实验诊断画面。真实 AgentView 是视觉主体，进度曲线是第二视觉焦点；界面应像一件经过校准的科研仪器，而不是通用 Web dashboard。动效只用于同步时间、揭示新证据和强调进度突变。

## Colors

- Background: `#071018`
- Panel: `#0E1A24`
- Foreground: `#E8F0F5`
- Muted: `#91A4B2`
- VAW accent: `#35B8D0`
- Progress: `#38C786`
- Regression: `#F05D5E`
- Unknown: `#E3B341`

## Typography

- Interface and task text: `IBM Plex Sans`, weights 350 and 700
- Measurements and function names: `IBM Plex Mono`, weights 400 and 600

## Motion

- Playhead is linear and physically synchronized with the source video.
- A score point is revealed only after its action segment ends.
- Score changes draw in over 0.18 seconds with restrained ease-out motion.
- No decorative looping animation.

## What NOT to Do

- No gradient text, neon glow, grain, floating cards, or ornamental particles.
- No dense gridlines, legends, multi-axis charts, or future-score ghost curves.
- Do not crop the source AgentView or cover task-relevant pixels.
- Do not display progress before the AFTER evidence exists.
