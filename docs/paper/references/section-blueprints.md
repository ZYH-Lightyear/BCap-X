# Section Blueprints

These are rhetorical blueprints, not rigid templates. Adapt length and headings to the venue.

## Abstract

Use a compact seven-move arc:

1. establish the setting and why it matters;
2. state the bottleneck, tension, or missing perspective;
3. reveal the key observation or conceptual hinge;
4. state the paper's question or purpose;
5. introduce the named method and its one-sentence mechanism;
6. report the strongest evidence across meaningful axes;
7. end with the bounded implication for the field.

Prefer 150–230 words unless the venue specifies otherwise. Give more space to the hinge and mechanism than to background. Include exact numbers only when they are verified and unusually informative.

Useful ending shape: “These results suggest that [new perspective/practice] provides a promising direction for [broader goal].” Do not end with generic “state-of-the-art performance.”

## Introduction

### Move 1: Important setting

Open with the capability or paradigm, then connect it to a concrete need. Keep broad history short.

### Move 2: Practical bottleneck

Use “However” to state a concrete failure, cost, or generalization gap. Explain why it matters in deployment or scientific understanding.

### Move 3: Mechanistic diagnosis

Name the hidden assumption, information gap, memory conflict, model drift, or missing degree of freedom. Use a small example or figure when it makes the diagnosis immediate.

### Move 4: Adjacent insight or decomposition

Introduce the analogy, equivalence, or two-axis decomposition. Explain both the similarity and the crucial gap; this prevents a superficial transfer story.

### Move 5: Explicit question

Ask one focused question when it sharpens the narrative. A displayed question is appropriate for an investigation paper. It should be answerable by the remaining paper.

### Move 6: Answer and method

State the hypothesis or finding before listing modules. Introduce the method name, its central mechanism, and why it addresses the diagnosis. Supporting modules should map visibly to previously named problems.

### Move 7: Evidence preview

Summarize the result pattern across relevant settings. Mention breadth by dimensions—models, datasets, heterogeneity, sequence length, OOD behavior, downstream quality—not by a raw count alone.

### Move 8: Contributions

Usually use three bullets:

- **Insight/formulation:** what is newly identified, connected, or reframed.
- **Method:** what is proposed and how it follows from the insight.
- **Evidence/implication:** what is theoretically or empirically validated and under which axes.

Each bullet should begin with an active verb and remain one coherent claim. A separate “takeaways” list is appropriate only for investigation-heavy papers with several reusable findings.

## Related work

Organize by the contrast that matters to the paper, not chronologically. For each cluster:

1. state what the family does;
2. give representative works;
3. identify the exact assumption or limitation relevant here;
4. distinguish the present paper in one restrained sentence.

Use phrases such as “while we focus on ...”, “these methods are complementary/orthogonal to ...”, and “however, they do not study ...” only with accurate scope. Do not turn related work into a sequence of dismissals.

## Preliminaries and problem formulation

Introduce only notation used by the argument. A typical flow is:

1. define the model, data, and time/client/task index;
2. state the standard objective or update;
3. define the target property after the operation;
4. explain the equation immediately in words;
5. isolate the quantity or condition the paper changes;
6. clarify special cases and scope.

Favor notation that makes the hinge visible—for example, a decoupled norm and relative weights, an anchor relation, an empty/refusal target, or a partially observed interaction history.

## Method

Open with a short overview tied to the diagnosis. Then order subsections by causal dependence, not code order.

For each component:

1. restate the specific failure it handles;
2. give the intuition in plain language;
3. define the operation or objective;
4. explain why it should change the desired property;
5. note cost, assumptions, or boundary conditions.

When there are two components, use parallel names and explain their interaction. Avoid presenting an implementation detail as a conceptual module.

## Experiments

Open with evaluation questions, even if not formatted as RQ1/RQ2. High-fit questions include:

- Does the observation hold across the settings implied by the thesis?
- Does the method improve the target property over strong and native baselines?
- Which component explains the gain?
- How does behavior change with the key control variable?
- What are the robustness, efficiency, OOD, or long-horizon boundaries?

### Result paragraph pattern

Use this sequence:

1. point to the figure or table;
2. state the dominant pattern;
3. give the most decision-relevant comparison or number;
4. explain what the pattern means for the paper's hypothesis;
5. acknowledge a tradeoff or exception when present.

Suitable language:

- “As shown in Table 2, ...”
- “We have three observations.”
- “The advantage becomes larger when ...”
- “This verifies/supports the hypothesis that ...”
- “A tradeoff exists between ... and ...”.
- “The method remains effective under ...”.

Avoid narrating every table cell. Results language should distinguish observation from explanation.

## Limitations and broader impacts

State the limitation directly, then bound it and identify the most natural extension. Do not claim that limitations “do not matter.” If there is a misuse risk, connect it to the mechanism and name a concrete mitigation or design advantage.

Broader impact should grow from the paper's technical object: memory control, personalization, user alignment, accessible model creation, or responsible deployment. Use “may,” “can,” or “we hope” for forward-looking claims.

## Conclusion

Use one compact paragraph unless the venue expects discussion. Reconstruct the causal chain:

`problem/reframing -> main finding -> method -> evidence -> bounded implication`

Characteristic opening: “In this paper, we [revisit/study/identify] ...”. Follow with “We find ...”, “Therefore, we propose ...”, and an evidence sentence. Do not repeat all benchmark details or introduce new technical claims.

## Style-preserving revision order

When revising an existing draft, work in this order:

1. thesis and conceptual hinge;
2. paragraph order and contrast structure;
3. claim-evidence alignment;
4. repeated terminology and transitions;
5. sentence clarity and grammar;
6. local word choice.

A thesaurus-only rewrite will not reproduce the style.
