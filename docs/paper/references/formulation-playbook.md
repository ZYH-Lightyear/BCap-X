# Formulation Playbook

Use this reference when choosing the paper idea, thesis, title, conceptual framing, or outline.

## Select the central formulation

Choose one primary archetype. Secondary moves may support it, but they should not compete with the main story.

### A. Revisit a convention

Best when a widely fixed design choice may not hold for modern models or a changed setting.

Form:

`accepted convention -> counter-observation -> decompose the hidden degree of freedom -> explain dynamics -> derive adaptive method`

The insight should matter without the final method. Prefer a convention readers recognize immediately.

### B. Bridge adjacent fields or tasks

Best when two communities optimize related objects with different language or baselines.

Form:

`task A and task B appear distinct -> show a special-case/equivalence/analogy -> identify what transfers and what does not -> adapt the transferred method -> advocate a broader view`

The bridge must yield a testable baseline, theorem, metric, or design—not only a metaphor.

### C. Resolve a structural tension

Best when existing method families occupy different corners of a tradeoff.

Form:

`required properties -> empirical or conceptual impossibility -> map method families to the failure -> infer missing mechanism -> design a bridge mechanism`

Name the tension compactly, such as a dilemma, triangle, gap, or conflict. Verify that the properties are genuinely hard to attain together.

### D. Extend a generative paradigm

Best when a familiar input-output mapping can be lifted to a new functional object.

Form:

`successful paradigm in existing modalities -> ask whether the semantic mapping extends -> define the new generation target -> show a feasible conditional generator -> validate generalization and practical use`

Separate the broad paradigm from the initial scoped setting. State what the paper proves now and what remains future work.

### E. Replace an inefficient scaling axis

Best when more computation addresses the wrong uncertainty.

Form:

`long-horizon/high-cost task -> hidden information gap -> current axis scales coverage or autonomy -> alternative axis acquires missing information -> train and evaluate the new behavior`

Make the cost asymmetry concrete and show why the alternative acts before the expensive failure.

## Idea quality filter

Advance an idea only if most answers are strong:

1. **One-sentence surprise:** Can the non-obvious observation be stated without method names?
2. **Structural importance:** Does it alter how researchers view the problem, baseline, or design space?
3. **Operational test:** Can the hinge be measured, falsified, or visualized?
4. **Method inevitability:** Does the proposed mechanism follow from the diagnosis?
5. **Compactness:** Can the core method be explained with one main mechanism and at most two or three supporting modules?
6. **Validation breadth:** Can experiments vary the axes implied by the claim, not only datasets?
7. **Boundary clarity:** Can the paper say when the idea fails or remains incomplete?
8. **Broader relevance:** Does the insight transfer to a community, setting, or interaction model beyond a single benchmark?

Prefer ideas with a strong conceptual hinge and a simple implementation over ideas whose novelty is only a stack of components or a small average gain.

## Build the thesis spine

Write these sentences before outlining:

1. **Setting:** `[Area] is important because [concrete capability or need].`
2. **Bottleneck:** `However, [failure] limits [goal], especially when [condition].`
3. **Diagnosis:** `We find that this failure stems from [hidden assumption/tension].`
4. **Hinge:** `[A] can be viewed as [relationship to B], which suggests [new possibility].`
5. **Question:** `Can/how/what [focused research question]?`
6. **Answer:** `We propose [name], which [single mechanism] by [key operation].`
7. **Evidence:** `Across [meaningful axes], [verified result pattern].`
8. **Implication:** `These findings suggest [bounded change in understanding or practice].`

If sentences 3–6 do not form a tight causal chain, revise the formulation before adding details.

## Choose and reject contributions

The usual contribution hierarchy is:

1. a phenomenon, perspective, formulation, or diagnosis;
2. a method or framework derived from it;
3. empirical/theoretical evidence and actionable findings.

Merge implementation details under the method contribution. Promote a module to a separate contribution only when it changes the general problem-solving principle.

Reject or demote:

- routine engineering;
- extra datasets with no new axis of evidence;
- an auxiliary loss that does not express the paper's diagnosis;
- a broad application claim demonstrated by one anecdote;
- a priority claim that has not been checked;
- a second “main idea” that weakens the first.

## Name concepts and methods

The corpus favors names that are memorable and semantically connected to the paper's hinge: FEDLAW, FedGuCci, WISE, Tina, IntentRL.

A good name should:

- expose the central operation, object, or metaphor;
- be easy to pronounce and reuse as a noun;
- support consistent module names;
- avoid an acronym expanded into unnatural English;
- avoid claiming a generality the method does not have.

Title patterns that fit:

- `Revisiting [Fundamental Operation] in [Setting]`
- `[Method]: Rethinking [Concept] for [Task]`
- `[Method]: [Action-Oriented Description] for [Goal]`
- `[Task A] as [Task B]: Are [Methods] Strong Baselines for [Setting]?`
- `[Method]: Training [Agent/System] for [Challenge] via [Mechanism]`

Use a question title only when the paper genuinely performs an investigation and can give a nuanced answer.

## Design evidence around the claim

Experiments should mirror the formulation:

- a convention paper varies the supposedly fixed quantity and studies its dynamics;
- a bridge paper compares transferred methods with native baselines and tests adaptation boundaries;
- a tension paper reports every property in the tension, not only an average score;
- a paradigm paper tests ID, OOD, unseen targets, and at least one realistic scenario;
- a new scaling-axis paper measures both immediate behavior and downstream task value under cost constraints.

Include boundary or failure experiments. In this style, knowing where a method stops working strengthens the conceptual claim.
