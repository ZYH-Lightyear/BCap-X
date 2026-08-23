# Quality Control

Apply this after drafting. Revise the draft rather than merely reporting checklist failures.

## A. Formulation audit

- Can the paper's non-obvious observation be stated in one sentence without the method name?
- Is there one dominant hinge—convention, bridge, tension, paradigm extension, or scaling-axis change?
- Does the method follow directly from the diagnosed cause?
- Do method modules correspond to previously stated problems?
- Are the top three contributions insight, solution, and evidence rather than three implementation details?
- Is every broad implication explicitly bounded by the evidence?

If the paper could swap in an unrelated method after the introduction without changing the story, the formulation is too generic.

## B. Author-style audit

- The opening reaches the concrete bottleneck quickly.
- The central contrast uses stable repeated labels.
- At least one paragraph makes the conceptual hinge explicit.
- Paragraphs usually open with a claim or logical transition.
- “We” carries the paper's actions; passive voice does not hide the contribution.
- Core verbs are direct: study, find, identify, propose, show, validate.
- Numbered decompositions are used only for real parallel structure.
- Interpretive claims use calibrated language such as may, suggests, or can.
- The draft sounds clear and idea-driven, not ornate or marketing-heavy.

## C. Section-specific audit

### Abstract

- Contains setting, bottleneck, hinge, method, evidence, and implication.
- Does not spend more than roughly one third on background.
- Names the method only after the problem and observation are understandable.

### Introduction

- Explains why prior approaches are insufficient without straw-manning them.
- States one focused research question when a question improves the narrative.
- Previews findings before or with contributions.
- Contributions are not a duplicate table of contents.

### Method/formulation

- Every symbol is defined before use.
- Every important equation is followed by a plain-language interpretation.
- The distinction between intuition, assumption, and proven/measured fact is clear.

### Experiments

- Baselines are strong, current for the paper's time, and relevant to the claim.
- Axes of evaluation follow from the thesis.
- Result paragraphs state pattern, evidence, interpretation, and boundary.
- No number or statistical claim is invented or rounded misleadingly.

### Conclusion

- Restates the causal story in a compact form.
- Introduces no new result.
- Ends with a specific bounded implication rather than generic future-work language.

## D. Caricature detector

Revise if any of these occur:

- “In this paper” appears in several neighboring paragraphs.
- “We propose” is used for minor implementation steps.
- more than one rhetorical question competes for attention;
- “fundamental,” “promising,” “holistic,” and “simple yet effective” are stacked without evidence;
- every paragraph follows the identical four-sentence template;
- the draft copies signature phrases but lacks an observation-led argument;
- grammar is intentionally degraded to mimic source-paper slips.
- em dashes or semicolons are used as a recurring sentence-building device;
- transitions such as “Moreover,” “Furthermore,” or “Notably” appear when sentence order already makes the relation clear;
- simple meanings are inflated with words such as “delve,” “leverage,” “pivotal,” “intricate,” “multifaceted,” “transformative,” or “underscore”;
- several paragraphs use the same balanced contrast, three-item list, or polished closing sentence.

## E. Natural-language pass

1. Replace decorative em dashes with a period, comma, or parentheses. Retain an em dash only when it marks a real interruption or afterthought.
2. Split semicolon-heavy sentences. Retain a semicolon only when the joined clauses are both independent and easier to understand together.
3. Replace inflated verbs and adjectives with common precise words. Do not simplify established technical terms.
4. Delete throat-clearing phrases such as “It is important to note that.” State the point directly.
5. Remove transitions that merely announce continuation. Let sentence order carry obvious relations.
6. Break repeated rhetorical symmetry. Keep parallel structure only when the underlying concepts are genuinely parallel.
7. Read the paragraph aloud. Revise any sentence that sounds like a polished template rather than a researcher explaining a specific idea.

## F. Final compression pass

1. Remove background that does not set up the hinge.
2. Replace vague pronouns with the core technical noun when ambiguity exists.
3. Merge repeated claims; keep repeated terminology.
4. Cut unsupported adjectives.
5. Turn long module lists into the one mechanism they collectively implement.
6. Check that the strongest finding appears in the abstract, introduction, results, and conclusion with consistent scope.

The final draft should feel confident, compact, and conceptually transparent. Its resemblance should come primarily from the way it selects and orders ideas.
