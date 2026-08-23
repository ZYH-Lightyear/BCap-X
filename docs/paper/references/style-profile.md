# Style Profile

## Corpus basis

This profile is distilled from six English AI/ML papers authored or co-authored by Zexi Li:

- *Revisiting Weighted Aggregation in Federated Learning with Neural Networks* (ICML 2023)
- *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models* (NeurIPS 2024)
- *FedGuCci: Making Local Models More Connected in Landscape for Federated Learning* (KDD 2025)
- *Editing as Unlearning: Are Knowledge Editing Methods Strong Baselines for Large Language Model Unlearning?* (AAAI 2026)
- *Tina: A Diffusion Neural Network for Generating Personalized AI Models from Text Prompts* (Patterns 2026)
- *IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning* (ICML 2026)

The papers span different collaborators and venues. Treat a feature as core only when it recurs across topics or when it expresses the same reasoning move in a venue-specific form.

## Core authorial signature

### 1. Observation-led, convention-aware framing

The paper rarely begins from “we built a better method.” It begins from a default assumption, overlooked relationship, or practical failure:

- normalized aggregation weights may be unnecessarily fixed;
- pairwise connectivity may extend to a group through an anchor;
- editing and unlearning may be two views of the same memory operation;
- long-term and working memory create an impossible triangle;
- content generation may extend to model generation;
- scaling interaction may be more efficient than scaling search.

The characteristic move is to make a familiar object newly questionable. Use verbs such as **revisit**, **rethink**, **study from the ... perspective**, **bridge the gap**, **identify**, and **reveal** only when the technical argument earns them.

### 2. A visible conceptual hinge

Each strong paper has one sentence that changes the reader's model of the problem. It is usually a contrast or equivalence:

- a standard setting is a special case of a broader setting;
- two distinct tasks share the same underlying operation;
- an optimization phenomenon is analogous to a known regularizer;
- two imperfect memory types imply a third design;
- one scaling axis is less efficient than another.

State this hinge plainly. Do not bury it after method details.

### 3. Diagnosis before construction

The method follows a diagnosis with named structure. The recurring order is:

1. expose a limitation or phenomenon;
2. decompose its cause into two or three factors;
3. define a new concept or lens;
4. derive a compact method with matching modules;
5. validate both the insight and the method.

This produces papers that read as “understanding plus solution,” not merely “architecture plus results.”

### 4. Explicit, memorable contrasts

Use paired or triadic concepts when they are technically real:

- optimization vs. regularization;
- absolute norm vs. relative weights;
- pretrained vs. edited knowledge;
- long-term vs. working memory;
- reliability, generalization, and locality;
- autonomy vs. interaction;
- scaling search vs. scaling interaction;
- in-distribution vs. out-of-distribution.

Reuse the same labels consistently throughout the paper. The author favors conceptual legibility over lexical variety.

### 5. Broad purpose anchored by concrete mechanism

The writing often opens or closes toward a larger purpose—generalization, responsible deployment, personalization, user alignment, human-AI interaction, or scientific access. The broad claim is connected to a concrete mechanism and is softened when speculative.

## Voice and sentence behavior

### Default voice

- Prefer first-person plural and present tense: “we study,” “we find,” “we propose,” “we evaluate.”
- Use “this paper” to mark purpose, boundary, or advocacy: “In this paper, we ...”.
- Use passive voice for standard setup details or when the object matters more than the actor.
- Avoid a detached survey voice in the paper's own argument.

### Sentence rhythm

Corpus sentences are generally medium length, commonly around 16–22 words outside formulas and tables. Alternate a compact claim with one or two explanatory sentences. Long sentences are acceptable when they encode a controlled contrast or numbered decomposition; do not accumulate unrelated clauses.

Use periods and commas as the default punctuation. An em dash should be rare and should express an interruption or afterthought that a new sentence cannot express as clearly. Do not use em dashes merely to add emphasis. Use semicolons only for two tightly related independent clauses or an unusually complex list. In most cases, split the sentence or use a comma with an explicit conjunction. Avoid placing more than one em dash or semicolon in a paragraph.

Typical paragraph rhythm:

1. Claim or contrast.
2. Mechanistic explanation.
3. Concrete implication, example, or citation.
4. Transition to the next question.

### Logical connectors

High-fit connectors include the following, but use them only when they clarify a real logical relation:

- contrast: **However**, **In contrast**, **While**, **Unlike**, **Differently**;
- consequence: **Therefore**, **Thus**, **As a result**, **This suggests/inspires us to**;
- focus: **Specifically**, **In particular**, **More precisely**, **Notably**;
- evidence: **As shown in Figure/Table ...**, **Our experiments show**, **It can be observed that**;
- derivation: **Based on the above finding(s)**, **To address this**, **Following this setup**;
- intuition: **Interestingly**, **Intuitively**—use sparingly and follow immediately with substance.

Do not begin several consecutive paragraphs with the same connector.

Do not make transitions conspicuous. If the order of two sentences already makes the relation clear, no transition phrase is needed. Avoid automatic openings such as **Moreover**, **Furthermore**, **Importantly**, and **It is worth noting that** unless they add a distinction that would otherwise be missed.

## Lexical palette

### Characteristic verbs by function

- framing: **study**, **investigate**, **revisit**, **rethink**, **consider**, **formulate**;
- discovery: **find**, **identify**, **observe**, **reveal**, **verify**;
- construction: **propose**, **introduce**, **design**, **use**, **combine**, **incorporate**;
- evidence: **show**, **demonstrate**, **validate**, **outperform**, **improve**;
- positioning: **bridge**, **extend**, **adapt**, **align**, **generalize**, **preserve**.

### Characteristic evaluative language

Use when supported:

- **fundamental perspective/question**
- **inherent connection/commonality**
- **simple yet effective**
- **practical method/recipe/insight**
- **promising results/paradigm/direction**
- **strong baseline**
- **broad(er) application/impact**
- **holistic understanding/control**
- **extensive experiments**
- **across various settings/tasks/models**

Avoid stacking more than one evaluative adjective around the same noun.

Treat this list as descriptive rather than mandatory. Repeatedly using these expressions makes the prose sound formulaic. Prefer a concrete description of the result over an evaluative label.

### Plain and natural wording

Choose the shortest familiar word that preserves the technical meaning. Prefer **use** to **utilize** or **leverage**, **help** to **facilitate**, **show** to **underscore** or **highlight**, and **study** to **delve into**. Prefer a concrete noun to vague abstractions such as **landscape**, **realm**, **ecosystem**, or **paradigm** unless that noun has an established technical meaning in context.

Avoid stock phrases often produced by generic writing assistants, including:

- **delve into**, **shed light on**, **pave the way for**, **at the forefront of**, and **a testament to**;
- **intricate**, **multifaceted**, **nuanced**, **pivotal**, **transformative**, **seamless**, and **remarkable** when a specific description would be clearer;
- **it is important/crucial to note that**, **it is worth mentioning that**, and **in today's rapidly evolving ...**;
- routine claims that a method **underscores**, **showcases**, or **highlights** something when **shows** states the result directly.

These words are not banned when they carry a precise technical meaning or appear in an accurate quotation, method name, or established term. The rule is to avoid using them as automatic decoration.

Do not make every sentence maximally polished or rhetorically symmetrical. Vary sentence length according to the content. Use a short sentence when the main finding deserves emphasis. Do not repeatedly arrange claims into “not only ... but also ...,” three parallel adjectives, or three abstract nouns. A list should reflect the actual structure of the work rather than a preference for triads.

### Stable phrase frames

Use these frames only when they fit the argument, and rewrite them when nearby sections already use the same pattern:

- “In this paper, we study [problem] from the perspective of [lens].”
- “Though [A] and [B] seem to be distinct, they share [mechanism].”
- “This observation inspires us to ask whether [question].”
- “However, there is a crucial gap between [established setting] and [target setting].”
- “Based on the above finding, we propose [method], which [core mechanism].”
- “To better understand [phenomenon], we decouple it into (i) ... and (ii) ...”.
- “Extensive experiments across [axes] validate [specific claim].”
- “The results suggest that [bounded interpretation].”

## Claim calibration

Use four levels deliberately:

1. **Definition or measured fact:** “X is ...”; “Table 2 shows ...”.
2. **Supported generalization:** “These results demonstrate ... across the evaluated settings.”
3. **Interpretation:** “This suggests that ...”; “may explain ...”.
4. **Vision or advocacy:** “We hope ...”; “may provide a foundation for ...”.

Do not move from a benchmark result directly to a universal claim. Scope statements such as “in the evaluated lifelong-editing setting,” “for pretrained knowledge,” or “under client heterogeneity” are part of the style.

## What not to imitate

PDF extraction contains broken words, missing spaces, and notation artifacts. Some source sentences also reflect normal co-author or draft-stage variation. Do not reproduce malformed typography, grammatical slips, or venue boilerplate. The target is a polished continuation of the author's strongest recurring habits.

Avoid:

- a literature-first introduction with no motivating tension;
- vague novelty claims before the observation is explained;
- a method assembled from many equal-weight modules;
- “significant” without a statistical or clearly practical basis;
- repeated hype words such as “novel,” “groundbreaking,” or “revolutionary”;
- replacing every repeated core term with synonyms;
- conclusions that introduce new results.
