# Abstract (draft v0 — 2026-05-17)

> 숫자(`[X]`)는 E1/E3/E7 결과 동결 후 채움. EMNLP long-paper ~180 words.

Large language models judge harmful *inaction* more leniently than equally
harmful *action* — an **omission bias** that prior work shows is amplified in
LLMs, but only around a single model anchor and a single
utilitarian↔deontologist axis. We ask whether this bias is systematic *across
models* and *which moral-philosophical conflicts* expose it. We build a
benchmark by re-casting MoralChoice high-ambiguity scenarios into paired
action↔omission mirror frames and typing each scenario by the disagreement of a
five-philosophy panel (utilitarian, deontologist, virtue, care,
contractualist). We show that this philosophy-disagreement filter exposes
framing-invariant inaction substantially better than random sampling
(`[+X]` OBR, consistent across models), validating the construction. Evaluating
five cross-vendor models, we find that omission bias is *non-uniform across
conflict types* and that each model carries a distinct **moral fingerprint** —
a characteristic alignment with particular philosophical camps. Finally, simple
prompt-level interventions — considering-the-opposite and a decision ledger —
reduce framing-invariant inaction without any change to the model or its
training. We make no causal claim; the panel is a construction-time
signal only. Benchmark and code are released.
