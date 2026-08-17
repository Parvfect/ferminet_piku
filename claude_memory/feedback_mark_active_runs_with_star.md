---
name: feedback-mark-active-runs-with-star
description: "In energy/run comparison tables, mark active (still-training) runs with a ★; user likes this convention"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 00e386a0-feae-488d-9a37-f2377cc47618
---

In run/energy comparison tables (and similar status tables), mark ACTIVE
(still-training) runs with a ★ next to the run name; leave frozen/closed/
not-started runs unmarked.

**Why:** The user explicitly said they like the ★-on-active-runs convention and
asked me to keep doing it. It makes "which numbers are still moving / are upper
bounds" instantly scannable.

**How to apply:** Whenever producing the diamond/silicon energy comparisons
([[project-diamond-pp-energy-comparison]], [[project-silicon-pp-energy-comparison]])
or any run-status table, append ★ to the Run column for rows that are RUNNING/
descending, and note in the table caption that ★ = active. Reuse frozen rows'
final values; only re-measure ★ rows with [[tool-energy-convergence]].
