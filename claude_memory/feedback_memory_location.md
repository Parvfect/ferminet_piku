---
name: feedback-memory-location
description: "Write auto-memory files directly to the real harness-loaded memory dir, not the repo's claude_memory/ folder"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 248c52b7-c847-454b-9e95-3038f4e68812
---

Write new/updated memory files directly to the real auto-loaded memory directory (`/home/u6em/parvfection.u6em/.claude/projects/-home-u6em-parvfection-u6em-ferminet-piku/memory/`), not to the repo's `claude_memory/` folder.

**Why:** `claude_memory/` in the repo is a git-tracked backup mirror a past session created, not the harness's auto-loaded location. Writing only there let the two drift — the repo copy had a full 2026-07-01 update (all jobs dead) while the real auto-loaded memory was still stuck on 2026-06-30 ("6 jobs RUNNING"), so a fresh conversation would have started from stale/wrong status. Reconciled once on 2026-07-01 (see [[current-status]]).

**How to apply:** Default to the real memory dir for all memory writes. If the user also wants a repo-tracked backup copy, treat that as a separate explicit copy step, not the primary write target.
