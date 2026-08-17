---
name: python-env-ferminet-piku
description: Use ferminet-piku conda env for numpy/analysis in this project
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ce11a168-063b-4f75-b5e7-29cd39b4d40d
---

Use `/home/u6em/parvfection.u6em/miniforge3/envs/ferminet-piku/bin/python` for data analysis
(numpy, etc.) and all repo tools in this project.

**Why:** The base miniforge env lacks numpy; ferminet-piku has the full stack.

**How to apply:** Always use the full path above when running analysis scripts via Bash.

⚠️ 2026-07-02 (post account migration): `conda` / `conda activate ferminet-piku` do NOT work in this
home dir's shell (`conda: command not found`; bare `python` also not on PATH). Conda init isn't
wired into the shell profile — **OPEN next-step to fix** (see [[current-status]]). Until then, ALWAYS
call the env python by the absolute path above. miniforge3 now lives at
`/home/u6em/parvfection.u6em/miniforge3` (conda binary `.../miniforge3/bin/conda`); the old
`/home/u6em/parvfect.u6em/...` path (old account) is dead.
