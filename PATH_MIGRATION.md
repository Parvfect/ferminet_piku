# Path Migration: parv → parvfection

**Why:** `/projects/u6em/parv/` is owned by `parvfect.u6em` (old account). Current account
`parvfection.u6em` can read it (group `brics.u6em`) but cannot write. All 36 config files
have hardcoded `save_path` / `restore_path` pointing there.

**Goal:** New writable base `/projects/u6em/parvfection/` owned by current account. Active
training runs resume from their latest checkpoint. Completed-run checkpoints stay readable in
parv until needed for inference (copy lazily then).

---

## Step 1 — Create new base directory

```bash
mkdir /projects/u6em/parvfection
```

Verify it's yours:
```bash
ls -la /projects/u6em/ | grep parvfection
# should show: drwxr-sr-x  parvfection.u6em  brics.u6em
```

---

## Step 2 — Copy latest checkpoints for the 6 active training runs

Only the most recent `.npz` is needed to resume. Total ~168 MB.

```bash
PARV=/projects/u6em/parv
NEW=/projects/u6em/parvfection

for dir in \
  silicon_unpaired/classical/t_relaxed \
  diamond/unpaired/t_relaxed/pp \
  "diamond/unpaired/classical/t_relaxed/pp" \
  diamond/unpaired/bc_relaxed/pp_bc_seeded \
  silicon_unpaired/bc_seeded \
  silicon_unpaired/t_seeded; do
    mkdir -p "$NEW/$dir"
    latest=$(ls "$PARV/$dir"/qmcjax_ckpt_*.npz 2>/dev/null | sort | tail -1)
    if [ -n "$latest" ]; then
      cp "$latest" "$NEW/$dir/"
      echo "OK  $dir  →  $(basename $latest)"
    else
      echo "MISSING  $dir"
    fi
done
```

Expected output (latest step per run at time of migration):

| Run dir | Step | Size |
|---|---|---|
| `silicon_unpaired/classical/t_relaxed` | 179742 | 16 MB |
| `diamond/unpaired/t_relaxed/pp` | 316000 | 34 MB |
| `diamond/unpaired/classical/t_relaxed/pp` | 161268 | 16 MB |
| `diamond/unpaired/bc_relaxed/pp_bc_seeded` | 232194 | 34 MB |
| `silicon_unpaired/bc_seeded` | 60000 | 34 MB |
| `silicon_unpaired/t_seeded` | 34000 | 34 MB |

If a run has continued since this plan was written, `ls … | sort | tail -1` will pick up
the newer checkpoint automatically — no need to update the script.

---

## Step 3 — Global path replacement in all configs

```bash
find /home/u6em/parvfection.u6em/ferminet_piku/ferminet/configs -name "*.py" \
  | xargs sed -i 's|/projects/u6em/parv/|/projects/u6em/parvfection/|g'
```

This touches all 36 config files (training + inference). Inference `restore_path`s will now
point into parvfection — fine for the active runs (checkpoint just copied) and fine for
future inference on newly completed runs. For older completed runs (jobs #1–10), see the
note below.

Spot-check a few files:
```bash
grep "parvfection\|parv" \
  ferminet/configs/diamond/bc_relaxed/bc_seeded.py \
  ferminet/configs/silicon/t_relaxed_classical.py \
  ferminet/configs/diamond/t_relaxed/inference_pp.py
# No line should still contain /projects/u6em/parv/
```

---

## Step 4 — Commit

```bash
cd /home/u6em/parvfection.u6em/ferminet_piku
git add ferminet/configs/
git commit -m "Migrate all save/restore paths from parv to parvfection account dir"
```

---

## Lazy copy for completed-run inference (do when needed, not now)

Inference configs for jobs #1–10 now point restore_path into parvfection, but the
checkpoints are still only in parv (still readable). Before re-running any of these
inferences, copy that run's checkpoint dir:

```bash
PARV=/projects/u6em/parv
NEW=/projects/u6em/parvfection

# Example: job #4 (classical BC, converged @338k)
mkdir -p "$NEW/diamond/unpaired/classical/bc"
cp "$PARV/diamond/unpaired/classical/bc"/qmcjax_ckpt_*.npz "$NEW/diamond/unpaired/classical/bc/"
```

Completed run → new dir mapping (all under base):

| Job | parv subdir | needed for |
|---|---|---|
| #4 | `diamond/unpaired/classical/bc` | `inference_pp_T.py` |
| #6 | `diamond/unpaired/unrelaxed/pp` | `test_run_inference.py` |
| #7 | `diamond/unpaired/classical/T_site/pp` | `t_site_inference_pp.py` |
| #8 | `diamond/unpaired/bc_relaxed/pp_relax_2` | `inference_pp_relax_2.py` |
| #9 | `silicon_unpaired/bc_relaxed` | `inference_bc_relaxed.py` |
| #2 | `silicon_unpaired/classic` | `t_site_inference.py` |
| #10 | `silicon_unpaired/t_relaxed` | `inference_t_relaxed.py` |
