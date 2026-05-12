# SLURM Job Status — checked 2026-05-12 ~14:10 PDT (server time)

11 jobs total — 8 running, 3 pending (waiting on GPU priority).

## Pending
- `1133179` maze_svdd-latent_exp0-1-2-3
- `1133195` coinrun_svdd-image_exp0-1-2-3
- `1133196` coinrun_svdd-latent_exp0-1-2-3

## Running — all making meaningful progress

### Why several logs *looked* stuck at first
Most jobs are running `eval_afhp.py` with adaptive threshold coverage sampling. Each sampling iteration runs a **5000-episode evaluation** on the test split. Video collection (`YRC/core/evaluator.py:1029`) logs the first 16 episodes as `Episode N - Filter 'all': N/16`, then the rest run silently until the per-eval summary block prints (`Steps: …, Threshold: …, Level AFHP: …`).

Cross-checked each exp's eval-cycle history (grep `Evaluation on test|Steps:`). Cycle lengths are internally consistent (2–8h, generally growing), so the current silences fit the cadence rather than indicating a hang.

### Per-job state

| JobID | Name | Eval cycle | Status |
|---|---|---|---|
| 1132467 | maze_proxy_fail_svdd-latent | active threshold sampling | all 4 exps mid-eval, 2–110min stale, normal |
| 1132777 | maze_max-prob_robust400 | ~3–6h | exp3 actively logging (5m), others 73–149min into cycle |
| 1132778 | maze_max-logit_robust400 | ~3–5h | all 4 in cycle, 62–102min stale |
| 1132779 | maze_ensemble-single_robust400 | **growing 2h → 8h** | exp0 silent 7h vs last cycle 6h52m (close to wire); exp1 5.9h vs 7h20m; exp2 fresh (42min) |
| 1132782 | coinrun_proxy_fail_max-prob | ~5h | exp0 fresh (6min), others mid-cycle |
| 1132783 | coinrun_proxy_fail_max-logit | ~5h | exp3 actively logging, exp1 mid-cycle |
| 1132784 | coinrun_proxy_fail_ensemble-single | consistent ~5h | all 4 mid-cycle, 162–241min stale, fits cadence |
| 1133178 | maze_svdd-image (smoke-neurips05) | 16-episode evals | newest (~45m old), all 4 logging actively |

## Things to watch
- **`1132779` exp0 & exp1**: cycles are growing each iteration (2h → 3h → 4h → 5h → 7h → …). exp0 silent 7h11m vs last cycle 6h52m — plausibly still running, but if no `Steps:` summary appears within another ~1h, treat as hung.
- The parent `*_exp0-1-2-3_<jid>.err` files show `srun: Connection reset / Insane message length / protocol_version 515 not supported` warnings. These are SLURM controller↔node comms hiccups; not necessarily fatal to the python processes but noted.
- Time limits are 3 days; long jobs are at 1d13h elapsed, so ~1.5d headroom.

## How I checked
1. `squeue -u $USER` — listed running/pending jobs.
2. `scontrol show job <jid>` — got `StdOut`/`StdErr` paths.
3. The parent `*_exp0-1-2-3_*.{out,err}` files contain almost nothing useful (only `Using conda env: ood-stable` + srun comm errors). The real logs are per-experiment files: `<name>_exp{0,1,2,3}_<jid>.err` in the same directory.
4. `stat -c %Y` on each per-exp `.err` file → staleness (minutes since last write).
5. `grep 'Evaluation on test|Steps:' <log>` → eval cycle history for each exp, to compare current silence against historical cadence.
6. `sstat --format=JobID,MaxRSS,AveCPU,AveCPUFreq -j <jid>.{0..3}` — initially suggested hangs (AveCPUFreq ≈ 0–4K vs 50–160K on active job 1133178), but this is *averaged over total elapsed time* so it's a poor instantaneous-activity signal for long-running jobs that spend most of their time GPU-bound during eval.
