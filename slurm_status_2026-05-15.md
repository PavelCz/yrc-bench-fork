# SLURM Job Status — checked 2026-05-15 ~13:35 PDT (server time)

20 jobs total — 18 running, 2 pending (waiting on GPU priority).

## Pending
- `1133603` maze_proxy_fail_svdd-latent_robust400_exp0-1-2-3 — (tmlr-proxy-fail2_robust400)
- `1136590` maze_proxy_fail_ensemble-single_exp3 — (tmlr-proxy-fail3)

## Running — all making meaningful progress

### How I checked
Same procedure as the prior status:
1. `squeue -u $USER` + `scontrol show job <jid>` → log paths
2. `stat -c %Y` on each per-exp `.err` → minutes since last write
3. `grep "Evaluation on test for 5000 episodes"` → completed sampler iterations
4. `grep "Phase|coverage_percentage"` + `tail` → current phase / mid-cycle position
5. The 16-episode video preamble logs (`Episode N - Filter 'all': N/16`)
   bracket each eval cycle; long silences between cycles fit the 5000-episode
   eval cadence, not hangs.

### Per-job state

| JobID | Name | Prefix | Elapsed | Iters (avg) | Status |
|---|---|---|---|---|---|
| 1133207 | maze_svdd-image_exp0-1-2-3 | tmlr3-svdd | **2d 21h 32m** | 13–14 | **time-limit risk:** ~6h headroom, ~6h/iter cadence → likely partial coverage |
| 1133231 | maze_svdd-latent_exp0-1-2-3 | tmlr3-svdd | **2d 20h 31m** | 15–16 | **time-limit risk:** ~7h 30m headroom, ~4–5h/iter cadence → may squeeze 1 more iter |
| 1133557 | maze_svdd-latent_robust400_exp0-1-2-3 | tmlr-robust-maze-2 | 2d 00h 21m | mid-eval | all 4 exps 48–135min stale, fits cadence |
| 1133558 | maze_proxy_fail_svdd-image_exp0-1-2-3 | tmlr-proxy-fail2 | 2d 00h 18m | 21–23 | exp3 actively logging (3min), others 67–105min into cycle |
| 1133559 | maze_proxy_fail_svdd-latent_exp0-1-2-3 | tmlr-proxy-fail2 | 1d 18h 41m | mid-eval | exp0 fresh (8min), 41–104min on others |
| 1133560 | coinrun_proxy_fail_svdd-image_exp0-1-2-3 | tmlr-proxy-fail2 | 1d 12h 03m | mid-eval | 26–110min stale, no outliers |
| 1133561 | coinrun_proxy_fail_svdd-latent_exp0-1-2-3 | tmlr-proxy-fail2 | 1d 12h 03m | 13–16 | exp1 fresh (16min), others 60–125min mid-cycle |
| 1133595 | coinrun_oracle-lb-random_exp0-1-2-3 | tmlr-oracle-lb | 1d 12h 00m | mid-eval | 27–163min stale, fits cadence |
| 1133596 | maze_oracle-lb-random_exp0-1-2-3 | tmlr-oracle-lb | 1d 07h 24m | 9+ | exp0 fresh (74min), exp1–3 between 153–173min |
| 1133597 | maze_oracle-lb-random_robust400_exp0-1-2-3 | tmlr-oracle-lb-robust400 | 16h 02m | mid-eval | 69–83min stale across all 4 |
| 1133599 | coinrun_proxy_fail_oracle-lb-random_exp0-1-2-3 | tmlr-oracle-lb-proxy_fail | 16h 02m | mid-eval | 44–83min stale, normal |
| 1133601 | maze_proxy_fail_oracle-lb-random_robust400_exp0-1-2-3 | tmlr-oracle-lb-proxy_fail_robust400 | 12h 35m | mid-eval | 12–46min stale, fresh on most exps |
| 1133602 | maze_proxy_fail_svdd-image_robust400_exp0-1-2-3 | tmlr-proxy-fail2_robust400 | 4h 12m | early | 68–84min stale, ramp-up |
| 1136586 | coinrun_proxy_fail_ensemble-single_exp3 | tmlr-proxy-fail3 | 6h 08m | mid-eval | unpacked single-exp, 20min stale |
| 1136587 | maze_proxy_fail_ensemble-single_exp0 | tmlr-proxy-fail3 | 5h 48m | mid-eval | unpacked single-exp, 7min stale (most active) |
| 1136588 | maze_proxy_fail_ensemble-single_exp1 | tmlr-proxy-fail3 | 4h 17m | mid-eval | unpacked single-exp, 12min stale |
| 1136589 | maze_proxy_fail_ensemble-single_exp2 | tmlr-proxy-fail3 | 3h 14m | early | unpacked single-exp, 17min stale |
| 1137495 | heist_sanity_eval | (sanity) | 0h 34s | n/a | 30min wall, just started |

## Things to watch

- **`1133207` & `1133231` time wall (highest priority).** Both at 2d 20h+
  of 3d limit. With ~5–6h per sampling iteration and 6–7.5h of headroom,
  expect at most 1 more iteration per exp before SLURM kills them. The
  AFHP sampler writes results incrementally, so partial coverage is
  recoverable. After they die, decide whether to:
  - Re-launch the missing iterations with `--coverage-fraction 0.2` to
    coarsen the bins (faster to fill remaining gaps), or
  - Resume with the same fraction and longer time limit.
- **`tmlr-proxy-fail3` ensemble-single jobs (1136586–1136589, 1136590
  pending).** These are launched as unpacked single-exp jobs (one job
  per exp). Each is GPU-bound on its own allocation — fine for cluster
  utilization, but it means `--runs-per-gpu` packing was not used here.
  Confirm intentional vs accidental.
- **Pending priority queue is short (2 jobs).** No new training/eval
  campaigns competing for slots right now.
- **No `srun` connection-reset noise observed this round.** The
  controller↔node comms hiccups from 2026-05-12 don't appear in
  today's parent `.err` files for the new jobs.

## Compared to 2026-05-12
- Last status had 11 jobs (8 running, 3 pending). Now: 20 jobs
  (18 running, 2 pending) — substantially more in flight.
- `1132779 maze_ensemble-single_robust400` (the growing-cycle concern
  from last time) is no longer in the queue, presumably completed or
  cancelled.
- `1133178 maze_svdd-image (smoke-neurips05)` was the smoke test;
  it completed cleanly along with siblings 1133176/177/179 — see the
  prior 16-of-16 verdict.
- The current oldest jobs (1133207, 1133231) are at the same end-of-life
  stage `1132779` was approaching in the last report.
