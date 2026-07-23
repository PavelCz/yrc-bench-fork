# GPU sharding for RL policy training on RNN

Procgen PPO training is env-step (CPU) bound; the policy network is a small
IMPALA-style CNN, so a whole GPU is heavily under-used. On the RNN (Berkeley
IST) cluster you can request a **fraction** of a GPU via SLURM's `shard` GRES
and pack several training runs onto one physical card.

## Cluster support

`GresTypes = gpu,shard` is enabled. Shards are only available on some nodes
(each shard is ~1 GB of VRAM by admin convention):

| Nodes | GPU | Shard config | shards/GPU | ~VRAM/shard |
|-------|-----|--------------|-----------:|------------:|
| `ddpg`, `dqn` | A6000 (48 GB) | `shard:A6000:384` | 48 | ~1 GB |
| `ppo`, `vae`  | A4000 (16 GB) | `shard:A4000:128` | 16 | ~1 GB |

The A100 nodes (`airl`, `sac`, `cirl`, `rlhf`) and the plain A6000 nodes
(`gail`, `gan`) have **no** shard GRES — there you can only take a whole GPU.

Requesting `--gres=shard:N` restricts the job to the shard-enabled nodes.
`shard` is cooperative (MPS-like time-slicing), **not** memory-isolated like
MIG: a run *can* exceed its shard count's worth of VRAM, so pick `N` to match
the run's real footprint — it is the honest "I will use ~N GB" signal that lets
SLURM co-schedule others without over-subscribing the card.

## Measured resource footprint

A short real coinrun PPO run (`param_name paper`, `num_threads 4`) measured on
an A4000 node. The same network / `num_envs` is used for weak, strong,
maze-robust, and ensemble policies, so these numbers cover every training job.

| Resource | Peak | Notes |
|----------|------|-------|
| **VRAM** | ~4.6 GB (4666 MiB) | steady-state; PyTorch's caching allocator retains the backward-pass peak |
| **Host RAM** | ~9.0 GB | cgroup peak; still drifting up slowly, and only 2.5M of 200M steps were measured, so a full run may sit higher |
| **CPU** | mean ~3.9 cores, peak ~8.3 | bimodal over the PPO cycle: rollout ~2.3 cores, optimization/eval bursts to 6–8 (torch intra-op threads). `num_threads 4` sets the rollout-worker count |

Measurement scripts are ad hoc (`nvidia-smi` per-process sampling for VRAM;
`/proc` tick deltas for CPU; cgroup-v1 `memory.max_usage_in_bytes` for RAM).
Re-run them if the model, `num_envs`, or `num_threads` changes.

## Recommended per-run request

```bash
--gpu-shards 6      # VRAM peak 4.6 GB → 5 shards + 1 headroom
--cpus-per-task 6   # mean ~4 cores, bursts to ~8; 6 covers rollout + most of the burst
--mem 24G           # RAM peak ~9 GB + headroom
```

- Use `--cpus-per-task 8` if you want zero throttling of the optimization
  bursts (at the cost of packing density).
- **Verify RAM on the first full-length run** — the measurement window was
  short (2.5M of 200M steps) and RAM was still creeping up.

### Packing density (A6000 shard node: 256 CPU, ~1 TB RAM, 384 shards)

| `--cpus-per-task` | shard-cap | cpu-cap | ram-cap @24G | jobs/node | bound by | vs whole-GPU (8/node) |
|---:|---:|---:|---:|---:|---|---:|
| 4 | 64 | 64 | 41 | 41 | ram | 5.1× |
| 6 | 64 | 42 | 41 | 41 | ram | 5.1× |
| 8 | 64 | 32 | 41 | 32 | cpu | 4.0× |

At `--mem 24G` the node is RAM-bound, so the shard count (6) is never the
bottleneck — there is no benefit to shrinking the VRAM request further.

## Orchestrator support

`scripts/train_policies.sh` and `scripts/train_ensemble_policies.sh` accept:

| Flag | Effect | Default |
|------|--------|---------|
| `--gpu-shards N` | `--gres=shard:N` instead of `--gres=gpu:1`. Accepts `N` or `TYPE:N` (e.g. `A6000:3`). Restricts jobs to shard nodes. | whole GPU |
| `--cpus-per-task N` | Sets SLURM CPUs per task explicitly. | cluster default |
| `--mem SIZE` | Overrides the per-job memory request. | 128G (policies) / 100G (ensemble) |

Example:

```bash
./scripts/train_policies.sh -e coinrun -x 4 --random-percent 0 \
    --gpu-shards 6 --cpus-per-task 6 --mem 24G
```

Omitting `--gpu-shards` preserves the previous whole-GPU behavior exactly.
