# Container Workflow

This repo uses a two-layer container setup for Procgen work:

1. Build a Docker base image on a machine with Docker. This layer contains CUDA, Python 3.8, PyTorch, CMake, Qt5, and system libraries.
2. Build the project image from that base. On cluster3 this is converted to Apptainer with `spython`, then built as a `.sif`.

The split keeps `apt-get` and heavyweight CUDA/PyTorch downloads out of cluster3 Apptainer builds.

## Local Docker Base Image

Build and push the base image from the repo root:

```bash
docker build -f docker/Dockerfile.base \
  -t docker.io/anonymous/yrc-bench-procgen-base:py38-cu121 .

docker push docker.io/anonymous/yrc-bench-procgen-base:py38-cu121
```

If you use another registry or tag, update the `FROM` line in `docker/Dockerfile` before building the app layer on cluster3.

## Local Docker App Image

For machines with Docker, build the runnable image directly:

```bash
docker build -f docker/Dockerfile -t yrc-bench-procgen:latest .
```

Run a command with the repo bind-mounted:

```bash
scripts/container/docker_run.sh python -c "import torch, procgen, acs, pyod; print(torch.__version__)"
```

The image contains dependencies plus installed `acs`, `procgen`, and `pyod`. The repo source is still mounted at `/workspace`, so normal code edits do not require rebuilding unless you change `requirements.txt`, `lib/acs`, `lib/procgen`, or `lib/pyod`.

## cluster3 Apptainer Build

Prerequisites on cluster3:

```bash
git submodule update --init --recursive
pipx install spython
```

Submit the build from the repo root:

```bash
sbatch scripts/cluster3/build_apptainer.sbatch
```

This writes `apptainer.def` and builds `yrc-bench-procgen.sif` at the repo root. To choose a different image name:

```bash
sbatch scripts/cluster3/build_apptainer.sbatch my-image.sif
```

The build script uses project-backed Apptainer temp/cache paths under:

```text
/path/to/cluster3/$USER/tmp/apptainer
```

Override them before submitting if needed:

```bash
export APPTAINER_TMPDIR=/path/to/cluster3/$USER/tmp/apptainer/tmp
export APPTAINER_CACHEDIR=/path/to/cluster3/$USER/tmp/apptainer/cache
```

## cluster3 Runtime

Submit an arbitrary command inside the image:

```bash
sbatch scripts/cluster3/run_container.sbatch \
  python eval_policy.py \
    -c configs/procgen_threshold.yaml \
    --model_file YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth \
    -num_rollouts 100
```

For a quick smoke test:

```bash
sbatch scripts/cluster3/run_container.sbatch \
  python -c "import torch, procgen, acs, pyod; print(torch.cuda.is_available())"
```

Runtime defaults:

- Repo bind: current repo root to `/workspace`
- cluster3 project bind: `/path/to/cluster3` to `/path/to/cluster3`
- Image: `yrc-bench-procgen.sif` at the repo root
- Working directory: `/workspace`

Useful overrides:

```bash
export YRC_SIF_PATH=/path/to/yrc-bench-procgen.sif
export REPO_DIR_OVERRIDE=/path/to/yrc-bench-fork
export YRC_EXTRA_BINDS=/scratch1/$USER:/scratch1/$USER,/project/$USER:/project/$USER
```

## cluster3 Eval Submission

`scripts/run_eval.py` can submit eval jobs that run inside the Apptainer image:

```bash
python scripts/run_eval.py \
  --server cluster3 \
  --use-container \
  --container-image yrc-bench-procgen.sif \
  --prefix icml \
  --exp-ids 0 1 2 3 \
  --env coinrun \
  --method max-prob \
  --runs-per-gpu 4
```

The cluster3 wrapper supplies `--server cluster3 --use-container --container-image ...`
for you:

```bash
scripts/cluster3/run_eval_container.sh \
  --prefix icml \
  --exp-ids 0 1 2 3 \
  --env coinrun \
  --method max-prob \
  --runs-per-gpu 4
```

Set `YRC_SIF_PATH` if the image is not at the repo root:

```bash
YRC_SIF_PATH=/path/to/cluster3/$USER/images/yrc-bench-procgen.sif \
  scripts/cluster3/run_eval_container.sh ...
```

Relative `--container-image` paths are resolved from the repo root. For `--server cluster3`,
the launcher automatically binds:

```text
/path/to/cluster3:/path/to/cluster3
```

Add extra binds as needed:

```bash
python scripts/run_eval.py ... \
  --use-container \
  --container-bind /scratch1/$USER:/scratch1/$USER
```

The default execution backend remains conda. `--use-container` is a shortcut for
`--execution apptainer`.

## Rebuild Triggers

Rebuild the base image when system packages, CUDA, Python, or PyTorch change.

Rebuild the app image or `.sif` when `requirements.txt`, `lib/acs`, `lib/procgen`, or `lib/pyod` changes.

Normal changes to `YRC/`, `configs/`, `scripts/`, checkpoints, or analysis code are picked up through the runtime bind mount.

## Notes

- This container path follows the current practical setup: Python 3.8 plus `requirements.txt`, with Procgen-specific local packages made explicit for the image.
- The image is Procgen-focused. MiniGrid and Cliport dependencies are intentionally not installed.
- `scripts/run_eval.py` supports containerized Slurm jobs through `--use-container`.
  Other Python Slurm launchers still generate conda-based jobs.
