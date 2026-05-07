# Cluster Robomimic Setup

Running robomimic training jobs on shared clusters (USC CARC and NCSA Delta) using Apptainer (Singularity) containers.

Each supported cluster has its own subfolder under `src/inf_ope/vgps_experiments/scripts/`:

- `scripts/carc/` — USC CARC (Discovery)
- `scripts/delta/` — NCSA Delta (ACCESS)

The "flag" for picking a cluster is the path you submit from. The actual command run inside the container (build, download, precompute, train) is identical across clusters; only the SLURM directives, module commands, and bind paths differ.

## Prerequisites (both clusters)

- `spython` installed on the cluster (`pipx install spython`)
- Git submodules initialized:
  ```bash
  git submodule update --init --recursive src/inf_ope/external/robomimic_infope src/inf_ope/external/robosuite
  ```
- Docker Hub authentication on your local machine (one-time, for pushing the base image — only needed when rebuilding the base):
  1. In Docker Hub, create a personal access token from **Account Settings → Personal access tokens**
  2. Log in:
     ```bash
     echo <YOUR_DOCKERHUB_TOKEN> | docker login -u <YOUR_DOCKERHUB_USERNAME> --password-stdin
     ```
  The base image lives under `docker.io/pavelcz/inf-ope-robomimic-base` (the repo owner's namespace). Collaborators with push access to that Docker Hub repository can push.

## Image architecture

The container image is split into two Dockerfiles:

- **`docker/Dockerfile.base`** — System packages (apt-get), Python venv, PyTorch. Rarely changes. Built with Docker locally and pushed to `docker.io/pavelcz/inf-ope-robomimic-base:latest`.
- **`docker/Dockerfile`** — Installs robosuite and robomimic on top of the base. This is converted to an Apptainer def and built directly on the cluster.

The split exists because shared-cluster Apptainer cannot run apt-get without fakeroot (which fails due to missing user namespace support on most NSF/CARC clusters). The app-layer Dockerfile only needs pip install, which works without root.

## Cluster-specific values

| Setting           | CARC                                          | Delta                                              |
|-------------------|-----------------------------------------------|----------------------------------------------------|
| Account (CPU)     | `biyik_1165`                                  | `bgya-delta-cpu` if allocated, else fall back to GPU account |
| Account (GPU)     | `biyik_1165`                                  | `bgya-delta-gpu`                                   |
| GPU partition     | `gpu`                                         | `gpuA100x4`                                        |
| Build account     | `biyik_1165` (CPU build)                      | `bgya-delta-gpu` + GPU partition (no CPU allocation) |
| GPU flag          | `--gres=gpu:1`                                | `--gpus-per-node=1`                                |
| Module init       | `module purge && module load apptainer`       | `module reset` (apptainer is system-wide on Delta) |
| Dataset bind      | `/scr/shared/datasets:/data`                  | `/projects/bgya/$USER/data:/data`                  |
| Project bind      | `/project2/biyik_1165:/project2/biyik_1165`   | `/projects/bgya:/projects/bgya`                    |
| Apptainer tmp     | `/project2/biyik_1165/$USER/tmp/apptainer`    | `/scratch/bgya/$USER/tmp/apptainer`                |

If you need to change the dataset path on Delta (e.g. to `/scratch/bgya/$USER/data` for faster I/O at the cost of periodic purges), edit the `--bind` lines in `scripts/delta/*.sbatch` and the download script.

## 1. Build / update the base image (rare, local machine)

Only needed when system packages or PyTorch version change:

```bash
docker build -f docker/Dockerfile.base -t docker.io/pavelcz/inf-ope-robomimic-base:latest .
docker push docker.io/pavelcz/inf-ope-robomimic-base:latest
```

## 2. Build the .sif on the cluster

Use a Slurm job (recommended on shared clusters):

```bash
# CARC
sbatch src/inf_ope/vgps_experiments/scripts/carc/build_apptainer.sbatch

# Delta
sbatch src/inf_ope/vgps_experiments/scripts/delta/build_apptainer.sbatch
```

The script generates `apptainer.def` from `docker/Dockerfile`, sets cluster-appropriate Apptainer cache/tmp directories, and builds `inf_ope.sif` at the repo root.

To build with a different output name:

```bash
sbatch src/inf_ope/vgps_experiments/scripts/<cluster>/build_apptainer.sbatch my_image.sif
```

Logs go to `logs/build-apptainer_<jobid>.out` and `.err`.

The build pulls the base image from Docker Hub and installs robosuite and robomimic on top.

`inf_ope.sif` and `apptainer.def` are both gitignored.

## 3. Download the dataset

The download script runs on a login/transfer node (no Slurm needed):

```bash
# CARC
bash src/inf_ope/vgps_experiments/scripts/carc/download_dataset.sh

# Delta
bash src/inf_ope/vgps_experiments/scripts/delta/download_dataset.sh
```

This binds the cluster's data directory to `/data` inside the container and runs `download_datasets.py` to fetch the `lift` task (multi-human, raw HDF5).

## 4. Submit a training job

Edit the appropriate `train_robomimic.sbatch` if you need to adjust bind mounts or dataset paths, then submit:

```bash
# CARC
sbatch src/inf_ope/vgps_experiments/scripts/carc/train_robomimic.sbatch

# Delta
sbatch src/inf_ope/vgps_experiments/scripts/delta/train_robomimic.sbatch
```

Logs go to `logs/robomimic-train_<jobid>.out` and `.err`.

## 5. Precompute target/behavior actions

```bash
sbatch src/inf_ope/vgps_experiments/scripts/<cluster>/precompute_actions.sbatch \
    <policy-dir> <policy-epoch> [extra args...]
```

Example on Delta:

```bash
sbatch src/inf_ope/vgps_experiments/scripts/delta/precompute_actions.sbatch \
    /projects/bgya/pczempin/data/p-ope/policies/bc_rnn_trained_models/bc_rnn_gmm/20260330213313 \
    200
```

## 6. Interactive GPU session (optional)

For debugging or short experiments, request an interactive session:

### CARC

```bash
salloc --account=biyik_1165 --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=1:00:00
module load apptainer
apptainer exec --nv \
    --bind "$PWD":/workspace \
    --bind /scr/shared/datasets:/data \
    --env MUJOCO_GL=egl \
    --pwd /workspace \
    inf_ope.sif bash
```

### Delta

```bash
srun --account=bgya-delta-gpu --partition=gpuA100x4-interactive \
     --gpus-per-node=1 --cpus-per-task=8 --mem=16G --time=1:00:00 \
     --pty bash
module reset
apptainer exec --nv \
    --bind "$PWD":/workspace \
    --bind /projects/bgya/$USER/data:/data \
    --env MUJOCO_GL=egl \
    --pwd /workspace \
    inf_ope.sif bash
```

Inside the container you can run commands directly:

```bash
python src/inf_ope/external/robomimic_infope/robomimic/scripts/train.py \
    --config src/inf_ope/external/robomimic_infope/robomimic/exps/templates/bc-rnn.json \
    --dataset /data/robomimic0.5/lift/mh/image_v15.hdf5
```

## Rebuilding the image

If you update robomimic or robosuite submodules, repeat step 2:

```bash
sbatch src/inf_ope/vgps_experiments/scripts/<cluster>/build_apptainer.sbatch
```

If you update system packages or PyTorch, repeat steps 1 and 2.

## Troubleshooting

- **`CUDA not available` inside the container**: Make sure you pass `--nv` to `apptainer exec` and that the job has a GPU allocated (`--gres=gpu:1` on CARC, `--gpus-per-node=1` on Delta).
- **`MUJOCO_GL` errors**: The image defaults to `osmesa` (CPU rendering). For GPU rendering, pass `--env MUJOCO_GL=egl`. If EGL fails, fall back to `--env MUJOCO_GL=osmesa`.
- **Permission errors on bind mounts**: Apptainer runs as your user by default. Make sure you have read access to the dataset paths. On Delta, the `/projects/bgya/...` path requires your user to be a member of the `bgya` project.
- **`Directory '/opt/src/robosuite' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.`**: Git submodules are not populated on the cluster. Run `git submodule update --init --recursive src/inf_ope/external/robomimic_infope src/inf_ope/external/robosuite` from the repo root, then re-run the build.
- **`pthread_create failed: Resource temporarily unavailable` during build**: Run the build through Slurm (`build_apptainer.sbatch`) instead of on login nodes, and keep `APPTAINER_TMPDIR` / `APPTAINER_CACHEDIR` on a large writable directory (the build scripts set these by default).
- **Out of disk building `.sif`**: Apptainer uses `/tmp` during builds. The build scripts set `APPTAINER_TMPDIR` to a large project-backed (CARC) or scratch-backed (Delta) directory. If you still hit space limits, override these env vars before submitting:
  ```bash
  # CARC
  export APPTAINER_TMPDIR=/project2/biyik_1165/$USER/tmp/apptainer/tmp
  export APPTAINER_CACHEDIR=/project2/biyik_1165/$USER/tmp/apptainer/cache

  # Delta
  export APPTAINER_TMPDIR=/scratch/bgya/$USER/tmp/apptainer/tmp
  export APPTAINER_CACHEDIR=/scratch/bgya/$USER/tmp/apptainer/cache
  ```
- **Delta: `Invalid account or account/partition combination specified`**: Check that you have a GPU allocation by running `accounts` on Delta. The account format is `<project>-delta-{cpu,gpu}` and must match the partition type. If you only have a `-delta-gpu` account, you must submit to a GPU partition (the build script does this — it requests 1 GPU even though the build itself doesn't use it).
