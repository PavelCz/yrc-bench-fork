# CARC Dockerfile to Apptainer Handoff

Use this file to tell another Codex session how to set up a Dockerfile-to-Apptainer pipeline for a different repo on USC CARC. The Slurm account and common CARC paths below are intentionally concrete because they are shared across our CARC projects. Replace placeholders such as `<repo>`, `<image>`, `<dockerhub-user>`, `<package-path>`, and `<entrypoint-command>`.

## Goal

Build a Docker image locally for system-level dependencies, push it to a registry, then build a repo-specific Apptainer `.sif` on CARC from a lightweight Dockerfile. The CARC build should run through Slurm, use project-backed temp/cache directories, and fail loudly if local source packages or submodules are missing.

This pattern is useful because CARC Apptainer builds often cannot run `apt-get` reliably under unprivileged fakeroot. Put `apt-get`, CUDA/PyTorch/system libraries, and other root-level setup in a base Docker image built outside CARC. On CARC, only run non-root-safe install steps such as `pip install`.

## Expected Repo Layout

Recommended layout:

```text
<repo>/
  docker/
    Dockerfile.base
    Dockerfile
  scripts/
    carc/
      build_apptainer.sbatch
      train.sbatch
  logs/
```

Add generated artifacts to `.gitignore`:

```gitignore
*.sif
apptainer.def
```

## Base Image

Create `docker/Dockerfile.base` for system dependencies. Build this locally or anywhere Docker is available, not on CARC.

Template:

```dockerfile
FROM nvidia/cuda:<cuda-tag>-devel-ubuntu<ubuntu-version>

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# Install heavyweight dependencies here, for example:
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Build and push:

```bash
docker build -f docker/Dockerfile.base -t docker.io/<dockerhub-user>/<image>-base:latest .
docker push docker.io/<dockerhub-user>/<image>-base:latest
```

## App Dockerfile

Create `docker/Dockerfile` for repo-specific Python installs. This is converted to an Apptainer definition by `spython`.

Avoid copying local source trees to `/tmp`; use a stable path such as `/opt/src`. CARC/Apptainer temp handling can make `/tmp` a bad staging location.

Use separate `RUN` commands or explicit `|| exit 1` so failed installs do not produce a misleading `.sif`.

Template:

```dockerfile
FROM docker.io/<dockerhub-user>/<image>-base:latest

COPY <package-path-1> /opt/src/<package-1>
RUN /opt/venv/bin/pip install --no-cache-dir /opt/src/<package-1> || exit 1
RUN rm -rf /opt/src/<package-1>

# Repeat for any local package or submodule that must be installed.
# COPY <package-path-2> /opt/src/<package-2>
# RUN /opt/venv/bin/pip install --no-cache-dir /opt/src/<package-2> || exit 1
# RUN rm -rf /opt/src/<package-2>

COPY requirements.txt /opt/src/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /opt/src/requirements.txt || exit 1

WORKDIR /workspace
CMD ["bash"]
```

If the repo itself is importable and should be installed into the image, add:

```dockerfile
COPY . /opt/src/<repo-name>
RUN /opt/venv/bin/pip install --no-cache-dir /opt/src/<repo-name> || exit 1
RUN rm -rf /opt/src/<repo-name>
```

If the repo will be bind-mounted at runtime and you only need dependencies in the image, do not copy the whole repo into the image.

## CARC Build Script

Create `scripts/carc/build_apptainer.sbatch`.

Template:

```bash
#!/bin/bash

#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=build-apptainer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load apptainer

REPO_DIR="${REPO_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$PWD}}"
if command -v git >/dev/null 2>&1; then
    GIT_ROOT="$(git -C "${REPO_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${GIT_ROOT}" ]]; then
        REPO_DIR="${GIT_ROOT}"
    fi
fi

if [[ ! -f "${REPO_DIR}/docker/Dockerfile" ]]; then
    echo "Could not locate repo root at '${REPO_DIR}'. Submit from repo root or set REPO_DIR_OVERRIDE."
    exit 1
fi

# Add one check for every local package or submodule copied by docker/Dockerfile.
for pkg_dir in \
    "<package-path-1>"
do
    if [[ ! -f "${REPO_DIR}/${pkg_dir}/setup.py" && ! -f "${REPO_DIR}/${pkg_dir}/pyproject.toml" ]]; then
        echo "Package sources missing or not installable: ${REPO_DIR}/${pkg_dir}"
        echo "If this is a submodule, initialize it before building:"
        echo "  git submodule update --init --recursive ${pkg_dir}"
        exit 1
    fi
done

SIF_NAME="${1:-<image>.sif}"
SIF_PATH="${REPO_DIR}/${SIF_NAME}"

mkdir -p "${REPO_DIR}/logs"

DEFAULT_APPTAINER_ROOT="/project2/biyik_1165/$USER/tmp/apptainer"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${DEFAULT_APPTAINER_ROOT}/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${DEFAULT_APPTAINER_ROOT}/cache}"
export GOMAXPROCS="${GOMAXPROCS:-4}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

if ! command -v spython >/dev/null 2>&1; then
    echo "spython is not available. Install it first, for example: pipx install spython"
    exit 1
fi

cd "${REPO_DIR}"
spython recipe docker/Dockerfile > apptainer.def

# Some Apptainer/spython combinations can continue after a failing %post command.
# Keep explicit '|| exit 1' on critical RUN commands in docker/Dockerfile.
apptainer build "${SIF_PATH}" apptainer.def
```

Submit from the repo root:

```bash
sbatch scripts/carc/build_apptainer.sbatch
```

Build a differently named image:

```bash
sbatch scripts/carc/build_apptainer.sbatch <other-name>.sif
```

## Runtime Script

Create a runtime Slurm script such as `scripts/carc/train.sbatch`.

Template:

```bash
#!/bin/bash

#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=<job-name>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load apptainer

REPO_DIR="${REPO_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$PWD}}"
if command -v git >/dev/null 2>&1; then
    GIT_ROOT="$(git -C "${REPO_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${GIT_ROOT}" ]]; then
        REPO_DIR="${GIT_ROOT}"
    fi
fi

SIF="${REPO_DIR}/<image>.sif"
mkdir -p "${REPO_DIR}/logs"

apptainer exec --nv \
    --bind "${REPO_DIR}":/workspace \
    --bind /project2/biyik_1165:/project2/biyik_1165 \
    --env PYTHONUNBUFFERED=1 \
    --pwd /workspace \
    "${SIF}" \
    <entrypoint-command>
```

Add dataset binds as needed, for example:

```bash
--bind /scr/shared/datasets:/data
```

For MuJoCo or OpenGL workloads, add one of:

```bash
--env MUJOCO_GL=egl
--env MUJOCO_GL=osmesa
```

## Interactive Test on CARC

Request an interactive GPU session:

```bash
salloc --account=biyik_1165 --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=1:00:00
module purge
module load apptainer
```

Run smoke tests:

```bash
apptainer exec --nv \
    --bind "$PWD":/workspace \
    --pwd /workspace \
    <image>.sif \
    /opt/venv/bin/python -c "import sys; print(sys.executable); print('ok')"
```

Test important imports:

```bash
apptainer exec --nv \
    --bind "$PWD":/workspace \
    --pwd /workspace \
    <image>.sif \
    /opt/venv/bin/python -c "import <module_1>; import <module_2>; print('imports ok')"
```

Check CUDA from inside the image if using PyTorch:

```bash
apptainer exec --nv <image>.sif /opt/venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## CARC Defaults and Notes

- Use `/project2/biyik_1165/$USER/tmp/apptainer` for `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR`.
- The script should create these directories with `mkdir -p`; they may not exist yet.
- Avoid `/scratch/$USER`; it may not exist or may not be writable on CARC.
- `xattr ... ignoring ENOTSUP` warnings from project filesystems usually mean the filesystem does not support extended attributes. They are usually not fatal by themselves.
- `User not listed in /etc/subuid, trying root-mapped namespace` is common on shared clusters. Treat later package install failures as the actionable issue.
- Always run builds through Slurm instead of on login nodes. Large builds can hit process, memory, time, or temp-space limits.
- Install `spython` on CARC before building, for example with `pipx install spython`.

## Failure Modes to Guard Against

`mkdir: cannot create directory '/scratch': Permission denied`

Use a project-backed temp path:

```bash
export APPTAINER_TMPDIR=/project2/biyik_1165/$USER/tmp/apptainer/tmp
export APPTAINER_CACHEDIR=/project2/biyik_1165/$USER/tmp/apptainer/cache
```

`Directory '/opt/src/<package>' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.`

The copied local package is empty, uninitialized, or not a Python package. If it is a submodule:

```bash
git submodule update --init --recursive <package-path>
```

If it is not a Python package, do not `pip install` that directory; install the correct subdirectory or use `requirements.txt`.

`Build complete` appears even though `pip install` printed errors

Do not trust the `.sif`. Add `|| exit 1` to critical Dockerfile `RUN` lines and rebuild. Verify imports inside the image before submitting long jobs.

`CUDA not available`

Make sure the runtime command uses `apptainer exec --nv`, the job requested a GPU with `#SBATCH --gres=gpu:1`, and the image has a CUDA-compatible PyTorch or framework build.

`Out of disk building .sif`

Move `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` to a larger project directory. Also make sure Docker or Apptainer caches are not filling a small home directory.

## Final Checklist for the Other Repo

1. Create `docker/Dockerfile.base` with system packages and heavyweight dependencies.
2. Build and push `docker.io/<dockerhub-user>/<image>-base:latest` outside CARC.
3. Create `docker/Dockerfile` that inherits from the base image and only performs non-root-safe app installs.
4. Stage local package copies under `/opt/src`, not `/tmp`.
5. Add explicit `|| exit 1` to critical install `RUN` commands.
6. Create `scripts/carc/build_apptainer.sbatch` with project-backed temp/cache paths.
7. Add preflight checks for every copied local package or submodule.
8. Build the `.sif` on CARC with `sbatch`.
9. Smoke-test imports and CUDA inside the `.sif`.
10. Create runtime Slurm scripts with `apptainer exec --nv`, repo bind, dataset binds, and the right entrypoint command.
