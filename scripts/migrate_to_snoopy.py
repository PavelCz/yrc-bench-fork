#!/usr/bin/env python3
"""Copy current RNN results and the artifacts needed to rerun them onto snoopy.

The destination is the snoopy layout in ``scripts/common.py``:
``snoopy:/scr/pavel/data/goal-misgen``. Seeds 0–7 are included. Nothing is
copied unless ``run`` is given ``--execute``. ``plan`` prints the files and
groups a chunk into parts of about 2GB; individual larger files stay whole.
Pass ``--part N`` to copy one of those parts.

What each chunk copies
----------------------
``seeds``
    ``seeds/icml/{0..7}.json``, ``seeds/icml/ensemble-{0..7}_{0..3}.json``,
    and ``seeds/neurips_extra_ood_train_1024/{0..7}.json``.
``svdd``
    Current image and latent models only. CoinRun is
    ``trained_svdd/neurips05``. Maze is
    ``trained_svdd/tmlr-maze-svdd-fix-model``. Heist is
    ``trained_svdd/tmlr-heist01``. Maze models under ``neurips05`` are omitted;
    those were replaced by the maze-fix models.
``checkpoints``
    The file evaluation would load: the highest ``model_<steps>.pth`` in the
    newest timestamp directory. That is the CoinRun novice (0%) and expert
    (50%), the Maze novice (0%) and non-robust 50% expert, the Heist novice
    (0%), the Heist 400M expert under ``policy/heist400`` (``model_400031744.pth``
    when that is the highest step), and the Maze robust400 expert under
    ``policy/neurips/maze_afh_random_start``. Ensemble members are taken from
    ``policy/icml/.../ensembles/`` when that copy exists, otherwise from the
    RNN checkout's ``lib/train-procgen-pytorch/logs/train/``. On snoopy they
    are always written under ``policy/icml/<env>/ensembles/``, which is where
    evaluation looks. Intermediate PPO checkpoints are not copied.
``evals-recoverable``, ``evals-proxy-fail``, ``evals-proxy-penalty``
    The newest ``.npz`` for each environment, method, and seed among the
    prefixes those figure scripts still read. Videos and images are not
    copied. An older curve for the same environment, method, and seed is
    dropped. The same timestamp in more than one tree keeps the RNN checkout,
    then ``yrc-bench-afhp-range-retry``, then the data-tree evals.
``rollouts-<env>-exp<N>``
    The gather directory that the current SVDD model for that seed was
    trained on. CoinRun is ``rollouts/neurips03/coinrun``. Heist is
    ``rollouts/tmlr-heist01/heist``. Maze seeds 0–3 are
    ``rollouts/tmlr-maze-svdd-fix/maze``. Maze seeds 4–7 are
    ``rollouts/neurips03/maze``. The large ``neurips03`` Maze gathers for
    seeds 0–3 are not the ones those models used.

Eval ``.npz`` files are read from three trees, in tie-break order:

1. ``/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork/experiments/evals``
2. ``/nas/ucb/czempin/code/goal-misgen/yrc-bench-afhp-range-retry/experiments/evals``
   (the Heist proxy-fail max-logit seed-5 curve was written only here)
3. ``/nas/ucb/czempin/data/goal-misgen/experiments/evals``

Other artifacts are read from ``/nas/ucb/czempin/data/goal-misgen`` on RNN,
except ensemble members that exist only in the checkout's train logs.

Disk
----
Measured on RNN in October 2026. ``du`` reports about 67 GB allocated and
about 188 GB apparent for the gathers. The memmaps have no holes: ``SEEK_HOLE``
on the seed-0 CoinRun, Maze, and Heist files reports one data extent covering
the whole file, and sampled windows contain no 4 KB run of zeros. The smaller
number is compressed allocation on the ``/mnt/tank`` NFS export. ``/scr`` on
snoopy is ext4, so the copy stores the logical bytes, about 188 GB for the
gathers. ``--sparse`` does not reproduce the tank's compressed size. Seeds,
SVDD models, and final checkpoints are about 2 GB more, and the winning
curves are about 4 GB. The copy needs about 195 GB free on ``/scr``. On
15 October 2026, ``/scr`` had about 124 GB free.

The default local relay stages one transfer at a time in the local temporary
directory (configurable with ``TMPDIR``). It needs free space for the largest
file or directory in the selected part, which can exceed 2 GB. Staged data
is removed after each transfer, including on failure.

SSH
---
``--source-host`` defaults to ``rnn`` and ``--dest-host`` to ``snoopy``.
Those are local SSH aliases. In ``~/.ssh/config``, ``snoopy`` is
``snoopy1.usc.edu``. The default ``--via local`` downloads each transfer to
local temporary storage, then uploads it to snoopy, so both aliases must
resolve here. Each transfer takes two sequential rsync operations.
``--via dest`` makes snoopy pull
from the source host instead, and that source host name must resolve on
snoopy. The local alias ``rnn`` usually does not.

Examples::

    python scripts/migrate_to_snoopy.py list
    python scripts/migrate_to_snoopy.py plan seeds
    python scripts/migrate_to_snoopy.py run seeds --execute
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Sequence

from common import (
    SERVER_PATHS,
    get_ensemble_run_dir,
    get_env_folder,
    get_heist400_run_dir,
    get_icml_run_dir,
)


EXP_IDS = tuple(range(8))
ENSEMBLE_MEMBERS = tuple(range(4))
# Target part size; individual larger files stay whole.
CHUNK_BUDGET_BYTES = 2_000_000_000
SOURCE_HOST = "rnn"
DEST_HOST = "snoopy"
RNN_REPO = "/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork"
RNN_RETRY_REPO = "/nas/ucb/czempin/code/goal-misgen/yrc-bench-afhp-range-retry"

# Prefixes still read by the paper figure scripts. A later timestamp for the
# same environment, method, and seed replaces an earlier one at plan time.
RECOVERABLE_PREFIXES = (
    "tmlr2-svdd",
    "tmlr3-svdd",
    "tmlr4-svdd",
    "tmlr5-svdd",
    "imcl04",
    "icml04",
    "tmlr-oracle-lb",
    "tmlr-robust-maze-1",
    "tmlr-robust-maze-3",
    "tmlr-oracle-lb-robust400",
    "tmlr-maze-svdd-fix",
    "tmlr-heist03-expert400",
)
PROXY_FAIL_PREFIXES = (
    "tmlr-proxy-fail1",
    "tmlr-proxy-fail2",
    "tmlr-proxy-fail3",
    "tmlr-proxy-fail4",
    "tmlr-oracle-lb-proxy_fail",
    "tmlr-proxy-fail-fix",
    "tmlr-heist03-expert400-proxy-fail",
)
PROXY_PENALTY_PREFIXES = (
    "proxy-penalty01",
    "tmlr-heist04-expert400-proxy-penalty",
)
EVAL_PREFIXES = {
    "recoverable": frozenset(RECOVERABLE_PREFIXES),
    "proxy-fail": frozenset(PROXY_FAIL_PREFIXES),
    "proxy-penalty": frozenset(PROXY_PENALTY_PREFIXES),
}
ALLOWED_EVAL_PREFIXES = frozenset().union(*EVAL_PREFIXES.values())

_ENVS = (
    "coinrun_proxy_penalty",
    "coinrun_proxy_fail",
    "maze_proxy_penalty",
    "maze_proxy_fail",
    "heist_proxy_penalty",
    "heist_proxy_fail",
    "coinrun",
    "maze_afh",
    "heist",
    "maze",
)
_ENV_PATTERN = "|".join(re.escape(env) for env in _ENVS)
_EXPERIMENT_RE = re.compile(rf"^(.+)_({_ENV_PATTERN})_exp(\d+)$")
_ROBUST_EXPERIMENT_RE = re.compile(
    rf"^(.+)_(robust200|robust400)_({_ENV_PATTERN})_exp(\d+)$"
)
_MODEL_RE = re.compile(r"model_(\d+)\.pth$")

SshFunc = Callable[[str, str], str]
_SSH_ATTEMPTS = 4
_SSH_OPTIONS = (
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=120",
    "-o",
    "ControlPath=/tmp/migrate-to-snoopy-%r@%h:%p",
)


@dataclass(frozen=True)
class Transfer:
    """One file or directory to copy."""

    source: str
    dest: str
    nbytes: int
    label: str
    is_dir: bool = False


@dataclass(frozen=True)
class NpzHit:
    """One eval curve file found under an experiments/evals tree."""

    source_root: str
    relpath: str
    prefix: str
    env: str
    method: str
    exp_id: int
    run_name: str
    nbytes: int
    source_priority: int


def data_root(server: str) -> Path:
    """Return the goal-misgen data root for a SERVER_PATHS entry."""
    return Path(SERVER_PATHS[server]["checkpoint_base"]).parents[1]


def robust_maze_run_dir(checkpoint_base: str, exp_id: int) -> Path:
    """Return the robust400 Maze expert run directory."""
    return (
        Path(checkpoint_base).parent
        / "neurips"
        / "maze_afh_random_start"
        / f"icml2_maze_exp{exp_id}_50p_random_start"
    )


def train_log_ensemble_dir(env: str, exp_id: int, member_id: int) -> Path:
    """Return the RNN train-log directory for one ensemble member."""
    return (
        Path(RNN_REPO)
        / "lib"
        / "train-procgen-pytorch"
        / "logs"
        / "train"
        / get_env_folder(env)
        / f"icml2_ensemble_{env}_exp{exp_id}_m{member_id}"
    )


def ensemble_dest_relpath(
    env: str, exp_id: int, member_id: int, timestamp: str, filename: str
) -> str:
    """Return the policy-tree path eval uses for an ensemble member."""
    folder = get_env_folder(env)
    return (
        f"policy/icml/{folder}/ensembles/"
        f"icml2_ensemble_{env}_exp{exp_id}_m{member_id}/{timestamp}/{filename}"
    )


def seed_relpaths(exp_ids: Sequence[int] = EXP_IDS) -> list[str]:
    """Return seed files for the given experiment ids."""
    relpaths = []
    for exp_id in exp_ids:
        relpaths.append(f"seeds/icml/{exp_id}.json")
        relpaths.append(f"seeds/neurips_extra_ood_train_1024/{exp_id}.json")
        for member_id in ENSEMBLE_MEMBERS:
            relpaths.append(f"seeds/icml/ensemble-{exp_id}_{member_id}.json")
    return relpaths


def svdd_relpaths(exp_ids: Sequence[int] = EXP_IDS) -> list[str]:
    """Return current SVDD model directories.

    Maze models under ``trained_svdd/neurips05`` are omitted. Eval and the
    later Maze trains use ``tmlr-maze-svdd-fix-model``.
    """
    relpaths = []
    for exp_id in exp_ids:
        for feature in ("image", "latent"):
            relpaths.append(
                f"trained_svdd/neurips05/svdd_coinrun_{feature}_exp{exp_id}"
            )
            relpaths.append(
                f"trained_svdd/tmlr-maze-svdd-fix-model/svdd_maze_{feature}_exp{exp_id}"
            )
            relpaths.append(
                f"trained_svdd/tmlr-heist01/svdd_heist_{feature}_exp{exp_id}"
            )
    return relpaths


def rollout_relpath(env: str, exp_id: int) -> str:
    """Return the gather directory the current SVDD model for this seed used."""
    if env == "coinrun":
        return f"rollouts/neurips03/coinrun/gather_coinrun_exp{exp_id}"
    if env == "heist":
        return f"rollouts/tmlr-heist01/heist/gather_heist_exp{exp_id}"
    if env == "maze" and exp_id <= 3:
        return f"rollouts/tmlr-maze-svdd-fix/maze/gather_maze_exp{exp_id}"
    if env == "maze":
        return f"rollouts/neurips03/maze/gather_maze_exp{exp_id}"
    raise ValueError(f"Unsupported rollout env: {env}")


_CHUNK_BLURBS = {
    "seeds": (
        "seeds/icml/{0..7}.json, ensemble-{0..7}_{0..3}.json, "
        "and neurips_extra_ood_train_1024/{0..7}.json"
    ),
    "svdd": (
        "CoinRun trained_svdd/neurips05, Maze tmlr-maze-svdd-fix-model, "
        "Heist tmlr-heist01; image and latent, seeds 0-7"
    ),
    "checkpoints": (
        "newest highest-step checkpoint per run, including Heist 400M, "
        "Maze robust400, and the non-robust Maze 50% experts; ensembles "
        "land under policy/icml/.../ensembles/"
    ),
    "evals-recoverable": "newest .npz for the recoverable figure prefixes",
    "evals-proxy-fail": "newest .npz for the proxy-fail figure prefixes",
    "evals-proxy-penalty": "newest .npz for the proxy-penalty figure prefixes",
}


def logical_chunk_names() -> list[str]:
    """Return chunk names in the order they should be copied."""
    names = list(_CHUNK_BLURBS)
    for env in ("coinrun", "maze", "heist"):
        for exp_id in EXP_IDS:
            names.append(f"rollouts-{env}-exp{exp_id}")
    return names


def chunk_description(name: str) -> str:
    """Return the one-line description printed by ``list``."""
    if name in _CHUNK_BLURBS:
        return _CHUNK_BLURBS[name]
    env, _, exp_text = name[len("rollouts-") :].rpartition("-exp")
    return rollout_relpath(env, int(exp_text))


def parse_experiment_dir(
    dir_name: str,
) -> Optional[tuple[str, str, int, Optional[str]]]:
    """Return ``(prefix, env, exp_id, robust_variant)`` for an eval directory."""
    robust_match = _ROBUST_EXPERIMENT_RE.match(dir_name)
    if robust_match is not None:
        prefix, variant, env, exp_id = robust_match.groups()
        return prefix, env, int(exp_id), variant
    match = _EXPERIMENT_RE.match(dir_name)
    if match is None:
        return None
    prefix, env, exp_id = match.groups()
    return prefix, env, int(exp_id), None


def canonicalize_method(method: str) -> str:
    """Normalize a method directory name to kebab-case, keeping robust suffixes."""
    for variant in ("robust200", "robust400"):
        suffix = f"_{variant}"
        if method.endswith(suffix):
            base = method[: -len(suffix)].replace("_", "-")
            return f"{base}{suffix}"
    return method.replace("_", "-")


def method_from_dir(dir_name: str, env: str, exp_id: int) -> Optional[str]:
    """Return the raw method name from ``{env}_{method}_exp{id}``."""
    prefix = f"{env}_"
    suffix = f"_exp{exp_id}"
    if dir_name.startswith(prefix) and dir_name.endswith(suffix):
        method = dir_name[len(prefix) : -len(suffix)]
        if method:
            return method
    return None


def classify_npz(relpath: str) -> Optional[tuple[str, str, str, int, str]]:
    """Return prefix, env, canonical method, exp id, and run name for an npz.

    ``relpath`` is relative to an ``experiments/evals`` directory.
    """
    parts = Path(relpath).parts
    if len(parts) != 4 or not parts[-1].endswith(".npz"):
        return None
    parsed = parse_experiment_dir(parts[0])
    if parsed is None:
        return None
    prefix, env, exp_id, robust_variant = parsed
    if prefix not in ALLOWED_EVAL_PREFIXES or exp_id not in EXP_IDS:
        return None
    method = method_from_dir(parts[1], env, exp_id)
    if method is None:
        return None
    if robust_variant is not None and not method.endswith(("_robust200", "_robust400")):
        method = f"{method}_{robust_variant}"
    return prefix, env, canonicalize_method(method), exp_id, parts[2]


def select_winning_npz(hits: Sequence[NpzHit]) -> list[NpzHit]:
    """Keep the newest curve for each environment, method, and seed.

    An equal timestamp keeps the hit with the higher ``source_priority``.
    Every npz in that winning run directory is kept.
    """
    best_rank: dict[tuple[str, str, int], tuple[str, int]] = {}
    for hit in hits:
        key = (hit.env, hit.method, hit.exp_id)
        rank = (hit.run_name, hit.source_priority)
        current = best_rank.get(key)
        if current is None or rank > current:
            best_rank[key] = rank
    return [
        hit
        for hit in hits
        if best_rank[(hit.env, hit.method, hit.exp_id)]
        == (hit.run_name, hit.source_priority)
    ]


def prefix_family(prefix: str) -> str:
    """Return the eval chunk family that owns this prefix."""
    for family, prefixes in EVAL_PREFIXES.items():
        if prefix in prefixes:
            return family
    raise KeyError(prefix)


def choose_model_file(
    entries: Sequence[tuple[str, str, int, str]],
) -> Optional[tuple[str, str, int, str]]:
    """Pick the highest-step model in the newest timestamp directory.

    Each entry is ``(timestamp, filename, nbytes, absolute_path)``.
    """
    parsed = []
    for timestamp, filename, nbytes, path in entries:
        match = _MODEL_RE.fullmatch(filename)
        if match is not None:
            parsed.append((timestamp, int(match.group(1)), nbytes, path, filename))
    if not parsed:
        return None
    newest = max(item[0] for item in parsed)
    timestamp, _steps, nbytes, path, filename = max(
        (item for item in parsed if item[0] == newest),
        key=lambda item: item[1],
    )
    return timestamp, filename, nbytes, path


def assign_models_to_runs(
    files: Sequence[tuple[int, str]],
    run_dirs: Sequence[str],
) -> dict[str, tuple[str, str, int, str]]:
    """Map each run directory to the model file eval would load.

    ``files`` are ``(nbytes, absolute_path)`` listings. A model counts only
    when it sits directly inside a timestamp directory of ``run_dirs``.
    """
    grouped: dict[str, list[tuple[str, str, int, str]]] = {run: [] for run in run_dirs}
    prefixes = [(run.rstrip("/"), run.rstrip("/") + "/") for run in run_dirs]
    for nbytes, path in files:
        for run, prefix in prefixes:
            if not path.startswith(prefix):
                continue
            parts = path[len(prefix) :].split("/")
            if len(parts) == 2:
                grouped[run].append((parts[0], parts[1], nbytes, path))
            break
    chosen = {}
    for run, entries in grouped.items():
        model = choose_model_file(entries)
        if model is not None:
            chosen[run] = model
    return chosen


def group_by_budget(
    transfers: Sequence[Transfer], budget: int = CHUNK_BUDGET_BYTES
) -> list[list[Transfer]]:
    """Pack transfers into parts that stay within ``budget`` when they can.

    A single transfer larger than ``budget`` stays in its own part.
    """
    parts: list[list[Transfer]] = []
    current: list[Transfer] = []
    current_bytes = 0
    for transfer in transfers:
        if current and current_bytes + transfer.nbytes > budget:
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(transfer)
        current_bytes += transfer.nbytes
    if current:
        parts.append(current)
    return parts


def _relocate(path: str, source_root: Path, dest_root: Path) -> str:
    try:
        relpath = Path(path).relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{path} is outside {source_root}") from exc
    return str(dest_root / relpath)


def acting_run_dirs(checkpoint_base: str, exp_ids: Sequence[int]) -> list[str]:
    """Return novice, expert, Heist400, and robust400 run directories."""
    run_dirs = []
    for exp_id in exp_ids:
        run_dirs.extend(
            [
                str(get_icml_run_dir("coinrun", exp_id, checkpoint_base, 0)),
                str(get_icml_run_dir("coinrun", exp_id, checkpoint_base, 50)),
                str(get_icml_run_dir("maze", exp_id, checkpoint_base, 0)),
                str(get_icml_run_dir("maze", exp_id, checkpoint_base, 50)),
                str(get_icml_run_dir("heist", exp_id, checkpoint_base, 0)),
                str(get_heist400_run_dir(exp_id, checkpoint_base)),
                str(robust_maze_run_dir(checkpoint_base, exp_id)),
            ]
        )
    return run_dirs


def ensemble_run_pairs(
    checkpoint_base: str, exp_ids: Sequence[int]
) -> list[tuple[str, str, str, int, int]]:
    """Return ``(policy_dir, train_log_dir, env, exp_id, member_id)``."""
    pairs = []
    for env in ("coinrun", "maze", "heist"):
        for exp_id in exp_ids:
            for member_id in ENSEMBLE_MEMBERS:
                pairs.append(
                    (
                        str(
                            get_ensemble_run_dir(
                                env, exp_id, member_id, checkpoint_base
                            )
                        ),
                        str(train_log_ensemble_dir(env, exp_id, member_id)),
                        env,
                        exp_id,
                        member_id,
                    )
                )
    return pairs


def checkpoint_transfers(
    policy_files: Sequence[tuple[int, str]],
    train_files: Sequence[tuple[int, str]],
    *,
    source_root: Path,
    dest_root: Path,
    checkpoint_base: str,
    exp_ids: Sequence[int] = EXP_IDS,
) -> list[Transfer]:
    """Choose final checkpoints, falling back to train logs for ensembles."""
    transfers = []
    run_dirs = acting_run_dirs(checkpoint_base, exp_ids)
    chosen = assign_models_to_runs(policy_files, run_dirs)
    missing_runs = [run_dir for run_dir in run_dirs if run_dir not in chosen]
    for run_dir in missing_runs:
        print(f"Warning: missing checkpoint under {run_dir}", file=sys.stderr)
    for run_dir, (timestamp, filename, nbytes, path) in chosen.items():
        del timestamp, filename
        transfers.append(
            Transfer(
                source=path,
                dest=_relocate(path, source_root, dest_root),
                nbytes=nbytes,
                label=run_dir,
            )
        )

    for policy_dir, train_dir, env, exp_id, member_id in ensemble_run_pairs(
        checkpoint_base, exp_ids
    ):
        chosen_policy = assign_models_to_runs(policy_files, [policy_dir])
        chosen_train = assign_models_to_runs(train_files, [train_dir])
        if policy_dir in chosen_policy:
            timestamp, filename, nbytes, path = chosen_policy[policy_dir]
            source = path
        elif train_dir in chosen_train:
            timestamp, filename, nbytes, path = chosen_train[train_dir]
            source = path
        else:
            print(
                f"Warning: missing {env} ensemble exp{exp_id} m{member_id}",
                file=sys.stderr,
            )
            continue
        dest_rel = ensemble_dest_relpath(env, exp_id, member_id, timestamp, filename)
        transfers.append(
            Transfer(
                source=source,
                dest=str(dest_root / dest_rel),
                nbytes=nbytes,
                label=dest_rel,
            )
        )
    return transfers


def eval_transfers(
    hits: Sequence[NpzHit], family: str, dest_root: Path
) -> list[Transfer]:
    """Build transfers for the winning npz files in one figure family."""
    transfers = []
    for hit in select_winning_npz(hits):
        if prefix_family(hit.prefix) != family:
            continue
        transfers.append(
            Transfer(
                source=str(Path(hit.source_root) / hit.relpath),
                dest=str(dest_root / "experiments" / "evals" / hit.relpath),
                nbytes=hit.nbytes,
                label=hit.relpath,
            )
        )
    return transfers


def ssh_argv(host: str, command: str) -> list[str]:
    """Return an ssh command that reuses one connection per host."""
    return ["ssh", *_SSH_OPTIONS, host, command]


def rsync_rsh() -> str:
    """Return the ``rsync -e`` shell, matching :func:`ssh_argv`."""
    return " ".join(shlex.quote(part) for part in ("ssh", *_SSH_OPTIONS))


def run_ssh(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """Run ``cmd``, retrying when the SSH handshake times out."""
    result = subprocess.run(cmd, **kwargs)
    for attempt in range(1, _SSH_ATTEMPTS):
        if result.returncode == 0 or result.returncode != 255:
            return result
        time.sleep(2 * attempt)
        result = subprocess.run(cmd, **kwargs)
    return result


def _checked(cmd: Sequence[str], **kwargs) -> None:
    result = run_ssh(cmd, check=False, **kwargs)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def ssh_text(host: str, command: str) -> str:
    """Run ``command`` on ``host`` and return stdout."""
    result = run_ssh(
        ssh_argv(host, command),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh {host} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _remote_du(ssh: SshFunc, host: str, path: str) -> Optional[int]:
    quoted = shlex.quote(path)
    command = f"if [ -e {quoted} ]; then du -sb {quoted} | cut -f1; fi"
    output = ssh(host, command).strip()
    if not output:
        return None
    return int(output.split()[0])


def _remote_model_files(ssh: SshFunc, host: str, root: str) -> list[tuple[int, str]]:
    quoted = shlex.quote(root)
    command = (
        f"if [ -d {quoted} ]; then "
        f"find {quoted} -name 'model_*.pth' -printf '%s\\t%p\\n'; fi"
    )
    files = []
    for line in ssh(host, command).splitlines():
        if not line.strip():
            continue
        size_text, path = line.split("\t", 1)
        files.append((int(size_text), path))
    return files


def _remote_files_under(ssh: SshFunc, host: str, root: str) -> list[tuple[int, str]]:
    quoted = shlex.quote(root)
    command = f"if [ -d {quoted} ]; then find {quoted} -type f -printf '%s\\t%P\\n'; fi"
    files = []
    for line in ssh(host, command).splitlines():
        if not line.strip():
            continue
        size_text, relpath = line.split("\t", 1)
        files.append((int(size_text), relpath))
    return files


def _discover_eval_hits(ssh: SshFunc, source_host: str) -> list[NpzHit]:
    source_data = data_root("chai")
    roots = (
        (2, f"{RNN_REPO}/experiments/evals"),
        (1, f"{RNN_RETRY_REPO}/experiments/evals"),
        (0, str(source_data / "experiments" / "evals")),
    )
    hits = []
    for priority, root in roots:
        quoted = shlex.quote(root)
        command = (
            f"if [ -d {quoted} ]; then "
            f"find {quoted} -name '*.npz' -printf '%s\\t%P\\n'; fi"
        )
        for line in ssh(source_host, command).splitlines():
            if not line.strip():
                continue
            size_text, relpath = line.split("\t", 1)
            classified = classify_npz(relpath)
            if classified is None:
                continue
            prefix, env, method, exp_id, run_name = classified
            hits.append(
                NpzHit(
                    source_root=root,
                    relpath=relpath,
                    prefix=prefix,
                    env=env,
                    method=method,
                    exp_id=exp_id,
                    run_name=run_name,
                    nbytes=int(size_text),
                    source_priority=priority,
                )
            )
    return hits


def _sized_path_transfers(
    ssh: SshFunc,
    source_host: str,
    relpaths: Sequence[str],
    *,
    source_root: Path,
    dest_root: Path,
    is_dir: bool,
) -> list[Transfer]:
    transfers = []
    missing = []
    for relpath in relpaths:
        source = str(source_root / relpath)
        nbytes = _remote_du(ssh, source_host, source)
        if nbytes is None:
            missing.append(relpath)
            continue
        transfers.append(
            Transfer(
                source=source,
                dest=str(dest_root / relpath),
                nbytes=nbytes,
                label=relpath,
                is_dir=is_dir,
            )
        )
    for relpath in missing:
        print(f"Warning: missing {relpath}", file=sys.stderr)
    return transfers


def resolve_chunk(
    name: str,
    ssh: SshFunc,
    *,
    source_host: str = SOURCE_HOST,
) -> list[Transfer]:
    """Resolve one logical chunk to concrete transfers. Directories may be large."""
    source_root = data_root("chai")
    dest_root = data_root("snoopy")
    checkpoint_base = SERVER_PATHS["chai"]["checkpoint_base"]

    if name == "seeds":
        return _sized_path_transfers(
            ssh,
            source_host,
            seed_relpaths(),
            source_root=source_root,
            dest_root=dest_root,
            is_dir=False,
        )
    if name == "svdd":
        return _sized_path_transfers(
            ssh,
            source_host,
            svdd_relpaths(),
            source_root=source_root,
            dest_root=dest_root,
            is_dir=True,
        )
    if name == "checkpoints":
        policy_files = _remote_model_files(
            ssh, source_host, str(source_root / "policy")
        )
        train_root = Path(RNN_REPO) / "lib" / "train-procgen-pytorch" / "logs" / "train"
        train_files = _remote_model_files(ssh, source_host, str(train_root))
        return checkpoint_transfers(
            policy_files,
            train_files,
            source_root=source_root,
            dest_root=dest_root,
            checkpoint_base=checkpoint_base,
        )
    if name.startswith("evals-"):
        family = name[len("evals-") :]
        if family not in EVAL_PREFIXES:
            raise SystemExit(f"Unknown chunk: {name}")
        return eval_transfers(_discover_eval_hits(ssh, source_host), family, dest_root)
    if name.startswith("rollouts-"):
        env, _, exp_text = name[len("rollouts-") :].rpartition("-exp")
        if env not in {"coinrun", "maze", "heist"} or not exp_text.isdigit():
            raise SystemExit(f"Unknown chunk: {name}")
        exp_id = int(exp_text)
        if exp_id not in EXP_IDS:
            raise SystemExit(f"Unknown chunk: {name}")
        relpath = rollout_relpath(env, exp_id)
        transfers = _sized_path_transfers(
            ssh,
            source_host,
            [relpath],
            source_root=source_root,
            dest_root=dest_root,
            is_dir=True,
        )
        if not transfers:
            raise SystemExit(f"Missing rollout directory: {relpath}")
        return _expand_large_dirs(ssh, source_host, transfers)
    raise SystemExit(f"Unknown chunk: {name}")


def _expand_large_dirs(
    ssh: SshFunc, source_host: str, transfers: Sequence[Transfer]
) -> list[Transfer]:
    """Replace a directory larger than the budget with its files."""
    expanded = []
    for transfer in transfers:
        if not transfer.is_dir or transfer.nbytes <= CHUNK_BUDGET_BYTES:
            expanded.append(transfer)
            continue
        for nbytes, relpath in _remote_files_under(ssh, source_host, transfer.source):
            expanded.append(
                Transfer(
                    source=str(Path(transfer.source) / relpath),
                    dest=str(Path(transfer.dest) / relpath),
                    nbytes=nbytes,
                    label=f"{transfer.label}/{relpath}",
                )
            )
    return expanded


def chunk_parts(
    name: str, ssh: SshFunc, *, source_host: str = SOURCE_HOST
) -> list[list[Transfer]]:
    """Resolve a chunk and pack it into budget-sized parts."""
    transfers = resolve_chunk(name, ssh, source_host=source_host)
    if not transfers:
        raise SystemExit(f"Chunk {name} resolved to no files")
    return group_by_budget(transfers)


def rsync_source_and_dest(transfer: Transfer) -> tuple[str, str]:
    """Return rsync endpoints. Directories use trailing slashes so reruns do not nest."""
    if transfer.is_dir:
        return transfer.source.rstrip("/") + "/", transfer.dest.rstrip("/") + "/"
    return transfer.source, transfer.dest


def copy_part(
    transfers: Sequence[Transfer],
    *,
    source_host: str,
    dest_host: str,
    via: str,
    execute: bool,
) -> None:
    """Copy one part. Without ``execute`` this only prints the commands."""
    if via not in {"dest", "local"}:
        raise SystemExit(f"Unknown --via {via}")
    print(f"{len(transfers)} item(s), {sum(item.nbytes for item in transfers)} bytes")
    for transfer in transfers:
        source, dest = rsync_source_and_dest(transfer)
        print(f"  {source_host}:{source}")
        print(f"    -> {dest_host}:{dest}")
    if not execute:
        print("Dry run; pass --execute to copy.")
        return
    if via == "dest":
        _copy_via_dest(transfers, source_host=source_host, dest_host=dest_host)
    else:
        _copy_via_local(transfers, source_host=source_host, dest_host=dest_host)


def _copy_via_dest(
    transfers: Sequence[Transfer], *, source_host: str, dest_host: str
) -> None:
    script_lines = ["set -euo pipefail"]
    for transfer in transfers:
        source, dest = rsync_source_and_dest(transfer)
        mkdir_target = dest.rstrip("/") if transfer.is_dir else str(Path(dest).parent)
        script_lines.append(f"mkdir -p {shlex.quote(mkdir_target)}")
        script_lines.append(
            "rsync -a --partial "
            f"{shlex.quote(source_host + ':' + source)} "
            f"{shlex.quote(dest)}"
        )
    result = run_ssh(
        ssh_argv(dest_host, "bash -s"),
        input="\n".join(script_lines) + "\n",
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def _copy_via_local(
    transfers: Sequence[Transfer], *, source_host: str, dest_host: str
) -> None:
    """Stage one transfer locally, then upload it; rsync needs one local endpoint."""
    for transfer in transfers:
        source, dest = rsync_source_and_dest(transfer)
        mkdir_target = dest.rstrip("/") if transfer.is_dir else str(Path(dest).parent)
        _checked(ssh_argv(dest_host, f"mkdir -p {shlex.quote(mkdir_target)}"))
        with TemporaryDirectory(prefix="migrate-to-snoopy-") as staging_dir:
            staged = Path(staging_dir) / "payload"
            if transfer.is_dir:
                staged.mkdir()
                staging_endpoint = str(staged) + "/"
            else:
                staging_endpoint = str(staged)
            _checked(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    rsync_rsh(),
                    f"{source_host}:{source}",
                    staging_endpoint,
                ]
            )
            _checked(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    rsync_rsh(),
                    staging_endpoint,
                    f"{dest_host}:{dest}",
                ]
            )


def _print_chunk_list() -> None:
    print(
        "Destination: "
        f"{DEST_HOST}:{data_root('snoopy')}\n"
        "Nothing is copied unless you pass run CHUNK --execute.\n"
        f"plan CHUNK splits anything over {CHUNK_BUDGET_BYTES} bytes "
        "into parts. Run one part at a time with --part N.\n"
        "Run --help for the source trees, the disk requirement, and the "
        "SSH host names.\n"
    )
    for name in logical_chunk_names():
        print(f"{name}: {chunk_description(name)}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("list", "plan", "run"))
    parser.add_argument("chunk", nargs="?")
    parser.add_argument("--part", type=int, help="1-based part when a chunk is split")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy files. Without this flag, run only prints the plan.",
    )
    parser.add_argument("--source-host", default=SOURCE_HOST)
    parser.add_argument("--dest-host", default=DEST_HOST)
    parser.add_argument(
        "--via",
        choices=("local", "dest"),
        default="local",
        help=(
            "local: stage one transfer at a time in local temporary storage "
            "(default). dest: snoopy pulls from the source host directly, which "
            "requires that host name to resolve on snoopy."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "list":
        _print_chunk_list()
        return 0
    if args.chunk is None:
        raise SystemExit(f"{args.command} requires a chunk name. See list.")
    if args.chunk not in logical_chunk_names():
        raise SystemExit(f"Unknown chunk: {args.chunk}")
    if args.execute and args.command != "run":
        raise SystemExit("--execute is only valid with run")

    parts = chunk_parts(args.chunk, ssh_text, source_host=args.source_host)
    print(f"{args.chunk}: {len(parts)} part(s)")
    if args.part is None and not args.execute:
        for index, part in enumerate(parts, start=1):
            print(f"\n# part {index}/{len(parts)}")
            copy_part(
                part,
                source_host=args.source_host,
                dest_host=args.dest_host,
                via=args.via,
                execute=False,
            )
        return 0
    if args.part is None:
        if len(parts) != 1:
            raise SystemExit(
                f"{args.chunk} has {len(parts)} parts; pass --part 1..{len(parts)}"
            )
        part_index = 1
    elif args.part < 1 or args.part > len(parts):
        raise SystemExit(
            f"--part {args.part} is outside 1..{len(parts)} for {args.chunk}"
        )
    else:
        part_index = args.part

    print(f"\n# part {part_index}/{len(parts)}")
    copy_part(
        parts[part_index - 1],
        source_host=args.source_host,
        dest_host=args.dest_host,
        via=args.via,
        execute=args.execute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
