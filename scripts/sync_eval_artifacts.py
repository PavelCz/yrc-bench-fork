#!/usr/bin/env python3
"""Sync only the artifacts needed by scripts/run_eval.py between machines."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from common import (
    ENSEMBLE_METHODS,
    METHOD_CONFIGS,
    ROBUST_MAZE_CHECKPOINT_STEPS,
    SERVER_PATHS,
    SVDD_METHODS,
    get_env_folder,
)


ARTIFACT_ENVS = {
    "coinrun_proxy_fail": "coinrun",
    "maze_proxy_fail": "maze",
}
EVAL_ENVS = ["maze", "coinrun", "coinrun_proxy_fail", "maze_proxy_fail"]
DEFAULT_SVDD_PREFIX = "neurips04"
DEFAULT_NUM_ENSEMBLE_MEMBERS = 4
EXPECTED_TIMESTEPS = 200015872


@dataclass(frozen=True)
class SyncItem:
    source: Path
    target: Path
    label: str


class RemotePathResolver:
    def __init__(self, host: str, ssh_options: list[str]) -> None:
        self.host = host
        self.ssh_options = ssh_options

    def _run(self, remote_command: str) -> subprocess.CompletedProcess[str]:
        command = ["ssh", *self.ssh_options, self.host, remote_command]
        return subprocess.run(command, text=True, capture_output=True)

    def check_reachable(self) -> bool:
        result = self._run("true")
        if result.returncode == 0:
            print(f"Reachability check passed: {self.host}")
            return True

        print(f"Could not reach source host: {self.host}")
        if result.stderr:
            print(result.stderr.strip())
        return False

    def exists(self, path: Path) -> bool:
        result = self._run(f"test -e {shlex.quote(str(path))}")
        return result.returncode == 0

    def list_child_dirs(self, parent: Path) -> list[str]:
        quoted_parent = shlex.quote(str(parent))
        result = self._run(
            f"if [ -d {quoted_parent} ]; then "
            f"find {quoted_parent} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n'; "
            "fi"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to list remote directory {self.host}:{parent}\n"
                f"{result.stderr.strip()}"
            )
        return [line for line in result.stdout.splitlines() if line]

    def list_model_files(self, parent: Path) -> list[str]:
        quoted_parent = shlex.quote(str(parent))
        result = self._run(
            f"if [ -d {quoted_parent} ]; then "
            f"find {quoted_parent} -mindepth 1 -maxdepth 1 -type f "
            "-name 'model_*.pth' -printf '%f\\n'; "
            "fi"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to list remote model files {self.host}:{parent}\n"
                f"{result.stderr.strip()}"
            )
        return [line for line in result.stdout.splitlines() if line]

    def newest_timestamp_dir(self, parent: Path) -> Optional[Path]:
        timestamp_dirs = [
            name for name in self.list_child_dirs(parent) if "__seed_" in name
        ]
        if not timestamp_dirs:
            return None
        if len(timestamp_dirs) > 1:
            print(f"Warning: multiple timestamp dirs in {self.host}:{parent}")
            for name in sorted(timestamp_dirs):
                print(f"  - {name}")
        return parent / sorted(timestamp_dirs)[-1]

    def best_model_checkpoint(self, ts_dir: Path) -> Optional[Path]:
        model_files: list[tuple[int, str]] = []
        for filename in self.list_model_files(ts_dir):
            match = re.fullmatch(r"model_(\d+)\.pth", filename)
            if match:
                model_files.append((int(match.group(1)), filename))
        if not model_files:
            return None
        timesteps, filename = sorted(model_files)[-1]
        if timesteps != EXPECTED_TIMESTEPS:
            print(
                f"Warning: {self.host}:{ts_dir} has max timesteps {timesteps}, "
                f"expected {EXPECTED_TIMESTEPS}"
            )
        return ts_dir / filename


def get_svdd_feature_type(method: str) -> str:
    if method == "svdd-image":
        return "image"
    if method == "svdd-latent":
        return "latent"
    raise ValueError(f"Unsupported SVDD method: {method}")


def map_path(source_path: Path, source_base: Path, target_base: Path) -> Path:
    return target_base / source_path.relative_to(source_base)


def add_if_exists(
    items: list[SyncItem],
    resolver: RemotePathResolver,
    source: Path,
    target: Path,
    label: str,
) -> None:
    if resolver.exists(source):
        items.append(SyncItem(source=source, target=target, label=label))
    else:
        print(f"Warning: missing {label}: {resolver.host}:{source}")


def resolve_acting_checkpoint(
    resolver: RemotePathResolver,
    checkpoint_base: Path,
    env: str,
    exp_id: int,
    pct: int,
) -> Optional[Path]:
    env_folder = get_env_folder(env)
    parent = checkpoint_base / env_folder / f"icml2_{env}_exp{exp_id}_{pct}p"
    ts_dir = resolver.newest_timestamp_dir(parent)
    if ts_dir is None:
        print(f"Warning: missing timestamp dir: {resolver.host}:{parent}")
        return None
    model = resolver.best_model_checkpoint(ts_dir)
    if model is None:
        print(f"Warning: missing model_*.pth under {resolver.host}:{ts_dir}")
    return model


def resolve_ensemble_member(
    resolver: RemotePathResolver,
    checkpoint_base: Path,
    env: str,
    exp_id: int,
    member_id: int,
) -> Optional[Path]:
    env_folder = get_env_folder(env)
    parent = (
        checkpoint_base
        / env_folder
        / "ensembles"
        / f"icml2_ensemble_{env}_exp{exp_id}_m{member_id}"
    )
    ts_dir = resolver.newest_timestamp_dir(parent)
    if ts_dir is None:
        print(f"Warning: missing ensemble timestamp dir: {resolver.host}:{parent}")
        return None
    model = resolver.best_model_checkpoint(ts_dir)
    if model is None:
        print(f"Warning: missing ensemble model_*.pth under {resolver.host}:{ts_dir}")
    return model


def build_sync_items(args: argparse.Namespace) -> list[SyncItem]:
    source_paths = SERVER_PATHS[args.source_server]
    target_paths = SERVER_PATHS[args.target_server]
    source_checkpoint_base = Path(source_paths["checkpoint_base"])
    target_checkpoint_base = Path(target_paths["checkpoint_base"])
    source_policy_base = source_checkpoint_base.parent
    target_policy_base = target_checkpoint_base.parent
    source_seeds_base = Path(source_paths["seeds_base"])
    target_seeds_base = Path(target_paths["seeds_base"])
    source_svdd_base = Path(source_paths["svdd_base"])
    target_svdd_base = Path(target_paths["svdd_base"])

    resolver = RemotePathResolver(args.source_host, args.ssh_option)
    if not resolver.check_reachable():
        raise RuntimeError(
            "Source host reachability check failed; fix SSH config or pass "
            "--source-host before retrying."
        )

    artifact_env = ARTIFACT_ENVS.get(args.env, args.env)
    robust_checkpoint_steps = None
    if args.robust200:
        robust_checkpoint_steps = ROBUST_MAZE_CHECKPOINT_STEPS["robust200"]
    elif args.robust400:
        robust_checkpoint_steps = ROBUST_MAZE_CHECKPOINT_STEPS["robust400"]

    if robust_checkpoint_steps is not None and artifact_env != "maze":
        raise ValueError("Robust maze checkpoints are only valid for maze artifacts.")

    items: list[SyncItem] = []
    for exp_id in args.exp_ids:
        weak_model = resolve_acting_checkpoint(
            resolver, source_checkpoint_base, artifact_env, exp_id, 0
        )
        if weak_model is not None:
            target = map_path(
                weak_model, source_checkpoint_base, target_checkpoint_base
            )
            items.append(SyncItem(weak_model, target, f"exp{exp_id} weak/sim"))

        if robust_checkpoint_steps is None:
            strong_model = resolve_acting_checkpoint(
                resolver, source_checkpoint_base, artifact_env, exp_id, 50
            )
            if strong_model is not None:
                target = map_path(
                    strong_model, source_checkpoint_base, target_checkpoint_base
                )
                items.append(SyncItem(strong_model, target, f"exp{exp_id} strong"))
        else:
            robust_parent = (
                source_policy_base
                / "neurips"
                / "maze_afh_random_start"
                / f"icml2_maze_exp{exp_id}_50p_random_start"
            )
            robust_ts_dir = resolver.newest_timestamp_dir(robust_parent)
            if robust_ts_dir is None:
                print(
                    f"Warning: missing robust timestamp dir: {resolver.host}:{robust_parent}"
                )
            else:
                source = robust_ts_dir / f"model_{robust_checkpoint_steps}.pth"
                target = map_path(source, source_policy_base, target_policy_base)
                add_if_exists(
                    items,
                    resolver,
                    source,
                    target,
                    f"exp{exp_id} robust strong {robust_checkpoint_steps}",
                )

        seed_source = source_seeds_base / f"{exp_id}.json"
        seed_target = map_path(seed_source, source_seeds_base, target_seeds_base)
        add_if_exists(
            items, resolver, seed_source, seed_target, f"exp{exp_id} level seeds"
        )

        for method in args.methods:
            if method in SVDD_METHODS:
                feature_type = get_svdd_feature_type(method)
                source = (
                    source_svdd_base
                    / args.svdd_prefix
                    / f"svdd_{artifact_env}_{feature_type}_exp{exp_id}"
                    / "trained.joblib"
                )
                target = map_path(source, source_svdd_base, target_svdd_base)
                add_if_exists(
                    items, resolver, source, target, f"exp{exp_id} {method} model"
                )

            if method in ENSEMBLE_METHODS:
                for member_id in range(args.num_ensemble_members):
                    member_model = resolve_ensemble_member(
                        resolver,
                        source_checkpoint_base,
                        artifact_env,
                        exp_id,
                        member_id,
                    )
                    if member_model is None:
                        continue
                    target = map_path(
                        member_model, source_checkpoint_base, target_checkpoint_base
                    )
                    items.append(
                        SyncItem(
                            member_model,
                            target,
                            f"exp{exp_id} {method} member m{member_id}",
                        )
                    )

    unique: dict[Path, SyncItem] = {}
    for item in items:
        unique[item.target] = item
    return list(unique.values())


def run_rsync(
    item: SyncItem,
    *,
    source_host: str,
    ssh_options: list[str],
    dry_run: bool,
) -> bool:
    command = [
        "rsync",
        "-av",
        "-e",
        "ssh " + " ".join(shlex.quote(option) for option in ssh_options),
        f"{source_host}:{shlex.quote(str(item.source))}",
        str(item.target),
    ]
    if dry_run:
        print("$ " + " ".join(shlex.quote(part) for part in command))
        return True

    item.target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print(f"Failed to sync {item.label}: {item.source}")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the checkpoint, seed, SVDD, and ensemble artifacts needed by "
            "scripts/run_eval.py from one machine layout to another."
        )
    )
    parser.add_argument("--env", required=True, choices=EVAL_ENVS)
    parser.add_argument(
        "--methods",
        "--method",
        nargs="+",
        required=True,
        choices=sorted(METHOD_CONFIGS.keys()),
        help="One or more run_eval.py methods to prepare.",
    )
    parser.add_argument("--exp-ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--source-server",
        default="chai",
        choices=sorted(SERVER_PATHS.keys()),
        help="Source path layout from scripts/common.py.",
    )
    parser.add_argument(
        "--target-server",
        default="carc",
        choices=sorted(SERVER_PATHS.keys()),
        help="Destination path layout from scripts/common.py.",
    )
    parser.add_argument(
        "--source-host",
        default="chai",
        help="SSH host that exposes the source-server paths.",
    )
    parser.add_argument(
        "--svdd-prefix",
        default=DEFAULT_SVDD_PREFIX,
        help=f"SVDD training prefix to copy when needed (default: {DEFAULT_SVDD_PREFIX}).",
    )
    parser.add_argument(
        "--num-ensemble-members",
        type=int,
        default=DEFAULT_NUM_ENSEMBLE_MEMBERS,
    )
    robust_group = parser.add_mutually_exclusive_group()
    robust_group.add_argument("--robust200", "--robust-200", action="store_true")
    robust_group.add_argument("--robust400", "--robust-400", action="store_true")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=["-o", "ConnectTimeout=10"],
        help="Extra ssh option. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print rsync commands without copying.",
    )
    return parser.parse_args()


def print_plan(items: Iterable[SyncItem], source_host: str) -> None:
    items = list(items)
    print(f"Resolved {len(items)} artifact file(s):")
    for item in items:
        print(f"  [{item.label}]")
        print(f"    {source_host}:{item.source}")
        print(f"    -> {item.target}")


def main() -> int:
    args = parse_args()
    items = build_sync_items(args)
    print_plan(items, args.source_host)
    if not items:
        return 1

    failed = 0
    for item in items:
        if not run_rsync(
            item,
            source_host=args.source_host,
            ssh_options=args.ssh_option,
            dry_run=args.dry_run,
        ):
            failed += 1

    if failed:
        print(f"{failed} sync(s) failed.")
        return 1
    if args.dry_run:
        print("Dry run complete; no files copied.")
    else:
        print("All requested artifacts synced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
