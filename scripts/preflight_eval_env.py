#!/usr/bin/env python3
"""Preflight checks for Procgen AFHP evaluation jobs.

Run this inside the same conda environment used by SLURM, for example:

    conda run -n ood-stable python scripts/preflight_eval_env.py \
        --env coinrun_proxy_fail
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ENVS = ["coinrun", "coinrun_proxy_fail", "maze", "maze_proxy_fail"]
ENV_CHOICES = [*SUPPORTED_ENVS, "all"]
LOCAL_PACKAGES = {
    "acs": REPO_ROOT / "lib" / "acs",
    "procgen": REPO_ROOT / "lib" / "procgen",
}


class Preflight:
    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self.failures: list[str] = []

    def run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - exercised manually
            self.failures.append(name)
            print(f"[FAIL] {name}: {exc}")
            if self.verbose:
                traceback.print_exc()
        else:
            print(f"[ OK ] {name}")

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)


def module_path(module: ModuleType) -> Path | None:
    file_name = getattr(module, "__file__", None)
    return Path(file_name).resolve() if file_name else None


def import_module(name: str) -> ModuleType:
    module = importlib.import_module(name)
    path = module_path(module)
    print(f"       {name}: {path if path is not None else '<no __file__>'}")
    return module


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def check_local_package(preflight: Preflight, package_name: str) -> None:
    module = import_module(package_name)
    path = module_path(module)
    expected_root = LOCAL_PACKAGES[package_name]

    preflight.require(
        path is not None,
        f"{package_name} has no __file__; cannot verify whether it is local",
    )
    preflight.require(
        is_relative_to(path, expected_root),
        (
            f"{package_name} is imported from {path}, expected it under "
            f"{expected_root}. Reinstall with: pip install -e {expected_root}"
        ),
    )


def check_eval_imports() -> None:
    import_module("flags")
    import_module("numpy")
    import_module("torch")
    import_module("pytorch_lightning")
    import_module("wandb")
    import_module("acs.types")
    import_module("acs.wait_policy_sampler")
    import_module("YRC.coverage.coverage_search")
    import_module("eval_afhp")


def check_procgen_env(env_name: str) -> None:
    procgen = import_module("procgen")
    procgen_env = getattr(procgen, "ProcgenEnv")
    env = procgen_env(
        num_envs=1,
        env_name=env_name,
        num_levels=1,
        start_level=0,
        distribution_mode="hard",
        random_percent=100 if env_name.endswith("_proxy_fail") else 0,
    )
    try:
        env.reset()
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the conda environment can run Procgen AFHP evals."
    )
    parser.add_argument(
        "--env",
        choices=ENV_CHOICES,
        default="coinrun_proxy_fail",
        help=(
            "Procgen environment to instantiate, or 'all' "
            "(default: coinrun_proxy_fail)"
        ),
    )
    parser.add_argument(
        "--skip-local-path-check",
        action="store_true",
        help="Do not require acs/procgen to import from this repo's lib/ directory.",
    )
    parser.add_argument(
        "--skip-procgen-env",
        action="store_true",
        help="Skip constructing a ProcgenEnv instance.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tracebacks for failing checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preflight = Preflight(verbose=args.verbose)

    print(f"Python: {sys.executable}")
    print(f"Repo:   {REPO_ROOT}")
    print()

    if not args.skip_local_path_check:
        preflight.run(
            "acs imports from repo-local lib/acs",
            lambda: check_local_package(preflight, "acs"),
        )
        preflight.run(
            "procgen imports from repo-local lib/procgen",
            lambda: check_local_package(preflight, "procgen"),
        )

    preflight.run("eval_afhp dependency imports", check_eval_imports)

    if not args.skip_procgen_env:
        env_names = SUPPORTED_ENVS if args.env == "all" else [args.env]
        for env_name in env_names:
            preflight.run(
                f"ProcgenEnv can construct {env_name}",
                lambda env_name=env_name: check_procgen_env(env_name),
            )

    if preflight.failures:
        print()
        print("Preflight failed:")
        for failure in preflight.failures:
            print(f"  - {failure}")
        return 1

    print()
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
