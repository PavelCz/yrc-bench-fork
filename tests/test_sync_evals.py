import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_evals.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_sync(tmp_path: Path, *args: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rsync_log = tmp_path / "rsync.log"

    _write_executable(
        bin_dir / "ssh",
        """#!/usr/bin/env bash
printf '%s\\n' "video-test_heist_exp0"
""",
    )
    _write_executable(
        bin_dir / "rsync",
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${SYNC_TEST_RSYNC_LOG}"
""",
    )

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SYNC_TEST_RSYNC_LOG": str(rsync_log),
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--with-videos", *args, "video-test"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, rsync_log.read_text() if rsync_log.exists() else ""


def test_sync_accepts_custom_source_base(tmp_path):
    source_base = (
        "rnn:/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork/experiments/evals"
    )
    result, rsync_args = _run_sync(tmp_path, "--source-base", source_base)

    assert result.returncode == 0, result.stderr
    assert f"{source_base}/video-test_heist_exp0" in rsync_args


def test_sync_accepts_equals_form_for_custom_source_base(tmp_path):
    source_base = "other-host:/custom/evals"
    result, rsync_args = _run_sync(tmp_path, f"--source-base={source_base}")

    assert result.returncode == 0, result.stderr
    assert f"{source_base}/video-test_heist_exp0" in rsync_args


def test_sync_uses_canonical_data_root_by_default(tmp_path):
    result, rsync_args = _run_sync(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (
        "rnn:/nas/ucb/czempin/data/goal-misgen/experiments/evals/video-test_heist_exp0"
    ) in rsync_args
