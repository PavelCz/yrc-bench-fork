import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_evals.sh"
PROXY_PENALTY_SOURCE = (
    "rnn:/nas/ucb/czempin/code/goal-misgen/yrc-bench-fork/experiments/evals"
)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_sync(tmp_path: Path, remote_dirs, *args: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rsync_log = tmp_path / "rsync.log"
    quoted_dirs = " ".join(f'"{name}"' for name in remote_dirs)

    _write_executable(
        bin_dir / "ssh",
        f"""#!/usr/bin/env bash
printf '%s\\n' {quoted_dirs}
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
        ["bash", str(SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, rsync_log.read_text() if rsync_log.exists() else ""


def test_sync_accepts_custom_source_base(tmp_path):
    source_base = PROXY_PENALTY_SOURCE
    result, rsync_args = _run_sync(
        tmp_path,
        ["video-test_heist_exp0"],
        "--with-videos",
        "--source-base",
        source_base,
        "video-test",
    )

    assert result.returncode == 0, result.stderr
    assert f"{source_base}/video-test_heist_exp0" in rsync_args


def test_sync_accepts_equals_form_for_custom_source_base(tmp_path):
    source_base = "other-host:/custom/evals"
    result, rsync_args = _run_sync(
        tmp_path,
        ["video-test_heist_exp0"],
        "--with-videos",
        f"--source-base={source_base}",
        "video-test",
    )

    assert result.returncode == 0, result.stderr
    assert f"{source_base}/video-test_heist_exp0" in rsync_args


def test_sync_uses_canonical_data_root_by_default(tmp_path):
    result, rsync_args = _run_sync(
        tmp_path,
        ["video-test_heist_exp0"],
        "--with-videos",
        "video-test",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "rnn:/nas/ucb/czempin/data/goal-misgen/experiments/evals/video-test_heist_exp0"
    ) in rsync_args


def test_sync_evals_rejects_source_base_without_host(tmp_path: Path) -> None:
    result, _ = _run_sync(
        tmp_path,
        ["campaign_heist_exp0"],
        "--source-base",
        "/custom/repo/experiments/evals",
        "campaign",
    )

    assert result.returncode == 1
    assert "expected HOST:PATH" in result.stderr


def test_sync_proxy_penalty_campaign_dirs(tmp_path):
    remote_dirs = [
        "proxy-penalty_coinrun_proxy_penalty_exp0",
        "proxy-penalty_robust400_maze_proxy_penalty_exp0",
    ]
    result, rsync_args = _run_sync(
        tmp_path,
        remote_dirs,
        "--source-base",
        PROXY_PENALTY_SOURCE,
        "proxy-penalty",
    )

    assert result.returncode == 0, result.stderr
    for name in remote_dirs:
        assert f"{PROXY_PENALTY_SOURCE}/{name}" in rsync_args
