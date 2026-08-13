from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_evals.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_sync_evals_accepts_custom_source_base(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"

    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
printf 'ssh %s\n' "$*" >> "$SYNC_TEST_LOG"
printf '%s\n' \
  campaign_heist_exp0 \
  campaign_heist_exp1 \
  campaign_heist_exp2 \
  campaign_heist_exp3
""",
    )
    _write_executable(
        fake_bin / "rsync",
        """#!/usr/bin/env bash
printf 'rsync %s\n' "$*" >> "$SYNC_TEST_LOG"
""",
    )

    result = subprocess.run(
        [
            "bash",
            str(SYNC_SCRIPT),
            "--source-base",
            "rnn:/custom/repo/experiments/evals",
            "campaign",
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SYNC_TEST_LOG": str(command_log),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text()
    assert "ls -1 /custom/repo/experiments/evals" in commands
    assert (
        "rnn:/custom/repo/experiments/evals/campaign_heist_exp0" in commands
    )
    assert commands.count("rsync ") == 4


def test_sync_evals_rejects_source_base_without_host(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SYNC_SCRIPT),
            "--source-base",
            "/custom/repo/experiments/evals",
            "campaign",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "expected HOST:PATH" in result.stderr
