"""Tests for the RNN-to-snoopy migration planner."""

import importlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

common = importlib.import_module("common")
migrate = importlib.import_module("migrate_to_snoopy")


def test_ssh_reuses_one_connection_per_host() -> None:
    command = migrate.ssh_argv("rnn", "echo ok")
    assert command[-2] == "rnn"
    assert command[-1] == "echo ok"
    assert "ControlMaster=auto" in command
    assert "ControlPath=/tmp/migrate-to-snoopy-%r@%h:%p" in command
    assert "ControlMaster=auto" in migrate.rsync_rsh()


def test_list_does_not_contact_a_server(capsys) -> None:
    assert migrate.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "rollouts-heist-exp7" in output
    assert "ssh" not in output
    assert "/scr/pavel/data/goal-misgen" in output
    assert "tmlr-maze-svdd-fix/maze/gather_maze_exp0" in output
    assert "neurips03/maze/gather_maze_exp4" in output
    assert "trained_svdd/neurips05" in output


def test_documentation_covers_sources_disk_and_hosts() -> None:
    doc = migrate.__doc__ or ""
    for phrase in (
        "yrc-bench-afhp-range-retry",
        "tmlr-maze-svdd-fix-model",
        "tmlr-heist01",
        "policy/icml",
        "robust400",
        "188 GB",
        "67 GB",
        "SEEK_HOLE",
        "snoopy1.usc.edu",
        "model_400031744.pth",
    ):
        assert phrase in doc


def test_eval_prefixes_match_figure_scripts() -> None:
    from analyzing.plot_main_figures import PANELS as main_panels
    from analyzing.plot_proxy_fail_figures import PANELS as fail_panels
    from analyzing.plot_proxy_penalty_figures import PANELS as penalty_panels

    def prefixes(panels):
        found = set()
        for panel in panels:
            found.update(panel["prefix"])
        return found

    assert prefixes(main_panels) == set(migrate.RECOVERABLE_PREFIXES)
    assert prefixes(fail_panels) == set(migrate.PROXY_FAIL_PREFIXES)
    assert prefixes(penalty_panels) == set(migrate.PROXY_PENALTY_PREFIXES)


def test_rollout_paths_follow_the_svdd_training_logs() -> None:
    assert migrate.rollout_relpath("maze", 0).endswith(
        "tmlr-maze-svdd-fix/maze/gather_maze_exp0"
    )
    assert migrate.rollout_relpath("maze", 3).endswith(
        "tmlr-maze-svdd-fix/maze/gather_maze_exp3"
    )
    assert migrate.rollout_relpath("maze", 4).endswith(
        "neurips03/maze/gather_maze_exp4"
    )
    assert "neurips03/maze/gather_maze_exp0" not in migrate.rollout_relpath("maze", 0)
    assert migrate.rollout_relpath("coinrun", 7).endswith(
        "neurips03/coinrun/gather_coinrun_exp7"
    )
    assert migrate.rollout_relpath("heist", 7).endswith(
        "tmlr-heist01/heist/gather_heist_exp7"
    )


def test_svdd_paths_skip_superseded_neurips05_maze_models() -> None:
    relpaths = migrate.svdd_relpaths()
    assert any(
        path.startswith("trained_svdd/neurips05/svdd_coinrun_") for path in relpaths
    )
    assert any("tmlr-maze-svdd-fix-model" in path for path in relpaths)
    assert not any("neurips05/svdd_maze_" in path for path in relpaths)
    assert not any(path.endswith("exp8") or "_exp8" in path for path in relpaths)


def test_classify_npz_keeps_current_prefixes_and_drops_superseded_ones() -> None:
    current = (
        "tmlr-heist03-expert400-proxy-fail_heist_proxy_fail_exp5/"
        "heist_proxy_fail_max-logit_exp5/20261001_172812/results.npz"
    )
    assert migrate.classify_npz(current)[:3] == (
        "tmlr-heist03-expert400-proxy-fail",
        "heist_proxy_fail",
        "max-logit",
    )
    superseded = (
        "tmlr-proxy-fail5_robust400_maze_proxy_fail_exp0/"
        "maze_proxy_fail_max-prob_exp0/20240101_000000/results.npz"
    )
    assert migrate.classify_npz(superseded) is None
    old_penalty = (
        "tmlr-heist03-expert400-proxy-penalty_heist_proxy_penalty_exp0/"
        "heist_proxy_penalty_max-prob_exp0/20240101_000000/results.npz"
    )
    assert migrate.classify_npz(old_penalty) is None
    assert migrate.classify_npz(current.replace("exp5", "exp8")) is None


def test_select_winning_npz_keeps_the_newer_curve() -> None:
    older = migrate.NpzHit(
        source_root="/data/evals",
        relpath="imcl04_coinrun_exp0/coinrun_max_prob_exp0/20200101_000000/a.npz",
        prefix="imcl04",
        env="coinrun",
        method="max-prob",
        exp_id=0,
        run_name="20200101_000000",
        nbytes=10,
        source_priority=0,
    )
    newer = migrate.NpzHit(
        source_root="/fork/evals",
        relpath="tmlr5-svdd_coinrun_exp0/coinrun_max-prob_exp0/20240101_000000/b.npz",
        prefix="tmlr5-svdd",
        env="coinrun",
        method="max-prob",
        exp_id=0,
        run_name="20240101_000000",
        nbytes=12,
        source_priority=2,
    )
    other_method = migrate.NpzHit(
        source_root="/fork/evals",
        relpath="tmlr-oracle-lb_coinrun_exp0/coinrun_oracle-lb-random_exp0/20210101_000000/c.npz",
        prefix="tmlr-oracle-lb",
        env="coinrun",
        method="oracle-lb-random",
        exp_id=0,
        run_name="20210101_000000",
        nbytes=8,
        source_priority=2,
    )
    chosen = migrate.select_winning_npz([older, newer, other_method])
    assert {hit.relpath for hit in chosen} == {newer.relpath, other_method.relpath}


def test_equal_timestamp_prefers_the_higher_source_priority() -> None:
    data_hit = migrate.NpzHit(
        source_root="/data/evals",
        relpath="same/coinrun_max-prob_exp1/20240101_000000/a.npz",
        prefix="imcl04",
        env="coinrun",
        method="max-prob",
        exp_id=1,
        run_name="20240101_000000",
        nbytes=1,
        source_priority=0,
    )
    fork_hit = migrate.NpzHit(
        source_root="/fork/evals",
        relpath="same/coinrun_max-prob_exp1/20240101_000000/b.npz",
        prefix="imcl04",
        env="coinrun",
        method="max-prob",
        exp_id=1,
        run_name="20240101_000000",
        nbytes=1,
        source_priority=2,
    )
    chosen = migrate.select_winning_npz([data_hit, fork_hit])
    assert chosen == [fork_hit]


def test_choose_model_file_uses_newest_timestamp_and_highest_step() -> None:
    chosen = migrate.choose_model_file(
        [
            ("2020__seed_0", "model_200015872.pth", 5, "/old/model_200015872.pth"),
            ("2024__seed_0", "model_100.pth", 1, "/new/model_100.pth"),
            ("2024__seed_0", "model_200015872.pth", 9, "/new/model_200015872.pth"),
        ]
    )
    assert chosen is not None
    assert chosen[3] == "/new/model_200015872.pth"


def test_checkpoint_transfer_prefers_policy_tree_and_falls_back_to_train_logs() -> None:
    source_root = Path("/nas/ucb/czempin/data/goal-misgen")
    dest_root = Path("/scr/pavel/data/goal-misgen")
    checkpoint_base = str(source_root / "policy" / "icml")
    policy_run = migrate.get_ensemble_run_dir("coinrun", 1, 0, checkpoint_base)
    policy_model = policy_run / "2024__seed_1" / "model_200015872.pth"
    train_run = migrate.train_log_ensemble_dir("maze", 1, 2)
    train_model = train_run / "2024__seed_1" / "model_200015872.pth"
    transfers = migrate.checkpoint_transfers(
        [(4, str(policy_model))],
        [(4, str(train_model))],
        source_root=source_root,
        dest_root=dest_root,
        checkpoint_base=checkpoint_base,
        exp_ids=(1,),
    )
    by_source = {transfer.source: transfer for transfer in transfers}
    assert by_source[str(policy_model)].dest.endswith(
        "policy/icml/coinrun/ensembles/icml2_ensemble_coinrun_exp1_m0/"
        "2024__seed_1/model_200015872.pth"
    )
    maze_dest = by_source[str(train_model)].dest
    assert maze_dest.endswith(
        "policy/icml/maze_afh/ensembles/icml2_ensemble_maze_exp1_m2/"
        "2024__seed_1/model_200015872.pth"
    )
    assert "/logs/train/" not in maze_dest


def test_robust_maze_run_dir_matches_common(tmp_path: Path) -> None:
    checkpoint_base = tmp_path / "policy" / "icml"
    parent = migrate.robust_maze_run_dir(str(checkpoint_base), 3)
    timestamp = parent / "2024__seed_3"
    timestamp.mkdir(parents=True)
    (timestamp / "model_400031744.pth").write_text("x")
    found = common.get_robust_maze_strong_checkpoint(3, str(checkpoint_base), 400031744)
    assert Path(found).parent == timestamp


def test_group_by_budget_splits_large_chunks_and_keeps_oversized_files() -> None:
    small = [
        migrate.Transfer("a", "b", 600, "a"),
        migrate.Transfer("c", "d", 600, "c"),
        migrate.Transfer("e", "f", 600, "e"),
    ]
    parts = migrate.group_by_budget(small, budget=1000)
    assert [len(part) for part in parts] == [1, 1, 1]

    packed = migrate.group_by_budget(
        [
            migrate.Transfer("a", "b", 400, "a"),
            migrate.Transfer("c", "d", 400, "c"),
            migrate.Transfer("e", "f", 400, "e"),
        ],
        budget=1000,
    )
    assert [len(part) for part in packed] == [2, 1]

    huge = [migrate.Transfer("big", "dest", 5_000, "big", is_dir=True)]
    assert migrate.group_by_budget(huge, budget=1000) == [huge]


def test_directory_rsync_uses_trailing_slashes() -> None:
    source, dest = migrate.rsync_source_and_dest(
        migrate.Transfer("/src/gather", "/dst/gather", 10, "gather", is_dir=True)
    )
    assert source == "/src/gather/"
    assert dest == "/dst/gather/"


@pytest.mark.parametrize("variant", ["robust200", "robust400"])
@pytest.mark.parametrize("method_suffix", [False, True])
def test_parent_robust_variant_keeps_curves_distinct(variant, method_suffix) -> None:
    method = f"max_prob_{variant}" if method_suffix else "max_prob"
    paths = [
        "icml04_maze_exp0/maze_max_prob_exp0/20261002_000000/a.npz",
        f"icml04_{variant}_maze_exp0/maze_{method}_exp0/20261001_000000/a.npz",
    ]
    hits = []
    for path in paths:
        classified = migrate.classify_npz(path)
        assert classified is not None
        prefix, env, method_name, exp_id, run_name = classified
        hits.append(
            migrate.NpzHit(
                "/src", path, prefix, env, method_name, exp_id, run_name, 1, 0
            )
        )
    assert {hit.method for hit in hits} == {"max-prob", f"max-prob_{variant}"}
    assert migrate.select_winning_npz(hits) == hits


@pytest.mark.parametrize("is_dir", [False, True])
def test_local_relay_copies_contents_and_can_be_rerun(tmp_path, monkeypatch, is_dir):
    if shutil.which("rsync") is None:
        pytest.skip("rsync is required for the local relay integration test")
    source = tmp_path / "source with spaces"
    dest = tmp_path / "destination with spaces" / "renamed"
    if is_dir:
        (source / "nested").mkdir(parents=True)
        payload = source / "nested" / "payload"
        copied = dest / "nested" / "payload"
    else:
        payload = source
        copied = dest
    payload.write_text("original payload")
    real_run = subprocess.run
    staged_paths = []

    def run_with_local_hosts(command, **kwargs):
        # Keep real rsync semantics while replacing SSH hosts with local paths.
        if command[0] == "ssh":
            assert command[-2] == "destination"
            return real_run(shlex.split(command[-1]), **kwargs)
        assert command[0] == "rsync"
        endpoints = command[-2:]
        remote = [":" in endpoint for endpoint in endpoints]
        assert sum(remote) == 1, "rsync cannot copy between two remote endpoints"
        if remote[0]:
            assert endpoints[0].startswith("source:")
            staged_paths.append(Path(endpoints[1]))
        else:
            assert endpoints[1].startswith("destination:")
        local_endpoints = [endpoint.split(":", 1)[-1] for endpoint in endpoints]
        return real_run(command[:-2] + local_endpoints, **kwargs)

    monkeypatch.setattr(migrate.subprocess, "run", run_with_local_hosts)
    transfer = migrate.Transfer(str(source), str(dest), 100, "test", is_dir)
    for contents in ("original payload", "updated payload with a different size"):
        payload.write_text(contents)
        migrate.copy_part(
            [transfer],
            source_host="source",
            dest_host="destination",
            via="local",
            execute=True,
        )
        assert copied.read_text() == contents
    if is_dir:
        assert list(dest.rglob("payload")) == [copied]
    assert len(staged_paths) == 2
    assert all(not path.exists() for path in staged_paths)


@pytest.mark.parametrize("failing_leg", [1, 2])
def test_local_relay_propagates_failure_and_cleans_staging(monkeypatch, failing_leg):
    calls = []
    staged_paths = []

    def fail_transfer(command, **kwargs):
        if command[0] == "ssh":
            return subprocess.CompletedProcess(command, 0)
        calls.append(command)
        if len(calls) == 1:
            staged_paths.append(Path(command[-1]))
        if len(calls) == failing_leg:
            raise subprocess.CalledProcessError(23, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(migrate.subprocess, "run", fail_transfer)
    with pytest.raises(subprocess.CalledProcessError):
        migrate.copy_part(
            [migrate.Transfer("/src/file", "/dst/file", 1, "test")],
            source_host="source",
            dest_host="destination",
            via="local",
            execute=True,
        )
    assert len(calls) == failing_leg
    assert all(not path.parent.exists() for path in staged_paths)
