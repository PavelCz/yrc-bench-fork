import importlib.util
from pathlib import Path

import numpy as np

from YRC.envs.procgen.heist_metrics import extract_heist_episode_data


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "collect_ood_failure_videos.py"


def load_collector_module():
    spec = importlib.util.spec_from_file_location(
        "collect_ood_failure_videos", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_collector_module()


def test_is_ood_timeout_keeps_only_incomplete_ood_episodes():
    assert collector.is_ood_timeout(is_ood=True, level_complete=False)
    assert not collector.is_ood_timeout(is_ood=True, level_complete=True)
    assert not collector.is_ood_timeout(is_ood=False, level_complete=False)
    assert not collector.is_ood_timeout(is_ood=False, level_complete=True)


def test_filename_includes_heist_counters():
    filename = collector.ood_timeout_video_filename(
        3,
        level_seed=12345,
        keys_collected=10,
        num_keys=10,
        chests_opened=4,
        total_chests=5,
        episode_return=4.0,
    )
    assert filename == "ood_timeout_003_seed12345_keys10-10_chests4-5_ret4.00"


def test_keep_rule_uses_heist_terminal_info():
    timeout_info = {
        "prev_level/keys_collected": 10,
        "prev_level/num_keys": 10,
        "prev_level/total_chests": 5,
        "prev_level/chests_opened": 4,
        "prev_level/total_steps": 1000,
        "prev_level_complete": 0,
        "prev_level/randomize_goal": True,
    }
    extracted = extract_heist_episode_data(timeout_info)
    is_ood = collector.episode_randomize_goal(timeout_info, True, False)

    assert extracted["level_complete"] is False
    assert collector.is_ood_timeout(
        is_ood=is_ood, level_complete=extracted["level_complete"]
    )


def test_save_ood_timeout_video_writes_gif(tmp_path):
    frames = []
    for _ in range(3):
        frames.append(
            {
                "obs": np.zeros((3, 8, 8), dtype=np.float32),
                "human_obs": np.zeros((16, 16, 3), dtype=np.uint8),
            }
        )
    path = collector.save_ood_timeout_video(
        frames,
        output_folder=tmp_path,
        filename="ood_timeout_000_seed1_keys2-4_chests1-2_ret1.00",
        caption="test caption",
        include_human_view=True,
    )
    assert path.exists()
    assert path.suffix == ".gif"
    assert (tmp_path / f"{path.stem}_caption.txt").read_text() == "test caption"
