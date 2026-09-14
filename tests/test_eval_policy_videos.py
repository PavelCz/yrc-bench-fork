import importlib
from types import SimpleNamespace

import numpy as np
from PIL import Image


eval_policy = importlib.import_module("eval_policy")


def test_copy_human_frame_copies_rgb():
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    copied = eval_policy._copy_human_frame({"rgb": rgb})
    assert copied is not None
    assert np.array_equal(copied, rgb)
    rgb[0, 0, 0] = 99
    assert copied[0, 0, 0] != 99
    assert eval_policy._copy_human_frame({}) is None


def test_save_videos_writes_human_stills(tmp_path):
    human = np.full((16, 16, 3), 40, dtype=np.uint8)
    human[4:12, 4:12] = 200
    frames = [
        {
            "obs": np.zeros((3, 8, 8), dtype=np.float32),
            "human_obs": None,
        },
        {
            "obs": np.ones((3, 8, 8), dtype=np.float32),
            "human_obs": human,
        },
    ]
    config = SimpleNamespace(
        evaluation=SimpleNamespace(
            video_logging_mode="folder",
            video_output_folder=None,
            include_human_view=True,
        )
    )

    eval_policy.save_videos(
        [
            {
                "frames": frames,
                "return": 2.0,
                "episode_idx": 0,
                "is_ood": True,
                "level_seed": 108033,
            }
        ],
        config,
        tmp_path,
        "test",
        None,
    )

    still = tmp_path / "videos" / "test" / "human" / "ood_seed108033_ep000_start.png"
    gif = tmp_path / "videos" / "test" / "episode_000_ood_seed108033_return_2.00.gif"
    assert still.exists()
    assert gif.exists()
    assert np.array_equal(np.array(Image.open(still)), human)
