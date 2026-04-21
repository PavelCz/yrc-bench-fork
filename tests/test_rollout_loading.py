import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from YRC.core.utils import load_rollouts_from_file


def test_load_rollouts_from_specific_file():
    with TemporaryDirectory() as tmp_dir:
        rollout_dir = Path(tmp_dir)
        rollout_file = rollout_dir / "rollouts_3levels.pt"
        config_file = rollout_dir / "rollouts_config_3levels.json"

        torch.save([torch.tensor([1.0]), torch.tensor([2.0])], rollout_file)
        config_file.write_text(json.dumps({"name": "dummy"}))

        rollout_obs = load_rollouts_from_file(rollout_file)

    assert [tensor.item() for tensor in rollout_obs] == [1.0, 2.0]
