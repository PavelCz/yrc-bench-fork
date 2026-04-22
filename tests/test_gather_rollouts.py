from types import SimpleNamespace

from gather_rollouts import get_gather_agent_path


def make_config(collect_data_agent="weak", skyline=0):
    return SimpleNamespace(
        general=SimpleNamespace(skyline=skyline),
        coord_policy=SimpleNamespace(collect_data_agent=collect_data_agent),
        agents=SimpleNamespace(
            sim_weak="sim_weak.pth",
            weak="weak.pth",
            strong="strong.pth",
        ),
    )


def test_gather_agent_path_matches_train_split_mapping_for_default_training():
    assert get_gather_agent_path(make_config("weak")) == ("sim_weak.pth", "weak")
    assert get_gather_agent_path(make_config("strong")) == ("weak.pth", "strong")


def test_gather_agent_path_matches_skyline_mapping():
    assert get_gather_agent_path(make_config("weak", skyline=1)) == (
        "weak.pth",
        "weak",
    )
    assert get_gather_agent_path(make_config("strong", skyline=1)) == (
        "strong.pth",
        "strong",
    )
