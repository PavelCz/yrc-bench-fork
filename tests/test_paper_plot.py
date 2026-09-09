import importlib

import matplotlib
import pytest


def ignore_backend_selection(*args, **kwargs):
    del args, kwargs


matplotlib.use = ignore_backend_selection
paper_plot = importlib.import_module("analyzing.paper_plot")


@pytest.mark.parametrize(
    "dir_name, expected",
    [
        (
            "proxy-penalty_coinrun_proxy_penalty_exp0",
            ("proxy-penalty", "coinrun_proxy_penalty", 0),
        ),
        (
            "proxy-penalty_maze_proxy_penalty_exp0",
            ("proxy-penalty", "maze_proxy_penalty", 0),
        ),
        (
            "proxy-penalty_heist_proxy_penalty_exp0",
            ("proxy-penalty", "heist_proxy_penalty", 0),
        ),
    ],
)
def test_parse_experiment_dir_supports_proxy_penalty(dir_name, expected):
    assert paper_plot.parse_experiment_dir(dir_name) == expected


def test_parse_robust_experiment_dir_supports_maze_proxy_penalty():
    parsed = paper_plot.parse_robust_experiment_dir(
        "proxy-penalty_robust400_maze_proxy_penalty_exp0"
    )

    assert parsed == ("proxy-penalty", "robust400", "maze_proxy_penalty", 0)


@pytest.mark.parametrize(
    "dir_name, expected",
    [
        (
            "coinrun_proxy_penalty_max-prob_exp0",
            ("coinrun_proxy_penalty", "max-prob", 0),
        ),
        (
            "maze_proxy_penalty_max-prob_robust400_exp0",
            ("maze_proxy_penalty", "max-prob_robust400", 0),
        ),
        (
            "heist_proxy_penalty_max-prob_exp0",
            ("heist_proxy_penalty", "max-prob", 0),
        ),
    ],
)
def test_parse_method_dir_supports_proxy_penalty(dir_name, expected):
    assert paper_plot.parse_method_dir(dir_name) == expected


def test_extract_icml_results_finds_proxy_penalty_campaign(tmp_path):
    coin_npz = (
        tmp_path
        / "proxy-penalty_coinrun_proxy_penalty_exp0"
        / "coinrun_proxy_penalty_max-prob_exp0"
        / "20260807_164626"
        / "eval_seed_1_test.npz"
    )
    maze_npz = (
        tmp_path
        / "proxy-penalty_robust400_maze_proxy_penalty_exp0"
        / "maze_proxy_penalty_max-prob_robust400_exp0"
        / "20260807_170047"
        / "eval_seed_1_test.npz"
    )
    coin_npz.parent.mkdir(parents=True)
    maze_npz.parent.mkdir(parents=True)
    coin_npz.write_bytes(b"npz")
    maze_npz.write_bytes(b"npz")

    coin = paper_plot.extract_icml_results(
        tmp_path,
        prefix_filter=["proxy-penalty"],
        env_filter="coinrun_proxy_penalty",
    )
    maze = paper_plot.extract_icml_results(
        tmp_path,
        prefix_filter=["proxy-penalty"],
        env_filter="maze_proxy_penalty",
    )

    assert coin["max-prob"][0] == coin_npz
    assert maze["max-prob_robust400"][0] == maze_npz


def test_shared_legend_splits_regular_methods_from_special_block():
    regular, special = paper_plot.build_shared_legend_entries(paper_mode=True)

    assert [label for _, label in regular] == [
        r"\textsc{Heuristic}",
        r"\textsc{Ensemble}",
        r"\textsc{MaxProb}",
        r"\textsc{MaxLogit}",
        r"\textsc{ImageSVDD}",
        r"\textsc{LatentSVDD}",
    ]
    assert [label for _, label in special] == [
        r"\textsc{PartialOracle}",
        r"\textsc{Novice}",
        r"\textsc{Expert}",
        r"\textsc{Random}",
    ]
    assert paper_plot.SHARED_LEGEND_METHODS == [
        *paper_plot.DEFAULT_METHOD_ORDER,
        "oracle-lb-random",
    ]
