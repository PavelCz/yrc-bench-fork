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


def _auc_triplet(median: float, lower: float, upper: float):
    return (median, lower, upper)


def test_format_auc_tabular_matches_paper_coinrun_table():
    tex = paper_plot.format_auc_tabular(
        {
            "oracle-lb-random": _auc_triplet(0.739, 0.722, 0.761),
            "ts-random": _auc_triplet(0.701, 0.679, 0.727),
            "ensemble-single": _auc_triplet(0.488, 0.467, 0.511),
            "max-prob": _auc_triplet(0.612, 0.596, 0.636),
            "max-logit": _auc_triplet(0.537, 0.487, 0.591),
            "svdd-image": _auc_triplet(0.526, 0.495, 0.568),
            "svdd-latent": _auc_triplet(0.513, 0.444, 0.578),
        },
        normalize_by_range=False,
    )

    assert tex == (
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "Method & AUC (Median [IQR]) \\\\\n"
        "\\midrule\n"
        "\\textbf{\\textsc{PartialOracle}} & "
        "\\textbf{0.739 [0.722, 0.761]} \\\\\n"
        "\\cmidrule(lr){1-2}\n"
        "\\textbf{\\textsc{Heuristic}} & "
        "\\textbf{0.701 [0.679, 0.727]} \\\\\n"
        "\\textsc{Ensemble} & 0.488 [0.467, 0.511] \\\\\n"
        "\\textsc{MaxProb} & 0.612 [0.596, 0.636] \\\\\n"
        "\\textsc{MaxLogit} & 0.537 [0.487, 0.591] \\\\\n"
        "\\textsc{ImageSVDD} & 0.526 [0.495, 0.568] \\\\\n"
        "\\textsc{LatentSVDD} & 0.513 [0.444, 0.578] \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    assert "\\begin{table}" not in tex


def test_format_auc_tabular_does_not_bold_partial_oracle_when_heuristic_wins():
    tex = paper_plot.format_auc_tabular(
        {
            "oracle-lb-random": _auc_triplet(0.743, 0.739, 0.748),
            "ts-random": _auc_triplet(0.787, 0.781, 0.791),
            "ensemble-single": _auc_triplet(0.735, 0.727, 0.744),
            "max-prob": _auc_triplet(0.627, 0.607, 0.638),
            "max-logit": _auc_triplet(0.679, 0.658, 0.699),
            "svdd-image": _auc_triplet(0.532, 0.520, 0.545),
            "svdd-latent": _auc_triplet(0.566, 0.558, 0.575),
        },
        normalize_by_range=False,
    )

    assert "\\textsc{PartialOracle} & 0.743 [0.739, 0.748] \\\\" in tex
    assert "\\cmidrule(lr){1-2}" in tex
    assert "\\textbf{\\textsc{PartialOracle}}" not in tex
    assert "\\textbf{\\textsc{Heuristic}} & \\textbf{0.787 [0.781, 0.791]} \\\\" in tex
