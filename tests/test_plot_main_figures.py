import importlib


plot_main_figures = importlib.import_module("analyzing.plot_main_figures")


def test_main_figure_script_bakes_in_the_three_paper_campaigns():
    panels = {panel["env"]: panel for panel in plot_main_figures.PANELS}

    assert panels["coinrun"]["prefix"] == [
        "tmlr2-svdd",
        "tmlr3-svdd",
        "tmlr4-svdd",
        "imcl04",
        "icml04",
        "tmlr-oracle-lb",
    ]
    assert panels["maze"]["prefix"] == [
        "tmlr-robust-maze-1",
        "tmlr-robust-maze-3",
        "tmlr-oracle-lb-robust400",
        "tmlr-maze-svdd-fix",
    ]
    assert panels["maze"]["robust_filter"] == "robust"
    assert panels["heist"]["prefix"] == ["tmlr-heist02-metrics"]
    assert plot_main_figures.SHARED_METHOD_FILTER == ["ensemble", "wait"]
    assert "heist-main-normalized.pdf" in plot_main_figures.LATEX_SNIPPET
    assert "main-legend.pdf" in plot_main_figures.LATEX_SNIPPET
