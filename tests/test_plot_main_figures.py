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
    assert panels["heist"]["prefix"] == ["tmlr-heist03-expert400"]
    assert panels["coinrun"]["auc_tex"] == "auc-coinrun.tex"
    assert panels["maze"]["auc_tex"] == "auc-maze.tex"
    assert panels["heist"]["auc_tex"] == "auc-kandc.tex"
    assert panels["coinrun"]["show_ylabel"] is True
    assert panels["maze"]["show_ylabel"] is False
    assert panels["heist"]["show_ylabel"] is False
    assert plot_main_figures.PANEL_FIGSIZE[1] > 4.5
    assert plot_main_figures.PANEL_AXIS_LABEL_SIZE > 18
    assert plot_main_figures.SHARED_METHOD_FILTER == ["ensemble", "wait"]
    assert r"\vspace{-1.0em}" in plot_main_figures.LATEX_SNIPPET
    assert "heist-main-normalized.pdf" in plot_main_figures.LATEX_SNIPPET
    assert r"\label{fig:results:kandc}" in plot_main_figures.LATEX_SNIPPET
    assert r"\label{fig:results:heist}" not in plot_main_figures.LATEX_SNIPPET
    assert r"\texttt{K\&C}" in plot_main_figures.LATEX_SNIPPET
    assert r"\textsc{Keys\&Chests}" not in plot_main_figures.LATEX_SNIPPET
    assert r"\texttt{Heist}" not in plot_main_figures.LATEX_SNIPPET
    assert "main-legend.pdf" in plot_main_figures.LATEX_SNIPPET
    assert r"\begin{wraptable}" in plot_main_figures.LATEX_TABLE_SNIPPET
    assert r"\input{tables/auc-kandc}" in plot_main_figures.LATEX_TABLE_SNIPPET
    assert r"\label{tab:auc_kandc}" in plot_main_figures.LATEX_TABLE_SNIPPET
    assert r"\kandc" in plot_main_figures.LATEX_TABLE_SNIPPET
