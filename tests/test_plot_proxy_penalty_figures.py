import importlib


plot_proxy_penalty_figures = importlib.import_module(
    "analyzing.plot_proxy_penalty_figures"
)


def test_proxy_penalty_script_bakes_in_coinrun_maze_and_kandc_placeholder():
    panels = {panel["env"]: panel for panel in plot_proxy_penalty_figures.PANELS}

    assert panels["coinrun_proxy_penalty"]["prefix"] == ["proxy-penalty01"]
    assert panels["maze_proxy_penalty"]["prefix"] == ["proxy-penalty01"]
    assert panels["maze_proxy_penalty"]["robust_filter"] == "robust"
    assert panels["heist_proxy_penalty"]["placeholder"] is True
    assert panels["coinrun_proxy_penalty"]["placeholder"] is False
    assert panels["maze_proxy_penalty"]["placeholder"] is False
    assert panels["coinrun_proxy_penalty"]["auc_tex"] == (
        "auc-coinrun-proxy-penalty.tex"
    )
    assert panels["maze_proxy_penalty"]["auc_tex"] == "auc-maze-proxy-penalty.tex"
    assert panels["heist_proxy_penalty"]["auc_tex"] == "auc-kandc-proxy-penalty.tex"
    assert panels["coinrun_proxy_penalty"]["save_name"] == (
        "coinrun-proxy-penalty-normalized.pdf"
    )
    assert panels["maze_proxy_penalty"]["save_name"] == (
        "maze-proxy-penalty-normalized.pdf"
    )
    assert panels["heist_proxy_penalty"]["save_name"] == (
        "heist-proxy-penalty-normalized.pdf"
    )
    assert panels["coinrun_proxy_penalty"]["show_ylabel"] is True
    assert panels["maze_proxy_penalty"]["show_ylabel"] is False
    assert panels["heist_proxy_penalty"]["show_ylabel"] is False
    assert plot_proxy_penalty_figures.SHARED_METHOD_FILTER == ["ensemble", "wait"]
    assert r"\label{fig:app:kandc-proxy-penalty}" in (
        plot_proxy_penalty_figures.LATEX_SNIPPET
    )
    assert "forthcoming" in plot_proxy_penalty_figures.LATEX_SNIPPET
    assert r"\texttt{K\&C}" in plot_proxy_penalty_figures.LATEX_SNIPPET
    assert "proxy-penalty-legend.pdf" in plot_proxy_penalty_figures.LATEX_SNIPPET
    assert r"\input{tables/auc-kandc-proxy-penalty}" in (
        plot_proxy_penalty_figures.LATEX_TABLE_SNIPPET
    )
    assert r"\textsc{PartialOracle} & -- \\" in (
        plot_proxy_penalty_figures.PLACEHOLDER_AUC_TABULAR
    )
