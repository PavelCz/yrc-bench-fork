import importlib


plot_proxy_fail_figures = importlib.import_module("analyzing.plot_proxy_fail_figures")


def test_proxy_fail_script_bakes_in_coinrun_maze_and_kandc_placeholder():
    panels = {panel["env"]: panel for panel in plot_proxy_fail_figures.PANELS}

    assert panels["coinrun_proxy_fail"]["prefix"] == [
        "tmlr-proxy-fail1",
        "tmlr-proxy-fail2",
        "tmlr-proxy-fail3",
        "tmlr-proxy-fail4",
        "tmlr-oracle-lb-proxy_fail",
    ]
    assert panels["maze_proxy_fail"]["prefix"] == [
        "tmlr-proxy-fail4",
        "tmlr-oracle-lb-proxy_fail",
        "tmlr-proxy-fail-fix",
    ]
    assert panels["maze_proxy_fail"]["robust_filter"] == "robust"
    assert panels["heist_proxy_fail"]["placeholder"] is True
    assert panels["coinrun_proxy_fail"]["auc_tex"] == "auc-coinrun-proxy-fail.tex"
    assert panels["maze_proxy_fail"]["auc_tex"] == "auc-maze-proxy-fail.tex"
    assert panels["heist_proxy_fail"]["auc_tex"] == "auc-kandc-proxy-fail.tex"
    assert panels["coinrun_proxy_fail"]["save_name"] == (
        "coinrun-proxy-fail-normalized.pdf"
    )
    assert panels["maze_proxy_fail"]["save_name"] == ("maze-proxy-fail-normalized.pdf")
    assert panels["heist_proxy_fail"]["save_name"] == (
        "heist-proxy-fail-normalized.pdf"
    )
    assert panels["coinrun_proxy_fail"]["show_ylabel"] is True
    assert panels["maze_proxy_fail"]["show_ylabel"] is False
    assert plot_proxy_fail_figures.SHARED_METHOD_FILTER == ["ensemble", "wait"]
    assert "proxy-fail-legend.pdf" in plot_proxy_fail_figures.LATEX_SNIPPET
    assert r"\label{fig:results:kandc-proxy-fail}" in (
        plot_proxy_fail_figures.LATEX_SNIPPET
    )
    assert "forthcoming" in plot_proxy_fail_figures.LATEX_SNIPPET
    assert r"\input{tables/auc-kandc-proxy-fail}" in (
        plot_proxy_fail_figures.LATEX_TABLE_SNIPPET
    )
