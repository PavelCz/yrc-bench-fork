import importlib


plot_unnormalized_figures = importlib.import_module(
    "analyzing.plot_unnormalized_figures"
)


def test_unnormalized_script_bakes_in_main_and_proxy_fail_rows():
    main = {panel["env"]: panel for panel in plot_unnormalized_figures.MAIN_PANELS}
    proxy_fail = {
        panel["env"]: panel for panel in plot_unnormalized_figures.PROXY_FAIL_PANELS
    }

    assert main["coinrun"]["prefix"] == [
        "tmlr2-svdd",
        "tmlr3-svdd",
        "tmlr4-svdd",
        "imcl04",
        "icml04",
        "tmlr-oracle-lb",
    ]
    assert main["maze"]["prefix"] == [
        "tmlr-robust-maze-1",
        "tmlr-robust-maze-3",
        "tmlr-oracle-lb-robust400",
        "tmlr-maze-svdd-fix",
    ]
    assert main["maze"]["robust_filter"] == "robust"
    assert main["heist"]["prefix"] == ["tmlr-heist03-expert400"]
    assert main["heist"]["placeholder"] is False

    assert proxy_fail["coinrun_proxy_fail"]["prefix"] == [
        "tmlr-proxy-fail1",
        "tmlr-proxy-fail2",
        "tmlr-proxy-fail3",
        "tmlr-proxy-fail4",
        "tmlr-oracle-lb-proxy_fail",
    ]
    assert proxy_fail["maze_proxy_fail"]["prefix"] == [
        "tmlr-proxy-fail4",
        "tmlr-oracle-lb-proxy_fail",
        "tmlr-proxy-fail-fix",
    ]
    assert proxy_fail["maze_proxy_fail"]["robust_filter"] == "robust"
    assert proxy_fail["heist_proxy_fail"]["placeholder"] is True
    assert proxy_fail["heist_proxy_fail"]["save_name"] == "heist-proxy-fail.pdf"

    assert plot_unnormalized_figures.COINRUN_MAZE_YLIM == (6.0, 10.0)
    assert main["coinrun"]["ylim"] == plot_unnormalized_figures.COINRUN_MAZE_YLIM
    assert main["maze"]["ylim"] == plot_unnormalized_figures.COINRUN_MAZE_YLIM
    assert main["heist"]["ylim"] is None
    assert proxy_fail["coinrun_proxy_fail"]["ylim"] == (
        plot_unnormalized_figures.COINRUN_MAZE_YLIM
    )
    assert proxy_fail["maze_proxy_fail"]["ylim"] == (
        plot_unnormalized_figures.COINRUN_MAZE_YLIM
    )
    assert proxy_fail["heist_proxy_fail"]["ylim"] is None
    assert plot_unnormalized_figures.SHARED_METHOD_FILTER == ["ensemble", "wait"]
    assert "heist-main.pdf" in plot_unnormalized_figures.LATEX_SNIPPET
    assert "heist-proxy-fail.pdf" in plot_unnormalized_figures.LATEX_SNIPPET
    assert r"\label{fig:app:kandc-unnormalized}" in (
        plot_unnormalized_figures.LATEX_SNIPPET
    )
    assert "forthcoming" in plot_unnormalized_figures.LATEX_SNIPPET
    assert r"$[6, 10]$" in plot_unnormalized_figures.LATEX_SNIPPET
    assert r"$[0, 10]$" not in plot_unnormalized_figures.LATEX_SNIPPET
