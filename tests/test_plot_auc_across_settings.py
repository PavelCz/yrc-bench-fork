import importlib


plot_auc_across_settings = importlib.import_module("analyzing.plot_auc_across_settings")


def test_parse_auc_table_medians_reads_bold_and_plain_rows(tmp_path):
    tex = tmp_path / "auc.tex"
    tex.write_text(
        r"""
\begin{tabular}{ll}
\toprule
Method & AUC (Median [IQR]) \\
\midrule
\textbf{\textsc{PartialOracle}} & \textbf{0.747 [0.731, 0.769]} \\
\cmidrule(lr){1-2}
\textbf{\textsc{Heuristic}} & \textbf{0.710 [0.688, 0.736]} \\
\textsc{Ensemble} & 0.498 [0.478, 0.521] \\
\textsc{MaxProb} & -- \\
\bottomrule
\end{tabular}
"""
    )

    medians = plot_auc_across_settings.parse_auc_table_medians(tex)
    assert medians["oracle-lb-random"] == 0.747
    assert medians["ts-random"] == 0.710
    assert medians["ensemble-single"] == 0.498
    assert "max-prob" not in medians


def test_auc_across_settings_script_bakes_in_three_envs_and_settings():
    assert plot_auc_across_settings.SETTINGS == (
        "recoverable",
        "penalty",
        "termination",
    )
    assert plot_auc_across_settings.SETTING_LABELS == (
        "Recoverable",
        "Penalty",
        "Termination",
    )
    tables = [panel["tables"] for panel in plot_auc_across_settings.PANELS]
    assert tables[0]["recoverable"] == "auc-coinrun.tex"
    assert tables[0]["penalty"] == "auc-coinrun-proxy-penalty.tex"
    assert tables[0]["termination"] == "auc-coinrun-proxy-fail.tex"
    assert tables[1]["recoverable"] == "auc-maze.tex"
    assert tables[2]["recoverable"] == "auc-kandc.tex"
    assert tables[2]["penalty"] == "auc-kandc-proxy-penalty.tex"
    assert tables[2]["termination"] == "auc-kandc-proxy-fail.tex"
    assert plot_auc_across_settings.SAVE_NAME == "auc-across-settings.pdf"
    assert "auc-across-settings.pdf" in plot_auc_across_settings.LATEX_SNIPPET
    assert r"\label{fig:app:auc-across-settings}" in (
        plot_auc_across_settings.LATEX_SNIPPET
    )
    assert plot_auc_across_settings.PANELS[0]["show_ylabel"] is True
    assert plot_auc_across_settings.PANELS[1]["show_ylabel"] is False
    assert plot_auc_across_settings.PANELS[2]["show_ylabel"] is False
