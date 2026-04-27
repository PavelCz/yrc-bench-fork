import json

from analyzing.policy_eval_plot import (
    extract_policy_eval_results,
    parse_experiment_dir,
)


def test_parse_experiment_dir_supports_coinrun_proxy_fail():
    parsed = parse_experiment_dir("study_coinrun_proxy_fail_strong_exp2")

    assert parsed == ("study", "coinrun_proxy_fail", "strong", 2)


def test_extract_policy_eval_results_filters_coinrun_proxy_fail(tmp_path):
    result_dir = (
        tmp_path / "study_coinrun_proxy_fail_weak_exp0" / "max_prob" / "20260426_120000"
    )
    result_dir.mkdir(parents=True)
    (result_dir / "policy_eval_results.json").write_text(
        json.dumps({"all_returns": [0.0], "level_ood_gt": [True]})
    )

    results, metadata = extract_policy_eval_results(
        tmp_path,
        prefix_filter=["study"],
        env_filter="coinrun_proxy_fail",
        agent_filter=["weak"],
    )

    assert list(results) == ["weak"]
    assert results["weak"][0] == result_dir / "policy_eval_results.json"
    assert metadata["weak"] == {
        "prefix": "study",
        "env": "coinrun_proxy_fail",
        "agent": "weak",
    }
