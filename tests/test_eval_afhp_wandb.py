from types import SimpleNamespace

import eval_afhp


def test_eval_afhp_skips_eval_wandb_logger_when_wandb_disabled(monkeypatch):
    def fail_init(*args, **kwargs):
        raise AssertionError("wandb init should not be called")

    monkeypatch.setattr(eval_afhp, "init_eval_wandb_run", fail_init)

    exp, logger = eval_afhp.create_eval_wandb_logger(SimpleNamespace(use_wandb=False))

    assert exp is None
    assert logger is None


def test_eval_afhp_continues_when_eval_wandb_init_fails(monkeypatch):
    monkeypatch.setattr(eval_afhp, "init_eval_wandb_run", lambda *args, **kwargs: None)

    exp, logger = eval_afhp.create_eval_wandb_logger(
        SimpleNamespace(use_wandb=True, exp_name="run")
    )

    assert exp is None
    assert logger is None
