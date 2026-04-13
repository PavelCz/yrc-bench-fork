# CLI Conventions

This repository uses two different CLI patterns on purpose.

## Script CLIs

Standalone Python scripts should use `tyro` with a typed dataclass CLI.

Use this pattern for scripts that primarily orchestrate jobs, analysis, or
filesystem work, such as:

- `scripts/run_eval.py`
- `scripts/train_svdd.py`
- `scripts/run_eval_strong_on_help.py`
- module-style analysis entrypoints under `analyzing/`

Conventions for script CLIs:

- Define a top-level dataclass named like `<ScriptName>Args`.
- Use `tyro.cli(<ScriptName>Args)` in a small `parse_args()` helper.
- Prefer long-form flag names derived from snake_case field names.
- Validate dynamic choices after parsing against shared repo constants instead of
  duplicating those constants in the CLI type definitions.
- Keep `main()` small and use helpers to hide orchestration details.

## App Entry Points

Config-driven application entrypoints should continue to use `flags.py` and
`jsonargparse`.

Use this pattern for entrypoints that merge CLI overrides into hierarchical YAML
configs and runtime config objects, such as:

- `python -m apps.train`
- `eval.py`
- `python -m apps.calibrate_afhp`
- `python -m apps.eval_afhp_bin`

Conventions for app entrypoints:

- Keep nested dotted overrides in `flags.py`.
- Do not introduce `tyro` directly into config-driven entrypoints unless the
  config loading path is being redesigned as part of a larger migration.

## Migration Rule

When touching an existing standalone script CLI, prefer converting it to `tyro`
instead of adding more `argparse` boilerplate.
