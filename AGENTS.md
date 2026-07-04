# Repository Guidelines

## Project Structure & Module Organization
CRISP-Real2Sim is a shell-driven Python research pipeline. Top-level orchestration lives in `run_crisp_video.sh` and `scripts/`, where numbered scripts `1-8` run video extraction, masks, reconstruction, alignment, fitting, and postprocessing. Environment and demo helpers live in `setups/`. Data preparation and external model components are under `prep/`, with details in `prep/README.md`. RL transfer, training, evaluation, and replay code live in `MotionTracking/`. Visualization code is under `vis_scripts/`, and static project assets are in `assets/`. Runtime compatibility shims are in `runtime_shims/`.

## Build, Test, and Development Commands
- `bash setups/setup_crisp.sh`: create the main `crisp` environment.
- `conda activate crisp`: activate the main pipeline environment before running CRISP scripts.
- `bash setups/validate_crisp_video_env.sh`: smoke-check the CRISP video environment.
- `bash setups/run_demo.sh`: run the packaged demo flow.
- `bash run_crisp_video.sh /abs/path/to/data/demo`: run the full pipeline; pass the path without the `_videos` or `_img` suffix.
- `bash scripts/8_postprocessing.sh smoke gv`: regenerate `post_scene` outputs and MotionTracking bridge artifacts for the demo.
- `bash setups/setup_crisp_rl.sh && conda activate crisp_rl`: prepare the RL environment, then work from `MotionTracking/`.

## Coding Style & Naming Conventions
Use Python 3 style with 4-space indentation and descriptive snake_case names for Python modules, functions, and script variables. Keep shell scripts Bash-compatible, quote path variables, and prefer explicit absolute paths in examples. Do not reformat vendored or third-party trees under `prep/`, `vis_scripts/viser*`, or `MotionTracking/smpllib` unless the change is scoped there. Where subprojects define tools, follow them locally: `prep/UFM` uses Black/isort, and `vis_scripts/viser*` uses Ruff.

## Testing Guidelines
There is no single top-level test suite. Validate changes with the smallest relevant command: environment checks via `setups/validate_*.sh`, demo execution via `setups/run_demo.sh`, or targeted subproject tests such as `pytest vis_scripts/viser/tests` when editing that package. For pipeline changes, document the sequence name, command, and output directory checked, especially `results/output/scene/` and `results/output/post_scene/`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Update readme.md`, `Add optional NKSR integration`, and `Clarify post_scene z-up description`. Keep commits focused on one behavior or documentation change. Pull requests should include a concise purpose, affected pipeline stage or directory, exact validation commands run, expected data/assets required, and screenshots or output paths for visualization changes.

## Security & Configuration Tips
Do not commit downloaded checkpoints, SMPL/SMPL-X body models, generated `data/`, `results/`, or machine-specific conda paths. Keep credentials and dataset links out of scripts unless they are public release links already documented in the README.
