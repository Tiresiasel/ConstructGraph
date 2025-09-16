# Prompt Lab (Experimental)

Prompt Lab provides a sandbox to iterate on extraction prompts and analyze outputs.

## Structure
- `inputs/` — Raw snippets or PDF-derived segments for focused prompting.
- `configs/` — Run configurations (model, temperature, prompt template, batch size, post-processing flags).
- `logs/` — Execution logs by model/version.
- `outputs/` — Per-run artifacts (JSON + text): `pdf_text.txt`, `prompt_template.txt`, `user_prompt_final.txt`, `response_raw.json`, `output.json`.
- `reports/` — Aggregated analysis such as precision/recall by construct or relationship type.
- `run_prompt_lab.py` — Entrypoint to execute a run based on a config.
- `analyze_outputs.py` — Utilities for aggregations and basic metrics.

## Quick start
```bash
python src/prompt_lab/run_prompt_lab.py \
  --config src/prompt_lab/configs/example_config.json
```

## Suggested workflow improvements
- Adopt a declarative pipeline (LangChain Runnable, LiteLLM Tasks, or Pydantic workflows):
  1) load → 2) chunk → 3) prompt → 4) parse → 5) validate → 6) write artifacts.
- Persist run metadata (config hash, code version, environment) with outputs for reproducibility.
- Add CLI wrappers: `prompt-lab run`, `prompt-lab analyze`.
- Add tests for prompt template rendering and structured parsing.

## Conventions
- Use `lower_snake_case` for config keys.
- Use ISO timestamps for run folders.
- Keep comments/docs in English.
