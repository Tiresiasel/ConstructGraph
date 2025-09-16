#!/usr/bin/env bash
set -euo pipefail

# Max parallel jobs (default 3, cap at 8)
MAX_PROCS="${MAX_PROCS:-3}"
if [[ "$MAX_PROCS" =~ ^[0-9]+$ ]]; then
	if (( MAX_PROCS > 8 )); then MAX_PROCS=8; fi
else
	MAX_PROCS=8
fi

# Resolve project root from this script's location
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAMPLES_DIR="$ROOT/scripts/prompt_lab/samples"
PROMPT_PATH="$ROOT/scripts/prompt_lab/prompts/blueprint_prompt.txt"
SYSTEM_PROMPT="You are a research assistant skilled in knowledge extraction. Output only valid JSON when requested."
PY_SCRIPT="$ROOT/scripts/prompt_lab/run_prompt_lab.py"
LOG_ROOT="$ROOT/scripts/prompt_lab/logs"

MODELS=("gpt-5" "gpt-5-mini")
TEMPERATURE="${TEMPERATURE:-0}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
	echo "ERROR: OPENAI_API_KEY is not set." >&2
	exit 1
fi

if [[ ! -f "$PROMPT_PATH" ]]; then
	echo "ERROR: Prompt file not found: $PROMPT_PATH" >&2
	exit 1
fi

if [[ ! -d "$SAMPLES_DIR" ]]; then
	echo "ERROR: Samples directory not found: $SAMPLES_DIR" >&2
	exit 1
fi

# Collect PDFs (handle spaces via NUL delim) - compatible with macOS Bash 3.x (no mapfile)
PDFS=()
while IFS= read -r -d '' f; do
	PDFS+=("$f")
done < <(find "$SAMPLES_DIR" -maxdepth 1 -type f -name '*.pdf' -print0)
if (( ${#PDFS[@]} == 0 )); then
	echo "No PDFs found in $SAMPLES_DIR" >&2
	exit 1
fi

# Ensure output/tmp directories exist per model
for model in "${MODELS[@]}"; do
	mkdir -p "$ROOT/scripts/prompt_lab/outputs/$model" "$ROOT/scripts/prompt_lab/tmp/$model"
    mkdir -p "$LOG_ROOT/$model"
done

export ROOT PROMPT_PATH SYSTEM_PROMPT PY_SCRIPT LOG_ROOT MAX_PROCS

# Build task stream: (model, pdf) pairs as NUL-delimited tokens
TASK_STREAM="$(mktemp)"
trap 'rm -f "$TASK_STREAM"' EXIT
{
	for model in "${MODELS[@]}"; do
		for pdf in "${PDFS[@]}"; do
			printf '%s\0%s\0' "$model" "$pdf"
		done
	done
} > "$TASK_STREAM"

# Run tasks with up to MAX_PROCS in parallel
xargs -0 -n 2 -P "$MAX_PROCS" bash -c '
	set -euo pipefail
	MODEL="$1"; PDF="$2"
	OUTPUT_DIR="$ROOT/scripts/prompt_lab/outputs/$MODEL"
	TMP_DIR="$ROOT/scripts/prompt_lab/tmp/$MODEL"
    LOGFILE="$LOG_ROOT/$MODEL/$(basename "$PDF").log"
    mkdir -p "$(dirname "$LOGFILE")"
	echo "[${MODEL}] $(basename "$PDF")"
	START_TS=$(date +%s)
	{
	  echo "=== START $(date -Iseconds) MODEL=${MODEL} FILE=$(basename "$PDF") ==="
	  echo "OUTPUT_DIR=$OUTPUT_DIR"
	  echo "TMP_DIR=$TMP_DIR"
	  echo "PROMPT_PATH=$PROMPT_PATH"
	  echo "SYSTEM_PROMPT=$SYSTEM_PROMPT"
	  echo "MAX_PROCS=$MAX_PROCS"
	  python3 "$PY_SCRIPT" \
		--pdf "$PDF" \
		--prompt "$PROMPT_PATH" \
		--model "$MODEL" \
		--system "$SYSTEM_PROMPT" \
		--temperature "$TEMPERATURE" \
		--json-output \
		--output-dir "$OUTPUT_DIR" \
		--tmp-dir "$TMP_DIR"
	  EXIT_CODE=$?
	  END_TS=$(date +%s)
	  DURATION=$((END_TS-START_TS))
	  echo "EXIT_CODE=$EXIT_CODE DURATION_SEC=$DURATION END=$(date -Iseconds)"
	  echo "=== END $(basename "$PDF") ==="
	  exit $EXIT_CODE
	} 2>&1 | tee -a "$LOGFILE"
' bash < "$TASK_STREAM"

echo "All tasks finished. Outputs under: $ROOT/scripts/prompt_lab/outputs/{gpt-5,gpt-5-mini}"
