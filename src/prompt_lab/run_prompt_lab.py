import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

from openai import OpenAI
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all visible text from a PDF as a single whitespace-normalized string."""
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    text = " ".join(pages_text)
    return " ".join(text.split())


def load_prompt(prompt_path: Path) -> str:
    """Load prompt template text from file."""
    return prompt_path.read_text(encoding="utf-8")


def build_user_prompt(template: str, paper_text: str) -> str:
    """
    Build the final user message content by injecting paper text.

    - If the template contains the placeholder "<PAPER_TEXT>", it will be replaced with the PDF text.
    - Otherwise, the PDF text will be appended after the template, separated by two newlines.
    """
    if "<PAPER_TEXT>" in template:
        return template.replace("<PAPER_TEXT>", paper_text)
    return f"{template}\n\n{paper_text}"


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_output: bool = True,
    temperature: Optional[float] = None,
) -> Tuple[str, dict]:
    """Call the chat completion API and return (content, raw_response_dict)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs = {"model": model, "messages": messages}
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    if temperature is not None:
        kwargs["temperature"] = float(temperature)

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    # serialize response to primitive dict
    raw = json.loads(resp.model_dump_json())
    return content, raw


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a PDF + Prompt through an LLM, save inputs/outputs for manual review."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to the prompt template (use <PAPER_TEXT> placeholder or the PDF text will be appended)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model ID for the LLM (default: gpt-5)",
    )
    parser.add_argument(
        "--system",
        default="You are a research assistant skilled in knowledge extraction.",
        help="Optional system prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Request JSON object output from the model",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "outputs"),
        help="Directory to store outputs (a timestamped run folder will be created)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=str(Path(__file__).parent / "tmp"),
        help="Directory to store intermediate artifacts (extracted text, etc.)",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    tmp_root = Path(args.tmp_dir).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    ensure_dir(output_root)
    ensure_dir(tmp_root)

    # Prepare run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{ts}"
    ensure_dir(run_dir)

    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)

    # Load and assemble prompt
    template = load_prompt(prompt_path)
    user_prompt = build_user_prompt(template, paper_text)

    # Persist inputs for auditing
    write_text(run_dir / "prompt_template.txt", template)
    write_text(run_dir / "pdf_text.txt", paper_text)
    write_text(run_dir / "user_prompt_final.txt", user_prompt)

    # Prepare model client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)

    # Call LLM
    content, raw = call_llm(
        client=client,
        model=args.model,
        system_prompt=args.system,
        user_prompt=user_prompt,
        json_output=bool(args.json_output),
        temperature=args.temperature,
    )

    # Save outputs
    # Try to pretty-print JSON if possible
    saved_as = "output.txt"
    try:
        parsed = json.loads(content)
        write_text(run_dir / "output.json", json.dumps(parsed, ensure_ascii=False, indent=2))
        saved_as = "output.json"
    except Exception:
        write_text(run_dir / "output.txt", content)

    (run_dir / "response_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "pdf": str(pdf_path),
        "prompt": str(prompt_path),
        "model": args.model,
        "json_output": bool(args.json_output),
        "system": args.system,
        "timestamp": ts,
        "saved_primary_output": saved_as,
    }
    write_text(run_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"Saved run to: {run_dir}")
    print(f"Primary output: {saved_as}")


if __name__ == "__main__":
    main()


