import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from typing import Any


@dataclass
class ModelOutput:
    model: str
    pdf_basename: str
    run_dir: Path
    output_json_path: Optional[Path]
    content: Optional[Dict[str, Any]]


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def clean_construct_name(term: Optional[str]) -> str:
    if not term:
        return ""
    cleaned = re.sub(r"\s*\([^)]*\)", "", term.strip())
    cleaned = re.sub(r"\s*\[[^\]]*\]", "", cleaned.strip())
    return normalize_text(cleaned)


def find_latest_runs(outputs_root: Path, models: List[str]) -> Dict[str, Dict[str, Path]]:
    """Return mapping: pdf_basename -> { model: latest_run_dir } for runs that have output.json or output.txt.
    Chooses the latest run per pdf per model based on run_dir mtime.
    """
    mapping: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for model in models:
        model_dir = outputs_root / model
        if not model_dir.exists():
            continue
        # Accept either run_* subfolders or a flat model_dir that contains run_* folders
        for run_dir in model_dir.glob("run_*"):
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                pdf_path = meta.get("pdf")
                if not pdf_path:
                    continue
                pdf_base = Path(pdf_path).name
                # ensure there is some output
                out_json = run_dir / "output.json"
                out_txt = run_dir / "output.txt"
                if not out_json.exists() and not out_txt.exists():
                    continue
                prev = mapping.get(pdf_base, {}).get(model)
                if not prev:
                    mapping[pdf_base][model] = run_dir
                else:
                    if run_dir.stat().st_mtime > prev.stat().st_mtime:
                        mapping[pdf_base][model] = run_dir
            except Exception:
                continue
    return mapping


def pairwise_within_model(outputs_root: Path, model_a: str, model_b: str) -> Dict[str, Dict[str, Path]]:
    """Build mapping for two synthetic 'models' that are actually two separate output roots (e.g., gpt-5_run1 vs gpt-5_run2)."""
    mapping: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for model in (model_a, model_b):
        model_dir = outputs_root / model
        if not model_dir.exists():
            continue
        for run_dir in model_dir.glob("run_*"):
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                pdf_path = meta.get("pdf")
                if not pdf_path:
                    continue
                pdf_base = Path(pdf_path).name
                prev = mapping.get(pdf_base, {}).get(model)
                if not prev or run_dir.stat().st_mtime > prev.stat().st_mtime:
                    mapping[pdf_base][model] = run_dir
            except Exception:
                continue
    return mapping


def load_output(run_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    out_json = run_dir / "output.json"
    out_txt = run_dir / "output.txt"
    if out_json.exists():
        try:
            return out_json, json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    if out_txt.exists():
        try:
            return out_txt, json.loads(out_txt.read_text(encoding="utf-8"))
        except Exception:
            return out_txt, None
    return None, None


def extract_sets(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute normalized sets and keyed structures for comparison."""
    result: Dict[str, Any] = {}

    # Paper metadata
    pm = doc.get("paper_metadata") or {}
    result["metadata"] = {
        "title": normalize_text(pm.get("title")),
        "authors": tuple(sorted(normalize_text(a) for a in (pm.get("authors") or []))),
        "publication_year": pm.get("publication_year"),
        "journal": normalize_text(pm.get("journal")),
        "research_type": normalize_text(pm.get("research_type")),
        "is_replication_study": bool(pm.get("is_replication_study", False)),
    }

    # Constructs
    constructs = doc.get("constructs") or []
    construct_terms = set()
    for c in constructs:
        construct_terms.add(clean_construct_name(c.get("term")))
    result["construct_terms"] = set(t for t in construct_terms if t)

    # Relationships
    rels = doc.get("relationships") or []
    rel_keys = set()
    for r in rels:
        st = clean_construct_name(r.get("subject_term"))
        ot = clean_construct_name(r.get("object_term"))
        status = normalize_text(r.get("status"))
        evidence_type = normalize_text(r.get("evidence_type"))
        effect = normalize_text(r.get("effect_direction"))
        if st and ot and status:
            rel_keys.add((st, ot, status, evidence_type, effect))
    result["relationship_keys"] = rel_keys

    # Measurements (by construct_term + name)
    meas = doc.get("measurements") or []
    meas_keys = set()
    for m in meas:
        ct = clean_construct_name(m.get("construct_term"))
        name = normalize_text(m.get("name"))
        if ct and name:
            meas_keys.add((ct, name))
    result["measurement_keys"] = meas_keys

    return result


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compare_models(per_model_docs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return heuristic comparison metrics and differences across models for one paper."""
    keys = sorted(per_model_docs.keys())
    per = {m: extract_sets(doc) for m, doc in per_model_docs.items()}

    # Choose a baseline (first by name)
    baseline = keys[0]
    base = per[baseline]

    comparison: Dict[str, Any] = {
        "baseline": baseline,
        "metrics": {},
        "diffs": {},
        "metadata": {m: per[m]["metadata"] for m in keys},
    }

    # Compare constructs
    base_ct = base["construct_terms"]
    for m in keys:
        ct = per[m]["construct_terms"]
        comparison["metrics"].setdefault("constructs", {})[m] = {
            "count": len(ct),
            "jaccard_with_baseline": jaccard(base_ct, ct),
            "missing_vs_baseline": sorted((base_ct - ct)),
            "additional_vs_baseline": sorted((ct - base_ct)),
        }

    # Compare relationships
    base_r = base["relationship_keys"]
    for m in keys:
        rk = per[m]["relationship_keys"]
        comparison["metrics"].setdefault("relationships", {})[m] = {
            "count": len(rk),
            "jaccard_with_baseline": jaccard(base_r, rk),
            "missing_vs_baseline": sorted(list(base_r - rk)),
            "additional_vs_baseline": sorted(list(rk - base_r)),
        }

    # Compare measurements
    base_m = base["measurement_keys"]
    for m in keys:
        mk = per[m]["measurement_keys"]
        comparison["metrics"].setdefault("measurements", {})[m] = {
            "count": len(mk),
            "jaccard_with_baseline": jaccard(base_m, mk),
            "missing_vs_baseline": sorted(list(base_m - mk)),
            "additional_vs_baseline": sorted(list(mk - base_m)),
        }

    return comparison


def llm_analyze(client: Any, paper_name: str, per_model_docs: Dict[str, Dict[str, Any]], heuristics: Dict[str, Any]) -> str:
    """Ask GPT-5 to produce a structured analysis given the raw JSONs and precomputed heuristics."""
    # Build compact JSONs to keep prompt size reasonable
    compact: Dict[str, Any] = {m: per_model_docs[m] for m in sorted(per_model_docs.keys())}
    heuristics_compact = heuristics

    system = (
        "You are an expert evaluator of information extraction consistency across models. "
        "Write a concise, structured analysis for each paper. Be specific about overlaps and differences."
    )

    instructions = (
        f"Compare the JSON outputs across models for paper: {paper_name}.\n"
        "Tasks:\n"
        "1) Assess metadata consistency (title, authors, year, research_type).\n"
        "2) Compare construct coverage: notable missing/additional constructs per model; overall consistency.\n"
        "3) Compare relationships (subject->object; status/evidence/effect). Summarize overlaps and important discrepancies.\n"
        "4) Compare measurements presence and naming conflicts.\n"
        "5) Provide a severity rating (Low/Medium/High) for divergence and a 1-100 consistency score.\n"
        "6) Provide brief recommendations to improve prompt or post-processing.\n"
        "Return Markdown with clear bullet points. Keep it under ~300-400 words."
    )

    user_payload = {
        "models": sorted(per_model_docs.keys()),
        "json_outputs": compact,
        "heuristics": heuristics_compact,
    }

    # Try OpenAI SDK v1 client first
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": instructions},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            return (resp.choices[0].message.content or "")
        except Exception as e:
            pass

    # Fallback to legacy openai v0 style if available
    try:
        import openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key  # type: ignore
        resp = openai.ChatCompletion.create(  # type: ignore
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instructions},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = resp["choices"][0]["message"]["content"]  # type: ignore
        return content or ""
    except Exception:
        return ""


def render_markdown_report(results: List[Tuple[str, Dict[str, Any], str]]) -> str:
    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"## Model Output Consistency Report")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append("")

    for paper_name, heur, llm_md in results:
        lines.append(f"### {paper_name}")
        lines.append("")
        # Brief heuristic summary
        lines.append("- Baseline model: **%s**" % heur.get("baseline", ""))
        cm = heur.get("metrics", {}).get("constructs", {})
        rm = heur.get("metrics", {}).get("relationships", {})
        mm = heur.get("metrics", {}).get("measurements", {})
        for section, data in (("Constructs", cm), ("Relationships", rm), ("Measurements", mm)):
            if not data:
                continue
            lines.append(f"- {section}:")
            for m in sorted(data.keys()):
                item = data[m]
                lines.append(
                    f"  - {m}: count={item['count']}, jaccard_vs_baseline={item['jaccard_with_baseline']:.2f}"
                )
        lines.append("")
        # LLM narrative
        if llm_md:
            lines.append(llm_md)
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and compare model JSON outputs per paper.")
    parser.add_argument(
        "--outputs-root",
        default=str(Path(__file__).parent / "outputs"),
        help="Root directory containing per-model output folders (default: scripts/prompt_lab/outputs)",
    )
    parser.add_argument(
        "--models",
        default="gpt-5,gpt-5-mini",
        help="Comma-separated model directory names under outputs-root",
    )
    parser.add_argument(
        "--within-model",
        action="store_true",
        help="Compare two runs of the same model (use --models like gpt-5_run1,gpt-5_run2)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum number of papers to include in the analysis (default: 5)",
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).parent / "reports" / "analysis_report.md"),
        help="Path to write the Markdown report",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use GPT-5 to produce a narrative analysis in addition to heuristic metrics",
    )

    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    mapping = pairwise_within_model(outputs_root, models[0], models[1]) if args.within_model else find_latest_runs(outputs_root, models)

    # Filter to papers that have all models present
    eligible = [pdf for pdf, by_model in mapping.items() if all(m in by_model for m in models)]
    eligible = sorted(eligible)[: args.max_papers]

    if not eligible:
        print("No papers with complete outputs across all models were found.")
        return

    # Prepare LLM client if requested
    client: Optional[Any] = None
    if args.use_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY is not set; proceeding without LLM narrative.")
        else:
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=api_key)
            except Exception as e:
                # v1 client unavailable; will attempt legacy fallback in llm_analyze
                print(f"OpenAI v1 client not available ({e}); will try legacy API fallback.")

    results: List[Tuple[str, Dict[str, Any], str]] = []

    for pdf_base in eligible:
        per_model_docs: Dict[str, Dict[str, Any]] = {}
        for model in models:
            run_dir = mapping[pdf_base][model]
            _, content = load_output(run_dir)
            if content is None:
                # If JSON parsing failed, skip this paper
                per_model_docs = {}
                break
            per_model_docs[model] = content
        if not per_model_docs:
            continue

        heuristics = compare_models(per_model_docs)
        llm_md = ""
        if args.use_llm:
            try:
                llm_md = llm_analyze(client, pdf_base, per_model_docs, heuristics)
            except Exception as e:
                llm_md = f"(LLM analysis failed: {e})"
        results.append((pdf_base, heuristics, llm_md))

    # Write report
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = render_markdown_report(results)
    report_path.write_text(report_md, encoding="utf-8")

    print(f"Saved analysis report: {report_path}")


if __name__ == "__main__":
    main()
