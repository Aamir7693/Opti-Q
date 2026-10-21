
#!/usr/bin/env python3
"""
SimpleQA Factuality Grading and Evaluation
==========================================

Description
-----------
This script evaluates LLM-generated answers on the SimpleQA dataset
using an OpenAI model as a factuality grader.

It compares each generated prediction against the original question
and annotated reference answer using the SimpleQA grading prompt
from OpenAI's simple-evals repository.

The script supports batch grading, parallel API requests, SQLite
caching, automatic column mapping, and evaluation reports.

Only CSV and JSONL files containing "_simpleqa" in their filenames
are processed. MMLU files are excluded.


Grading Logic
-------------
Each prediction is assigned one of three factuality labels:

    A -> CORRECT
         The prediction correctly answers the question.

    B -> INCORRECT
         The prediction provides an incorrect answer.

    C -> NOT_ATTEMPTED
         The prediction does not attempt to answer the question.

If grading fails after the configured retry attempts, the result
is labeled GRADER_ERROR.

The grading model and prompt are specified in config.yaml.


Evaluation Metrics
------------------
The script computes the following metrics:

    Correct Rate:
        CORRECT / Valid Grades

    Incorrect Rate:
        INCORRECT / Valid Grades

    Not Attempted Rate:
        NOT_ATTEMPTED / Valid Grades

    Accuracy Given Attempted:
        CORRECT / (CORRECT + INCORRECT)

    SimpleQA F1:
        Harmonic mean of Correct Rate and
        Accuracy Given Attempted.

    Cosine Similarity Diagnostics:
        - Mean cosine similarity for correct predictions
        - Mean cosine similarity for incorrect predictions
        - Pearson correlation between cosine and correctness
        - High-cosine incorrect prediction counts and rates

Cosine similarity is used only for supplementary analysis.
The primary factuality metric is Correct Rate.

GRADER_ERROR rows are excluded from valid-grade denominators.


Input Requirements
------------------
Supported formats:
    - CSV (.csv)
    - JSON Lines (.jsonl)

Required logical fields:
    - question   : Original question
    - target     : Annotated reference answer
    - prediction : LLM-generated answer

Optional fields:
    - method     : Model or evaluation method
    - cosine     : Previously computed cosine similarity

Column names and file-specific mappings are configured using
aliases and file_overrides in config.yaml.


Configuration
-------------
The script requires a config.yaml file containing the relevant
paths, grader settings, and column mappings.

Configuration includes:
    - input_dir
    - output_dir
    - cache_db
    - prompt_cache
    - log_file
    - grader.model
    - grader.prompt_url
    - grader.max_retries
    - grader.max_workers
    - grader.max_output_tokens

Set the OpenAI API key before grading:

    export OPENAI_API_KEY="your_api_key"

Optionally override the grader model:

    export SIMPLEQA_GRADER_MODEL="model_name"


How to Run
----------
1. Fetch and cache the SimpleQA reference grading prompt:

    python factuality_grader.py fetch-prompt

2. Inspect input files and detected column mappings:

    python factuality_grader.py inspect

3. Inspect a specific input file:

    python factuality_grader.py inspect --only results_simpleqa.csv

4. Grade all discovered SimpleQA files:

    python factuality_grader.py grade

5. Grade a specific file:

    python factuality_grader.py grade --only results_simpleqa.csv

6. Test grading on a limited number of rows:

    python factuality_grader.py grade --only results_simpleqa.csv --limit 10

7. Force regrading of a specific file, ignoring cached results:

    python factuality_grader.py grade --only results_simpleqa.csv --force-rerun

8. Grade multiple selected files:

    python factuality_grader.py grade --only file1_simpleqa.csv file2_simpleqa.csv

9. Use a custom configuration file:

    python factuality_grader.py --config custom_config.yaml grade

Note:
    Replace run.py with the actual Python script filename.

    The --force-rerun option requires --only to prevent
    accidentally regrading every input file.


Caching and Parallel Processing
-------------------------------
Grading results are stored in a SQLite database.

Cache keys are generated from:
    - Question
    - Reference answer
    - Model prediction
    - Grader model
    - Grading prompt hash

Previously graded examples are reused unless --force-rerun
is specified.

API requests are processed concurrently using ThreadPoolExecutor.
Failed grading requests are retried with exponential backoff.


Output Files
------------
For each processed input file, the script creates:

    normalized_input.csv
        Standardized question, target, prediction, method,
        and cosine similarity fields.

    graded.csv
        Row-level factuality grades, raw grader responses,
        error messages, and cache-hit indicators.

    summary.csv
        Aggregated factuality and cosine similarity metrics.

    disagreements.csv
        Predictions with high cosine similarity but
        INCORRECT factuality grades.

    errors.csv
        Rows where the factuality grader failed.

    run_metadata.json
        Grader model, prompt hash, column mapping,
        processing timestamp, and evaluation settings.

Combined output files:

    all_graded.csv
    combined_summary.csv
    report.html

All outputs are saved under the configured output_dir.


Dependencies
------------
    pip install pandas requests pyyaml openai

Python standard-library modules are used for SQLite caching,
parallel execution, logging, hashing, and report generation.


Notes
-----
- This script evaluates SimpleQA factuality, not MMLU accuracy.
- Grading requires access to the configured OpenAI model.
- The reference grading prompt must be fetched before grading.
- Existing cached grades are reused when available.
- API grading may incur usage costs.
- Factuality grades are separate from cosine similarity scores.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import time

import pandas as pd
import requests
import yaml
from openai import OpenAI


LABEL_MAP = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS grades (
  cache_key TEXT PRIMARY KEY,
  grader_model TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  question TEXT NOT NULL,
  target TEXT NOT NULL,
  prediction TEXT NOT NULL,
  grade TEXT NOT NULL,
  raw_output TEXT,
  error TEXT,
  created_at TEXT NOT NULL
)
"""


def load_config(path="config.yaml"):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent

    for key in ["input_dir", "output_dir", "cache_db", "prompt_cache", "log_file"]:
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (base / p).resolve())

    if os.getenv("SIMPLEQA_GRADER_MODEL"):
        cfg["grader"]["model"] = os.environ["SIMPLEQA_GRADER_MODEL"]

    return cfg


def setup_logging(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_template(source):
    """
    Extract GRADER_TEMPLATE from OpenAI's simple-evals source without
    executing the downloaded Python file.
    """
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id == "GRADER_TEMPLATE"
            for target in targets
        ):
            continue

        # Current simple-evals uses:
        # GRADER_TEMPLATE = <string literal>.strip()
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "strip"
            and not value.args
            and not value.keywords
        ):
            value = value.func.value

        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value.strip()

        raise RuntimeError(
            "Found GRADER_TEMPLATE, but its assignment format "
            "is unsupported."
        )

    raise RuntimeError(
        "Could not locate GRADER_TEMPLATE in reference source"
    )


def fetch_prompt(cfg):
    url = cfg["grader"]["prompt_url"]
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    template = extract_template(response.text)

    path = Path(cfg["prompt_cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")
    return template, sha256_text(template)


def load_prompt(cfg):
    path = Path(cfg["prompt_cache"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python run.py fetch-prompt"
        )
    template = path.read_text(encoding="utf-8").strip()
    return template, sha256_text(template)


def discover_files(cfg, only=None):
    root = Path(cfg["input_dir"])
    # files = sorted(list(root.glob("*.csv")) + list(root.glob("*.jsonl")))
    # Only process SimpleQA files.
    # This prevents MMLU files in the same directory from being graded
    # using the SimpleQA grader.
    files = sorted(
        p for p in (
            list(root.glob("*.csv")) +
            list(root.glob("*.jsonl"))
        )
        if "_simpleqa" in p.stem.lower()
    )
    if only:
        wanted = set(only)
        files = [p for p in files if p.name in wanted]
    return files


def read_table(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def pick_alias(columns, aliases):
    cols = list(columns)
    for a in aliases:
        if a in cols:
            return a
    low = {str(c).lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None


def resolve_mapping(path, df, cfg):
    override = cfg.get("file_overrides", {}).get(path.name, {})
    aliases = cfg.get("aliases", {})
    m = {}

    for logical in ["question", "target", "prediction", "method", "cosine"]:
        m[logical] = override.get(
            logical,
            pick_alias(df.columns, aliases.get(logical, []))
        )

    if "method_value" in override:
        m["method_value"] = str(override["method_value"])

    if "prediction_columns" in override:
        m["prediction_columns"] = dict(override["prediction_columns"])

    if not m.get("question"):
        raise ValueError(f"{path.name}: question column unresolved")
    if not m.get("target"):
        raise ValueError(f"{path.name}: target/gold column unresolved")
    if not m.get("prediction") and not m.get("prediction_columns"):
        raise ValueError(f"{path.name}: prediction column unresolved")

    return m


def normalize(path, cfg):
    raw = read_table(path)
    m = resolve_mapping(path, raw, cfg)

    qcol, tcol = m["question"], m["target"]
    ccol = m.get("cosine")
    rows = []

    if m.get("prediction_columns"):
        for idx, row in raw.iterrows():
            for method, pcol in m["prediction_columns"].items():
                rows.append({
                    "source_file": path.name,
                    "source_row": int(idx),
                    "method": str(method),
                    "question": row.get(qcol, ""),
                    "target": row.get(tcol, ""),
                    "prediction": row.get(pcol, ""),
                    "cosine": row.get(ccol) if ccol else None,
                })
    else:
        pcol = m["prediction"]
        method_col = m.get("method")
        fixed_method = m.get("method_value")
        for idx, row in raw.iterrows():
            method = (
                fixed_method
                if fixed_method is not None
                else row.get(method_col, path.stem)
                if method_col
                else path.stem
            )
            rows.append({
                "source_file": path.name,
                "source_row": int(idx),
                "method": str(method),
                "question": row.get(qcol, ""),
                "target": row.get(tcol, ""),
                "prediction": row.get(pcol, ""),
                "cosine": row.get(ccol) if ccol else None,
            })

    out = pd.DataFrame(rows)
    for col in ["question", "target", "prediction", "method"]:
        out[col] = out[col].fillna("").astype(str)
    out["cosine"] = pd.to_numeric(out["cosine"], errors="coerce")
    return out, m


def parse_grade(text):
    if not text:
        return None
    cleaned = text.strip().upper()
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned]
    match = re.search(r"\b([ABC])\b", cleaned)
    return LABEL_MAP[match.group(1)] if match else None


def cache_key(question, target, prediction, model, prompt_hash):
    payload = "\n---\n".join(
        [question, target, prediction, model, prompt_hash]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(CACHE_SCHEMA)
        self.conn.commit()

    def get(self, key):
        row = self.conn.execute(
            "SELECT grade, raw_output, error FROM grades WHERE cache_key=?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return {"grade": row[0], "raw": row[1] or "", "error": row[2]}

    def put(self, *, key, model, prompt_hash, question, target, prediction,
            grade, raw, error):
        self.conn.execute(
            """INSERT OR REPLACE INTO grades
               (cache_key,grader_model,prompt_hash,question,target,prediction,
                grade,raw_output,error,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                key, model, prompt_hash, question, target, prediction,
                grade, raw, error, datetime.now(timezone.utc).isoformat()
            ),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


class Grader:
    def __init__(self, cfg, template):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.model = cfg["model"]
        self.template = template
        self.max_retries = int(cfg.get("max_retries", 4))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 20))

    def grade(self, question, target, prediction):
        prompt = self.template.format(
            question=question,
            target=target,
            predicted_answer=prediction,
        )

        last_error, last_raw = None, ""

        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=self.max_output_tokens,
                )
                raw = (response.output_text or "").strip()
                last_raw = raw
                parsed = parse_grade(raw)
                if parsed:
                    return parsed, raw, None
                last_error = f"Could not parse A/B/C from: {raw!r}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.max_retries - 1:
                time.sleep(min(2 ** attempt, 8))

        return "GRADER_ERROR", last_raw, last_error


def summarize_group(group, threshold):
    grade = group["factuality_grade"]
    correct = int((grade == "CORRECT").sum())
    incorrect = int((grade == "INCORRECT").sum())
    na = int((grade == "NOT_ATTEMPTED").sum())
    errors = int((grade == "GRADER_ERROR").sum())

    valid = correct + incorrect + na
    attempted = correct + incorrect
    cr = correct / valid if valid else math.nan
    ir = incorrect / valid if valid else math.nan
    nr = na / valid if valid else math.nan
    aga = correct / attempted if attempted else math.nan
    f1 = (
        2 * cr * aga / (cr + aga)
        if valid and pd.notna(aga) and cr + aga > 0
        else (0.0 if valid else math.nan)
    )

    result = {
        "n": len(group),
        "valid_grades": valid,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": na,
        "grader_error": errors,
        "correct_rate": cr,
        "incorrect_rate": ir,
        "not_attempted_rate": nr,
        "accuracy_given_attempted": aga,
        "simpleqa_f1": f1,
    }

    cosine = pd.to_numeric(group["cosine"], errors="coerce")
    if cosine.notna().sum() >= 2:
        tmp = group.loc[cosine.notna()].copy()
        tmp["cosine_num"] = cosine.loc[cosine.notna()]
        tmp["is_correct"] = (tmp["factuality_grade"] == "CORRECT").astype(int)

        result["mean_cosine_correct"] = tmp.loc[
            tmp["factuality_grade"] == "CORRECT", "cosine_num"
        ].mean()
        result["mean_cosine_incorrect"] = tmp.loc[
            tmp["factuality_grade"] == "INCORRECT", "cosine_num"
        ].mean()
        result["cosine_factuality_pearson"] = tmp["cosine_num"].corr(
            tmp["is_correct"]
        )

        high = tmp["cosine_num"] >= threshold
        high_n = int(high.sum())
        bad = int((high & (tmp["factuality_grade"] == "INCORRECT")).sum())
        result["high_cosine_threshold"] = threshold
        result["high_cosine_count"] = high_n
        result["high_cosine_incorrect_count"] = bad
        result["high_cosine_incorrect_rate"] = (
            bad / high_n if high_n else math.nan
        )

    return result


def make_summary(graded, threshold):
    rows = []
    for (source_file, method), group in graded.groupby(
        ["source_file", "method"], dropna=False
    ):
        row = {"source_file": source_file, "method": method}
        row.update(summarize_group(group, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements(graded, threshold):
    cosine = pd.to_numeric(graded["cosine"], errors="coerce")
    mask = (
        cosine.notna()
        & (cosine >= threshold)
        & (graded["factuality_grade"] == "INCORRECT")
    )
    return graded.loc[mask].copy()


def write_report(summary, path, model, prompt_hash):
    cols = [
        c for c in [
            "source_file", "method", "n", "correct_rate",
            "incorrect_rate", "not_attempted_rate",
            "accuracy_given_attempted", "simpleqa_f1",
            "cosine_factuality_pearson",
            "high_cosine_incorrect_count", "grader_error",
        ]
        if c in summary.columns
    ]

    def f(v):
        if pd.isna(v):
            return ""
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    th = "".join(f"<th>{escape(c)}</th>" for c in cols)
    trs = []
    for _, row in summary.iterrows():
        td = "".join(f"<td>{escape(f(row.get(c)))}</td>" for c in cols)
        trs.append(f"<tr>{td}</tr>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SimpleQA factuality report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:32px;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f3f3f3}} code{{background:#f3f3f3;padding:2px 4px}}
</style></head><body>
<h1>SimpleQA factuality report</h1>
<p>Grader: <code>{escape(model)}</code><br>
Prompt SHA-256: <code>{escape(prompt_hash)}</code></p>
<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>
<p><strong>Primary factual metric:</strong> correct_rate.
Cosine metrics are supplementary diagnostics.</p>
</body></html>"""
    Path(path).write_text(html, encoding="utf-8")


def inspect_command(cfg, only):
    files = discover_files(cfg, only)
    if not files:
        print("No input files found.")
        return

    for path in files:
        print(f"\nFILE: {path.name}")
        try:
            df, m = normalize(path, cfg)
            print(f"  normalized rows : {len(df)}")
            print(f"  question        : {m.get('question')}")
            print(f"  target          : {m.get('target')}")
            if m.get("prediction_columns"):
                print("  predictions:")
                for method, col in m["prediction_columns"].items():
                    print(f"    {method}: {col}")
            else:
                print(f"  prediction      : {m.get('prediction')}")
                print(
                    "  method          : "
                    + str(
                        m.get("method_value")
                        or m.get("method")
                        or path.stem
                    )
                )
            print(f"  cosine          : {m.get('cosine')}")
        except Exception as exc:
            print(f"  ERROR: {exc}")


def grade_command(cfg, only, limit, force_rerun=False):
    setup_logging(cfg["log_file"])
    files = discover_files(cfg, only)
    if not files:
        raise RuntimeError("No input CSV/JSONL files found.")

    template, prompt_hash = load_prompt(cfg)
    grader = Grader(cfg["grader"], template)
    cache = Cache(cfg["cache_db"])
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    threshold = float(cfg.get("analysis", {}).get("high_cosine_threshold", 0.8))

    all_graded, all_summary = [], []

    try:
        for path in files:
            logging.info("Processing %s", path.name)
            normalized, mapping = normalize(path, cfg)
            if limit:
                normalized = normalized.head(limit).copy()

            out_dir = output_root / path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(out_dir / "normalized_input.csv", index=False)

            graded = normalized.copy()
            graded["factuality_grade"] = None
            graded["grader_raw"] = None
            graded["grader_error"] = None
            graded["cache_hit"] = None

            tasks = []
            for idx, row in graded.iterrows():
                key = cache_key(
                    row["question"], row["target"], row["prediction"],
                    grader.model, prompt_hash
                )
                cached = None if force_rerun else cache.get(key)

                if cached:
                    graded.at[idx, "factuality_grade"] = cached["grade"]
                    graded.at[idx, "grader_raw"] = cached["raw"]
                    graded.at[idx, "grader_error"] = cached["error"]
                    graded.at[idx, "cache_hit"] = True
                else:
                    tasks.append((
                        idx,
                        key,
                        row["question"],
                        row["target"],
                        row["prediction"]
                    ))

            logging.info(
                "%s: %d rows, %d API grades, %d cache hits",
                path.name, len(graded), len(tasks), len(graded) - len(tasks)
            )

            def one(task):
                idx, key, q, t, p = task
                grade, raw, error = grader.grade(q, t, p)
                return idx, key, q, t, p, grade, raw, error

            with ThreadPoolExecutor(
                max_workers=int(cfg["grader"].get("max_workers", 6))
            ) as pool:
                future_map = [pool.submit(one, task) for task in tasks]
                for future in as_completed(future_map):
                    idx, key, q, t, p, grade, raw, error = future.result()
                    graded.at[idx, "factuality_grade"] = grade
                    graded.at[idx, "grader_raw"] = raw
                    graded.at[idx, "grader_error"] = error
                    graded.at[idx, "cache_hit"] = False
                    cache.put(
                        key=key, model=grader.model, prompt_hash=prompt_hash,
                        question=q, target=t, prediction=p, grade=grade,
                        raw=raw, error=error
                    )

            graded.to_csv(out_dir / "graded.csv", index=False)

            summary = make_summary(graded, threshold)
            summary.to_csv(out_dir / "summary.csv", index=False)

            disagreements(graded, threshold).to_csv(
                out_dir / "disagreements.csv", index=False
            )

            graded.loc[
                graded["factuality_grade"] == "GRADER_ERROR"
            ].to_csv(out_dir / "errors.csv", index=False)

            metadata = {
                "source_file": path.name,
                "grader_model": grader.model,
                "prompt_hash": prompt_hash,
                "prompt_source_url": cfg["grader"]["prompt_url"],
                "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                "normalized_rows": len(normalized),
                "mapping": mapping,
                "high_cosine_threshold": threshold,
            }
            (out_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )

            all_graded.append(graded)
            all_summary.append(summary)

        combined = pd.concat(all_graded, ignore_index=True)
        summary = pd.concat(all_summary, ignore_index=True)

        combined.to_csv(output_root / "all_graded.csv", index=False)
        summary.to_csv(output_root / "combined_summary.csv", index=False)

        write_report(
            summary,
            output_root / "report.html",
            grader.model,
            prompt_hash,
        )

        print("\nDone.")
        print("Combined summary:", output_root / "combined_summary.csv")
        print("HTML report:", output_root / "report.html")

    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--only", nargs="*", default=None)

    sub.add_parser("fetch-prompt")

    p_grade = sub.add_parser("grade")
    p_grade.add_argument("--only", nargs="*", default=None)
    p_grade.add_argument("--limit", type=int, default=None)

    p_grade.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Ignore cached grades for the files supplied with --only "
            "and overwrite only those cache entries."
        ),
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "inspect":
        inspect_command(cfg, args.only)
    elif args.command == "fetch-prompt":
        template, h = fetch_prompt(cfg)
        print("Reference prompt cached.")
        print("Characters:", len(template))
        print("SHA-256:", h)
        print("Path:", cfg["prompt_cache"])

    elif args.command == "grade":

        # Safety: do not accidentally rerun every file in input_dir.
        if args.force_rerun and not args.only:
            parser.error(
                "--force-rerun requires --only. "
                "Specify exactly which file(s) should be regraded."
            )

        grade_command(
            cfg,
            args.only,
            args.limit,
            force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
