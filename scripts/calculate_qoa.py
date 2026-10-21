#!/usr/bin/env python3
"""Compute Quality of Answer (QoA) scores for SimpleQA and MMLU-Pro CSVs.

Scoring:
    SimpleQA: MiniLM embedding cosine similarity, with exact-answer containment
              scored as 1.0. Missing answers receive NaN.
    MMLU-Pro: Binary 1/0 if the output contains the correct option letter in a
              recognized form OR the full correct option text. Invalid gold
              options or unparseable questions receive NaN.

Usage:
    # SimpleQA (explicit mode)
    python calculate_qoa.py -i simpleqa_results.csv -o simpleqa_scored.csv --dataset simpleqa

    # MMLU-Pro (explicit mode)
    python calculate_qoa.py -i mmlu_results.csv -o mmlu_scored.csv --dataset mmlu

    # Auto-detect from filename containing 'simpleqa', 'simple_qa', or 'mmlu'
    python calculate_qoa.py -i results_mmlu.csv -o results_mmlu_scored.csv

    # Custom CSV columns
    python calculate_qoa.py -i input.csv -o output.csv --dataset mmlu \
        --ann-col "Annotated Answer" --out-col "Final_Output" \
        --question-col "Original_Query"

    # SimpleQA embedding settings
    python calculate_qoa.py -i input.csv -o output.csv --dataset simpleqa \
        --batch-size 64 --chunk-size-words 250

Required columns:
    SimpleQA: 'Annotated Answer', 'Final_Output'
    MMLU-Pro: same columns plus 'Original_Query'

Output:
    All datasets: 'Quality_of_Answer' added to the original CSV columns.
    MMLU-Pro: additional MMLU audit columns for gold option, answer text,
              match flags, match type, and score status.

Dependencies:
    pip install numpy pandas torch transformers
    (PyTorch and transformers are loaded only for SimpleQA scoring.)
"""
import argparse
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s • %(levelname)s • %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Lazy loading: MMLU scoring does not require PyTorch or transformers.

VALID_GOLD_RE = re.compile(r"^[A-J]$", re.IGNORECASE)
QUESTION_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])\)\s*")
INSTRUCTION_SUFFIX_RE = re.compile(r"\s+Above are multiple-choice questions\b", re.IGNORECASE)


def safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_text(text: str) -> str:
    """Normalize punctuation and whitespace without folding meaningful case."""
    text = unicodedata.normalize("NFKC", safe_str(text))
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def parse_options(question: str) -> Dict[str, str]:
    """Find the longest ordered A), B), ... option sequence in the question."""
    question = safe_str(question)
    matches = list(QUESTION_LABEL_RE.finditer(question))
    best = None
    for start_idx, start in enumerate(matches):
        if start.group(1).upper() != "A":
            continue
        sequence = [start]
        expected = ord("B")
        for candidate in matches[start_idx + 1:]:
            if expected > ord("J"):
                break
            if candidate.group(1).upper() == chr(expected):
                sequence.append(candidate)
                expected += 1
        if len(sequence) >= 2 and (best is None or len(sequence) > len(best)):
            best = sequence
    if not best:
        return {}
    options = {}
    for i, match in enumerate(best):
        end = best[i + 1].start() if i + 1 < len(best) else len(question)
        answer_text = question[match.end():end].strip()
        if i + 1 == len(best):
            answer_text = INSTRUCTION_SUFFIX_RE.split(answer_text, maxsplit=1)[0].strip()
        options[match.group(1).upper()] = answer_text
    return options


def contains_gold_option(output: str, gold: str) -> bool:
    """Recognize explicit option selections, not arbitrary prose letters."""
    output, gold = safe_str(output), safe_str(gold).upper()
    if not output or not VALID_GOLD_RE.fullmatch(gold):
        return False
    g = re.escape(gold)
    if re.fullmatch(rf"\s*\(?\s*{g}\s*(?:\)|\.|-)?\s*", output, re.IGNORECASE):
        return True
    contextual = re.compile(
        rf"\b(?:final\s+answer|answer|option|choice)"
        rf"\s*(?:is|:|=|-)?\s*\(?\s*{g}\s*(?:\)|\.|-)?(?=\s|$)",
        re.IGNORECASE,
    )
    if contextual.search(output):
        return True
    punctuated = re.compile(
        rf"(?<![A-Za-z0-9])(?:\(\s*{g}\s*\)|\(\s*{g}(?![A-Za-z0-9])|{g}\s*\)|{g}[.-](?![A-Za-z0-9]))",
        re.IGNORECASE,
    )
    return bool(punctuated.search(output))


def contains_gold_text(output: str, gold_text: str) -> bool:
    gold_norm = normalize_text(gold_text)
    output_norm = normalize_text(output)
    return bool(gold_norm and output_norm and gold_norm in output_norm)


def score_mmlu_row(question: str, gold: str, output: str) -> Tuple[float, str, str, bool, bool]:
    gold, output = safe_str(gold).upper(), safe_str(output)
    if not VALID_GOLD_RE.fullmatch(gold):
        return np.nan, "invalid_gold", "", False, False
    options = parse_options(question)
    if gold not in options:
        return np.nan, "option_parse_failed", "", False, False
    gold_text = options[gold]
    if not output or output.upper().startswith("ERROR:"):
        return 0.0, "empty_or_error", gold_text, False, False
    option_match = contains_gold_option(output, gold)
    text_match = contains_gold_text(output, gold_text)
    if option_match and text_match:
        status = "option_and_text_match"
    elif option_match:
        status = "option_match"
    elif text_match:
        status = "text_match"
    else:
        status = "no_match"
    return float(option_match or text_match), status, gold_text, option_match, text_match


def score_dataframe(df: pd.DataFrame, ann_col: str, out_col: str, question_col: str) -> pd.DataFrame:
    missing = [c for c in (ann_col, out_col, question_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    result = df.copy()
    scores, gold_options, gold_texts, option_matches, text_matches, statuses = [], [], [], [], [], []
    for question, gold_raw, output in zip(result[question_col], result[ann_col], result[out_col]):
        gold = safe_str(gold_raw).upper()
        qoa, status, gold_text, option_match, text_match = score_mmlu_row(question, gold, output)
        scores.append(qoa)
        gold_options.append(gold if VALID_GOLD_RE.fullmatch(gold) else "")
        gold_texts.append(gold_text)
        option_matches.append(option_match)
        text_matches.append(text_match)
        statuses.append(status)
    result["Quality_of_Answer"] = scores
    result["MMLU_Gold_Option"] = gold_options
    result["MMLU_Gold_Text"] = gold_texts
    result["MMLU_Option_Match"] = option_matches
    result["MMLU_Text_Match"] = text_matches
    result["MMLU_Match_Type"] = statuses
    result["MMLU_Score_Status"] = [
        s if s in {"invalid_gold", "option_parse_failed"} else "scored" for s in statuses
    ]
    return result


def run_self_tests():
    for s in ["A", "(A)", "A)", "(A", "A.", "A-", "answer A", "answer is A", "Answer: A", "final answer (A)", "option A", "choice A", "the answer is A) Paris", "I choose A) Paris"]:
        assert contains_gold_option(s, "A"), s
    for s in ["I think the answer is B", "A spacecraft was launched", "This is incorrect", "Paris", "the answer is B) London"]:
        assert not contains_gold_option(s, "A"), s
    q = "What is the capital? A) London B) Paris C) Berlin D) Madrid Above are multiple-choice questions (with answers) related to geography. Respond with the option letter only."
    opts = parse_options(q)
    assert opts == {"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"}, opts
    assert score_mmlu_row(q, "B", "The answer is B) Paris.")[0] == 1
    assert score_mmlu_row(q, "B", "The answer is Paris.")[0] == 1
    assert score_mmlu_row(q, "B", "The answer is C) Paris.")[0] == 1
    assert score_mmlu_row(q, "B", "The answer is C) Berlin.")[0] == 0
    assert score_mmlu_row(q, "B", "")[0] == 0
    assert score_mmlu_row(q, "B", "ERROR: request failed")[0] == 0
    assert np.isnan(score_mmlu_row(q, "Paris", "Paris")[0])
    genetics_q = "Question. A) Genes are linked, genotypes were ABab ABab B) Genes are linked, genotypes were ABab aBAb Above are multiple-choice questions (with answers) related to biology."
    assert score_mmlu_row(genetics_q, "A", "The answer is B) Genes are linked, genotypes were ABab aBAb.")[0] == 0
    logger.info("Self-tests passed.")


def simpleqa_safe_str(obj: Any) -> str:
    # Preserve original SimpleQA conversion behavior.
    if isinstance(obj, (str, int, float)):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if obj is None:
        return ""
    try:
        if pd.isna(obj):
            return ""
    except Exception:
        pass
    return str(obj)


def mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def chunk_text_words(text: str, chunk_size: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return [""]
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def build_text_embedding_map(texts: List[str], chunk_size_words: int = 250,
                             batch_size: int = 64) -> Dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    if batch_size <= 0 or chunk_size_words <= 0:
        raise ValueError("batch_size and chunk_size_words must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    unique_texts = list(dict.fromkeys(texts))
    chunk_list: List[str] = []
    chunk_owner: List[int] = []
    for ti, t in enumerate(unique_texts):
        for c in chunk_text_words(t, chunk_size_words):
            chunk_list.append(c)
            chunk_owner.append(ti)

    logger.info("Embedding %d unique texts via %d total chunks (batch_size=%d)",
                len(unique_texts), len(chunk_list), batch_size)
    embs = []
    with torch.inference_mode():
        for i in range(0, len(chunk_list), batch_size):
            batch = chunk_list[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])
            embs.append(F.normalize(pooled, p=2, dim=1).detach().cpu().numpy())
    chunk_embs = np.vstack(embs) if embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    D = chunk_embs.shape[1] if chunk_embs.size else model.config.hidden_size
    sums = np.zeros((len(unique_texts), D), dtype=np.float32)
    counts = np.zeros((len(unique_texts),), dtype=np.int32)
    for ci, ti in enumerate(chunk_owner):
        sums[ti] += chunk_embs[ci]
        counts[ti] += 1
    text_map: Dict[str, np.ndarray] = {}
    for ti, t in enumerate(unique_texts):
        if counts[ti] == 0:
            vec = np.zeros((D,), dtype=np.float32)
        else:
            vec = sums[ti] / float(counts[ti])
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        text_map[t] = vec
    return text_map


def compute_qoa_series(annotated: pd.Series, outputs: pd.Series,
                       text_emb: Dict[str, np.ndarray]) -> np.ndarray:
    qoa = np.full((len(annotated),), np.nan, dtype=np.float32)
    for i, (a_raw, o_raw) in enumerate(zip(annotated.values, outputs.values)):
        a = simpleqa_safe_str(a_raw).strip()
        o = simpleqa_safe_str(o_raw).strip()
        if not a or not o:
            continue
        if a.lower() in o.lower():
            qoa[i] = 1.0
            continue
        ea = text_emb.get(a)
        eo = text_emb.get(o)
        if ea is not None and eo is not None:
            qoa[i] = float(np.dot(ea, eo))
    return qoa


def score_simpleqa_dataframe(df: pd.DataFrame, ann_col: str, out_col: str,
                             chunk_size_words: int, batch_size: int) -> pd.DataFrame:
    ann = df[ann_col].fillna("").astype(str)
    out = df[out_col].fillna("").astype(str)
    all_texts = [simpleqa_safe_str(x).strip() for x in pd.concat([ann, out]).tolist()]
    text_emb = build_text_embedding_map(all_texts, chunk_size_words, batch_size)
    result = df.copy()
    result["Quality_of_Answer"] = compute_qoa_series(ann, out, text_emb)
    return result


def detect_dataset(input_file: str) -> str:
    name = Path(input_file).stem.lower()
    is_mmlu = "mmlu" in name
    is_simpleqa = "simpleqa" in name or "simple_qa" in name
    if is_mmlu and not is_simpleqa:
        return "mmlu"
    if is_simpleqa and not is_mmlu:
        return "simpleqa"
    raise ValueError(
        "Cannot determine dataset from filename. Supply --dataset simpleqa or --dataset mmlu."
    )


def main():
    p = argparse.ArgumentParser(description="Compute QoA for SimpleQA or MMLU-Pro CSVs.")
    p.add_argument("-i", "--input-file", required=True, help="Input CSV path")
    p.add_argument("-o", "--output-file", required=True, help="Output CSV path")
    p.add_argument("--ann-col", default="Annotated Answer", help="Annotated answer column")
    p.add_argument("--out-col", default="Final_Output", help="Model output column")
    p.add_argument("--question-col", default="Original_Query", help="MMLU question/options column")
    p.add_argument("--dataset", choices=["auto", "simpleqa", "mmlu"], default="auto",
                   help="Scoring mode; auto detects dataset from input filename")
    p.add_argument("--batch-size", type=int, default=64, help="SimpleQA embedding batch size")
    p.add_argument("--chunk-size-words", type=int, default=250, help="SimpleQA chunk size in words")
    p.add_argument("--skip-tests", action="store_true", help="Skip MMLU regression tests")
    args = p.parse_args()

    dataset = detect_dataset(args.input_file) if args.dataset == "auto" else args.dataset
    logger.info("Loading %s (dataset=%s)", args.input_file, dataset)
    df = pd.read_csv(args.input_file, dtype=str)
    required = [args.ann_col, args.out_col] + ([args.question_col] if dataset == "mmlu" else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if dataset == "mmlu":
        if not args.skip_tests:
            run_self_tests()
        scored = score_dataframe(df, args.ann_col, args.out_col, args.question_col)
    else:
        scored = score_simpleqa_dataframe(df, args.ann_col, args.out_col,
                                          args.chunk_size_words, args.batch_size)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_file, index=False)
    logger.info("QoA summary:\n%s", scored["Quality_of_Answer"].describe().to_string())
    if dataset == "mmlu":
        logger.info("MMLU score statuses:\n%s", scored["MMLU_Match_Type"].value_counts().to_string())
    logger.info("Saved results to %s", args.output_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
