"""Shared paths and helpers for the evaluation harness."""
import json
import os
import re
import string
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
ROOT = EVAL_DIR.parent
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
RAW_DIR = DATA_DIR / "raw"
THIRD_PARTY = EVAL_DIR / "third_party"
CACHE_DIR = EVAL_DIR / ".cache"

# Keep NLTK fully offline and self-contained (punkt_tab / stopwords / tagger fetched once into
# eval/.nltk_data from nltk/nltk_data at a pinned commit; no pickle-based punkt).
os.environ.setdefault("NLTK_DATA", str(EVAL_DIR / ".nltk_data"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_items(dataset: str, pool: str | None = None, ids: list[str] | None = None):
    """Load processed items of `dataset`, restricted to split pool `pool` (via data/splits files)."""
    rows = read_jsonl(PROC_DIR / f"{dataset}.jsonl")
    if pool:
        sp = SPLIT_DIR / dataset / f"{pool}.json"
        if not sp.exists():
            avail = sorted(p.stem for p in (SPLIT_DIR / dataset).glob("*.json"))
            raise SystemExit(f"dataset {dataset} has no pool '{pool}' (available: {avail})")
        keep = set(json.load(open(sp)))
        rows = [r for r in rows if r["id"] in keep]
        assert all(r["split_pool"] == pool for r in rows), "processed split_pool disagrees with split file"
    if ids is not None:
        want = set(ids)
        rows = [r for r in rows if r["id"] in want]
    return rows


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str | None) -> str:
    """Remove Qwen3-style <think>...</think> blocks (and an unterminated leading block)."""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag consumed by the template, closing tag in the output
        text = text.split("</think>", 1)[1]
    if "<think>" in text:   # truncated inside thinking -> no final answer
        text = text.split("<think>", 1)[0]
    return text.strip()


def normalize_answer(s: str) -> str:
    """SQuAD / TriviaQA official normalization: lower, strip punctuation, articles, extra spaces."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.replace("_", " ")
    return " ".join(s.split())


def last_boxed(s: str):
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def extract_code_block(text: str, lang: str = "python") -> str:
    """Last fenced code block (```python preferred); else the whole text."""
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.S)
    py = re.findall(rf"```{lang}\n(.*?)```", text, re.S)
    if py:
        return py[-1]
    if blocks:
        return blocks[-1]
    return text
