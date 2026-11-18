"""Deterministic graders: math (math-verify), MCQ letter extraction, short-answer EM/F1."""
import re
from collections import Counter

from ..common import last_boxed, normalize_answer

# ----------------------------------------------------------------------------- math
_MV = None


def _mv():
    global _MV
    if _MV is None:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
        _MV = (parse, verify, LatexExtractionConfig, ExprExtractionConfig)
    return _MV


def extract_math_answer(text: str):
    """Final answer string: last \\boxed{}; else 'answer is X'; else last number."""
    b = last_boxed(text)
    if b is not None:
        return b
    m = re.findall(r"answer is[:\s]*\$?([^\n$]+?)\$?(?:\.|\n|$)", text, re.I)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def grade_math(text: str, refs) -> tuple[float, str | None]:
    """Return (score, parsed). refs: str or list[str] (OlympiadBench multi-answer -> all must match)."""
    parse, verify, *_ = _mv()
    pred = extract_math_answer(text)
    if pred is None:
        return 0.0, None
    refs = refs if isinstance(refs, list) else [refs]
    gold_str = ", ".join(str(r) for r in refs)
    try:
        g = parse(f"\\boxed{{{gold_str}}}")
        p = parse(f"\\boxed{{{pred}}}")
        ok = bool(g) and bool(p) and verify(g, p)
        if not ok:  # string fallback (e.g. text answers) after normalization
            ok = _norm_math_str(pred) == _norm_math_str(gold_str)
    except Exception:
        ok = _norm_math_str(pred) == _norm_math_str(gold_str)
    return float(ok), pred


def _norm_math_str(s):
    s = str(s).strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    return s.lower()


# ----------------------------------------------------------------------------- MCQ
LETTERS = "ABCDEFGHIJ"


def extract_letter(text: str, n_options: int = 10):
    """MMLU-Pro official extraction order: 'answer is (X)' -> 'Answer: X' -> last standalone letter."""
    L = LETTERS[:n_options]
    b = last_boxed(text)
    if b:
        m = re.search(rf"\b([{L}])\b", re.sub(r"\\text\{(.*?)\}", r"\1", b))
        if m:
            return m.group(1)
    m = re.findall(rf"answer is \(?([{L}])\)?", text)
    if m:
        return m[-1]
    m = re.findall(rf"[aA]nswer:\s*\(?([{L}])\)?", text)
    if m:
        return m[-1]
    m = re.findall(rf"\b([{L}])\b", text)
    return m[-1] if m else None


def grade_mcq(text: str, gold: str, n_options: int = 10):
    p = extract_letter(text, n_options)
    return float(p == gold), p


# ----------------------------------------------------------------------------- short answer
def extract_short_answer(text: str) -> str:
    m = re.findall(r"answer\s*:\s*(.+)", text, re.I)
    ans = m[-1] if m else (text.strip().splitlines() or [""])[-1]
    ans = re.sub(r"\*\*", "", ans).strip()
    b = last_boxed(ans)
    return (b if b is not None else ans).strip().rstrip(".")


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def grade_short(text: str, aliases: list[str]):
    """Normalized exact match over aliases (TriviaQA/FreebaseQA/SQuAD). Also returns F1 / contains."""
    pred = extract_short_answer(text)
    np_ = normalize_answer(pred)
    golds = [normalize_answer(a) for a in aliases if str(a).strip()]
    em = float(any(np_ == g for g in golds))
    f1 = max((f1_score(pred, a) for a in aliases), default=0.0)
    contains = float(any(g and re.search(rf"(^|\s){re.escape(g)}(\s|$)", normalize_answer(text)) for g in golds))
    return em, pred, {"f1": f1, "contains": contains}
