"""T-Eval (official evaluators from third_party/T-Eval), API-Bank (official parser), MetaTool."""
import importlib
import re
import sys
import types

from ..common import THIRD_PARTY

# ----------------------------------------------------------------------------- T-Eval
_TEVAL = None


def _teval():
    """Import the official T-Eval evaluator classes. Their heavy/unused imports (mmengine.load is only
    used for dataset loading, which we bypass; `evaluate` and `termcolor` are unused in the code paths
    we call) are satisfied with tiny stubs so we do not install the full lagent/mmengine stack."""
    global _TEVAL
    if _TEVAL is None:
        for name, attrs in {"mmengine": {"load": lambda p: __import__("json").load(open(p))},
                            "evaluate": {}, "termcolor": {"colored": lambda s, *a, **k: s}}.items():
            if name not in sys.modules:
                try:
                    importlib.import_module(name)
                except ImportError:
                    m = types.ModuleType(name)
                    for k, v in attrs.items():
                        setattr(m, k, v)
                    sys.modules[name] = m
        root = str(THIRD_PARTY / "T-Eval")
        if root not in sys.path:
            sys.path.insert(0, root)
        from teval.evaluators.instruct_evaluator import InstructEvaluator
        from teval.evaluators.reason_retrieve_understand_evaluator import ReasonRetrieveUnderstandEvaluator
        from teval.evaluators.review_evaluator import ReviewEvaluator
        _TEVAL = (InstructEvaluator, ReviewEvaluator, ReasonRetrieveUnderstandEvaluator)
    return _TEVAL


def teval_postprocess_prediction(text: str) -> str:
    """Same as `split_special_tokens` in T-Eval's test.py (applied to every prediction)."""
    for tok in ["<eoa>", "<TOKENS_UNUSED_1>", "<|im_end|>", "\nuser", "\nassistant", "\nUSER", "[INST]", "<|user|>"]:
        text = text.split(tok)[0]
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    return text.strip("`").strip()


TEVAL_DETERMINISTIC = {"instruct_v2", "review_str_v2", "retrieve_str_v2", "understand_str_v2"}


def grade_teval(item: dict, prediction: str):
    """Per-item official metric. Returns (score, metrics). Subsets needing the sentence-embedding
    model (plan_*, reason_str_v2, reason_retrieve_understand_json_v2) return (None, ...)."""
    sub = item["metadata"]["subset"]
    Instruct, Review, RRU = _teval()
    pred = teval_postprocess_prediction(prediction)
    datum = {"template": item["metadata"]["template"], "prediction": pred, "ground_truth": item["reference"],
             "meta_data": item["metadata"]["meta_data"], "origin_prompt": item["messages"]}
    if sub == "instruct_v2":
        ev = Instruct.__new__(Instruct)
        m = ev._evaluate(ev._process_response(datum))
        fmt = item["metadata"]["meta_data"]["response_format"]
        return float(m[f"{fmt}_args_em_metric"]), m
    if sub == "review_str_v2":
        ev = Review.__new__(Review)
        m = ev._evaluate(ev._process_response(datum))
        return float(m["review_quality"]), m
    if sub in ("retrieve_str_v2", "understand_str_v2"):
        ev = RRU.__new__(RRU)
        ev.default_prompt_type = "str"
        ev.eval_type = "retrieve" if sub.startswith("retrieve") else "understand"
        ev.sentence_model = None  # 'thought' similarity is never computed for these str subsets
        sample, _err = ev._process_response(datum)
        m = ev._post_process([sample])
        key = "name" if ev.eval_type == "retrieve" else "args"
        return float(m[key]), {k: float(v) for k, v in m.items()}
    return None, {"status": "needs_sentence_embedding_model (all-mpnet-base-v2); use --teval-full"}


def teval_gold_prediction(item: dict) -> str:
    """Build a response that the official evaluator should score 1.0 (for self-tests / oracle)."""
    import json
    sub, gt, tpl = item["metadata"]["subset"], item["reference"], item["metadata"]["template"]
    fmt = (item["metadata"]["meta_data"] or {}).get("response_format")
    if sub == "instruct_v2":
        if fmt == "json":
            return json.dumps({tpl["thought"]: "ok", tpl["action"]: gt["action"], tpl["args"]: gt["args"]},
                              ensure_ascii=False)
        return (tpl["thought_start"] + "ok" + tpl["thought_end"] + tpl["action_start"] + gt["action"]
                + tpl["action_end"] + tpl["args_start"] + repr(gt["args"]) + tpl["args_end"])  # parsed by ast.literal_eval
    if sub == "review_str_v2":
        return f"Answer: {gt['answer']}"
    if sub == "retrieve_str_v2":
        return gt["name"]
    if sub == "understand_str_v2":
        return gt["args"] if isinstance(gt["args"], str) else json.dumps(gt["args"], ensure_ascii=False)
    return json.dumps(gt, ensure_ascii=False)


# ----------------------------------------------------------------------------- API-Bank
_APIBANK_PARSE = None


def _apibank_parse():
    """Official regex-based parser `parse_api_call` from api-bank/api_call_extraction.py (no eval())."""
    global _APIBANK_PARSE
    if _APIBANK_PARSE is None:
        import importlib.util
        p = THIRD_PARTY / "DAMO-ConvAI-apibank" / "api-bank" / "api_call_extraction.py"
        spec = importlib.util.spec_from_file_location("apibank_api_call_extraction", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _APIBANK_PARSE = mod.parse_api_call
    return _APIBANK_PARSE


def _norm_param(v):
    return " ".join(str(v).strip().strip("'\"").split()).lower()


def grade_apibank(item: dict, output: str):
    """Level-1/2 API-call items: parse model and gold calls with the official parser; correct iff the
    API name matches and every parameter matches after whitespace/quote/case normalization.
    (The official evaluator additionally *executes* the call against simulated APIs and compares
    results; for the given-description JSON test files that execution check reduces to argument
    agreement except for a few time-format variants -- see eval/README.md caveats.)"""
    if item["metadata"]["task_type"] != "api_call":
        return None, {"status": "level-3 requires interactive tool execution; not graded"}
    parse = _apibank_parse()
    try:
        g_name, g_params = parse(item["reference"])
    except Exception:
        return None, {"status": "gold unparsable"}
    try:
        p_name, p_params = parse(output)
    except Exception:
        return 0.0, {"parsed": None}
    ok = (p_name == g_name and set(p_params) == set(g_params)
          and all(_norm_param(p_params[k]) == _norm_param(g_params[k]) for k in g_params))
    return float(ok), {"parsed": {"api": p_name, "params": p_params}}


# ----------------------------------------------------------------------------- MetaTool
_TOOL_LINE = re.compile(r"^\s*\d+\.\s*tool name:\s*(.+?),\s*tool description:", re.M)


def metatool_candidates(prompt: str):
    return _TOOL_LINE.findall(prompt)


def _mentioned(text: str, names: list[str]):
    t = text.lower()
    return {n for n in names if re.search(rf"(?<![\w-]){re.escape(n.lower())}(?![\w-])", t)}


def grade_metatool(item: dict, output: str):
    """MetaTool has no released scoring script; this re-implements the paper's accuracy definitions:
    Task1 (awareness): first yes/no in the answer must equal the label.
    Task2 sub1/2/3 (single-tool selection): the set of candidate tool names mentioned must be exactly
      {gold}; when the gold tool is absent from the list (sub3 reliability), the answer must say
      'None' and mention no candidate.
    Task2 sub4 (multi-tool): mentioned candidates must equal the two gold tools."""
    task, ref = item["metadata"]["task"], item["reference"]
    text = output.strip()
    if task == "Task1":
        m = re.search(r"\b(yes|no)\b", text.lower())
        pred = m.group(1) if m else None
        return float(pred == ref), {"parsed": pred}
    cands = metatool_candidates(item["prompt"])
    ment = _mentioned(text, cands)
    if task == "Task2-Subtask4":
        return float(ment == set(ref)), {"parsed": sorted(ment)}
    if ref == "None":
        says_none = re.search(r"\bnone\b", text.lower()) is not None
        return float(says_none and not ment), {"parsed": "None" if says_none and not ment else sorted(ment)}
    return float(ment == {ref}), {"parsed": sorted(ment)}


# ----------------------------------------------------------------------------- Seal-Tools
def _parse_calls(text: str):
    """Extract the api_calling list from a model answer: the outermost [...] span, parsed as JSON,
    falling back to a Python literal (Seal-Tools gold uses Python-literal quoting). No eval()."""
    import ast
    import json
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e <= s:
        return None
    span = text[s:e + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(span)
        except Exception:
            continue
        if isinstance(v, list) and all(isinstance(c, dict) for c in v):
            return v
    return None


def _call_key(c: dict):
    params = c.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    return (str(c.get("api", "")).strip(), tuple(sorted((str(k), _norm_param(v)) for k, v in params.items())))


def _prf(pred: list, gold: list):
    from collections import Counter
    P, G = Counter(pred), Counter(gold)
    tp = sum((P & G).values())
    p = tp / sum(P.values()) if P else 0.0
    r = tp / sum(G.values()) if G else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def grade_sealtool(item: dict, output: str):
    """Seal-Tools: score = 1 iff the multiset of predicted (api, parameters) calls equals the gold
    multiset ("responses" placeholders are ignored; values compared after whitespace/quote/case
    normalization). Metrics follow the Seal-Tools paper: format validity, API-name P/R/F1 and
    parameter-level ((api, key, value)) P/R/F1."""
    gold = item["reference"]
    pred = _parse_calls(output)
    if pred is None:
        return 0.0, {"parsed": None, "format_ok": False, "api_f1": 0.0, "param_f1": 0.0}
    pk, gk = [_call_key(c) for c in pred], [_call_key(c) for c in gold]
    api_p, api_r, api_f1 = _prf([k[0] for k in pk], [k[0] for k in gk])
    par = lambda ks: [(k[0], kv) for k in ks for kv in k[1]]
    par_p, par_r, par_f1 = _prf(par(pk), par(gk))
    from collections import Counter
    ok = Counter(pk) == Counter(gk)
    return float(ok), {"parsed": pred, "format_ok": True, "api_p": api_p, "api_r": api_r, "api_f1": api_f1,
                       "param_p": par_p, "param_r": par_r, "param_f1": par_f1}
