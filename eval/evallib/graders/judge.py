"""OPTIONAL LLM-judge graders (not used in main experiments): MT-Bench single-answer grading
(FastChat judge prompts) and AlpacaEval 2 pairwise win rate vs the gpt4_1106_preview baseline
(alpaca_eval weighted_alpaca_eval_gpt4_turbo prompt). The judge endpoint/model are fixed in
eval/configs/tasks.yaml (judge:) so scores are comparable across runs.
Length-controlled AlpacaEval (GLM debiasing) is not implemented; raw and logprob-weighted win
rates are reported."""
import hashlib
import json
import math
import re

from ..common import RAW_DIR

NEED_REF_CATS = {"math", "reasoning", "coding"}
_PROMPTS = None


def mt_prompts():
    global _PROMPTS
    if _PROMPTS is None:
        p = RAW_DIR / "mt_bench/fastchat/llm_judge/data/judge_prompts.jsonl"
        _PROMPTS = {d["name"]: d for d in map(json.loads, open(p))}
    return _PROMPTS


def mt_bench_judge_messages(item: dict, answers: list[str]) -> list[list[dict]]:
    """Return judge message lists for turn 1 and turn 2 (FastChat single-v1 / single-math-v1)."""
    P = mt_prompts()
    q = [m["content"] for m in item["messages"]]
    ref = item["reference"] or ["", ""]
    use_ref = item["metadata"]["category_mt"] in NEED_REF_CATS
    out = []
    p1 = P["single-math-v1" if use_ref else "single-v1"]
    kw = {"question": q[0], "answer": answers[0]}
    if use_ref:
        kw["ref_answer_1"] = ref[0]
    out.append([{"role": "system", "content": p1["system_prompt"]},
                {"role": "user", "content": p1["prompt_template"].format(**kw)}])
    p2 = P["single-math-v1-multi-turn" if use_ref else "single-v1-multi-turn"]
    kw = {"question_1": q[0], "answer_1": answers[0], "question_2": q[1], "answer_2": answers[1]}
    if use_ref:
        kw.update(ref_answer_1=ref[0], ref_answer_2=ref[1])
    out.append([{"role": "system", "content": p2["system_prompt"]},
                {"role": "user", "content": p2["prompt_template"].format(**kw)}])
    return out


def parse_rating(text: str):
    m = re.search(r"\[\[(\d+\.?\d*)\]\]", text) or re.search(r"\[(\d+\.?\d*)\]", text)
    return float(m.group(1)) if m else None


_AE_TPL = None


def alpaca_template():
    global _AE_TPL
    if _AE_TPL is None:
        t = open(RAW_DIR / "alpaca_eval_judge/src/alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/"
                           "alpaca_eval_clf.txt").read()
        msgs = []
        for role, body in re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", t, re.S):
            msgs.append({"role": role, "content": body.strip()})
        _AE_TPL = msgs
    return _AE_TPL


def alpaca_judge_messages(item: dict, model_output: str):
    """Model vs baseline with a deterministic per-item order swap. Returns (messages, model_is_m)."""
    swap = int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % 2 == 1
    o1, o2 = (item["reference"], model_output) if swap else (model_output, item["reference"])
    msgs = []
    for m in alpaca_template():
        c = m["content"].replace("{instruction}", item["prompt"]).replace("{output_1}", o1).replace("{output_2}", o2)
        msgs.append({"role": m["role"], "content": c})
    return msgs, (not swap)


def alpaca_preference(choice: dict, raw_resp: dict, model_is_m: bool):
    """P(model output preferred). Uses top logprobs of the first token if present, else the text."""
    try:
        lp = raw_resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        pm = sum(math.exp(x["logprob"]) for x in lp if x["token"].strip() == "m")
        pM = sum(math.exp(x["logprob"]) for x in lp if x["token"].strip() == "M")
        if pm + pM > 0:
            p_m = pm / (pm + pM)
            return p_m if model_is_m else 1 - p_m
    except Exception:
        pass
    t = (choice.get("content") or "").strip()
    if t[:1] in ("m", "M"):
        win_m = t[:1] == "m"
        return float(win_m == model_is_m)
    return None
