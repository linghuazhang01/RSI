"""Task registry: how to turn a processed item into chat requests, how to grade the outputs, and
what an oracle (gold / wrong) response looks like (used by eval/tests/oracle_server.py).

A task produces one or more *units* per item (e.g. SQuAD: one unit per question; MT-Bench: two
sequential turns). Each unit gets n samples; unit score = mean over samples (avg@n), item score
= mean over its units. Records are written per unit with the parent item id (`id`) so paired
bootstrap can resample items.
"""
import json
import re

from .common import strip_thinking
from .graders import basic

TEMPLATES = {}  # filled from configs/tasks.yaml


def render(tpl_name: str, **kw) -> str:
    return TEMPLATES[tpl_name].format(**kw)


def user(content):
    return [{"role": "user", "content": content}]


class Task:
    stages = 1

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.dataset = cfg["dataset"]
        self.category = cfg["category"]

    # --- generation
    def units(self, item, stage=0, prev=None):
        """Return list of (unit_id, request_fields). request_fields has 'messages' (+ 'tools')."""
        raise NotImplementedError

    # --- grading: outputs maps unit_id -> list[choice]; returns list of per-unit records
    def grade(self, item, outputs):
        recs = []
        for uid, choices in outputs.items():
            texts = [strip_thinking(c["content"]) for c in choices]
            scores, parsed, metrics = [], [], []
            for t in texts:
                s, p, m = self.grade_text(item, uid, t)
                scores.append(s)
                parsed.append(p)
                metrics.append(m)
            recs.append(self.make_record(item, uid, choices, scores, parsed, metrics))
        return recs

    def grade_text(self, item, uid, text):
        raise NotImplementedError

    def make_record(self, item, uid, choices, scores, parsed, metrics, status=None):
        valid = [s for s in scores if s is not None]
        n = len(choices)
        rec = {
            "id": item["id"], "unit_id": uid, "task": self.name, "dataset": self.dataset,
            "pool": item["split_pool"],
            "raw_output": [c["content"] for c in choices] if n != 1 else (choices[0]["content"] if choices else None),
            "reasoning": ([c.get("reasoning") for c in choices] if n != 1 else choices[0].get("reasoning")) if choices else None,
            "finish_reason": [c.get("finish_reason") for c in choices],
            "parsed_answer": parsed if n != 1 else (parsed[0] if parsed else None),
            "sample_scores": scores,
            "score": (sum(valid) / len(valid)) if valid and len(valid) == len(scores) else None,
            "metrics": metrics if n != 1 else (metrics[0] if metrics else None),
            "status": status or ("ok" if choices else "generation_failed"),
        }
        if not choices:
            rec["score"] = 0.0  # failed generation counts as wrong (recorded in status)
        return rec

    # --- oracle
    def oracle(self, item, uid, mode):
        raise NotImplementedError


# ============================================================================= math
class MathTask(Task):
    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(render(self.cfg.get("template", "math_cot"), problem=item["prompt"]))})]

    def grade_text(self, item, uid, text):
        s, p = basic.grade_math(text, item["reference"])
        return s, p, None

    def oracle(self, item, uid, mode):
        ref = item["reference"]
        ref = ", ".join(ref) if isinstance(ref, list) else ref
        if mode == "wrong":
            return {"content": "Reasoning... so the answer is \\boxed{-987654.321}"}
        return {"content": f"Working it out step by step.\nTherefore the final answer is \\boxed{{{ref}}}."}


# ============================================================================= MCQ / QA
class MCQTask(Task):
    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(render(self.cfg.get("template", "mcq"), question=item["prompt"],
                                               letters=basic.LETTERS[:item["metadata"]["n_options"]]))})]

    def grade_text(self, item, uid, text):
        s, p = basic.grade_mcq(text, item["reference"], item["metadata"]["n_options"])
        return s, p, None

    def oracle(self, item, uid, mode):
        g = item["reference"]
        if mode == "wrong":
            g = basic.LETTERS[(basic.LETTERS.index(g) + 1) % item["metadata"]["n_options"]]
        return {"content": f"Let me think. The answer is ({g})"}


class ShortQATask(Task):
    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(render(self.cfg.get("template", "short_qa"), question=item["prompt"]))})]

    def grade_text(self, item, uid, text):
        em, p, m = basic.grade_short(text, item["reference"])
        return em, p, m

    def oracle(self, item, uid, mode):
        return {"content": "Answer: " + ("zqxj vortlemb" if mode == "wrong" else item["reference"][0])}


class SquadClosedBookTask(Task):
    """Knowledge injection eval: answer each question of a passage WITHOUT the passage (the passage
    is supposed to be in the LoRA). One unit per question; unit id = SQuAD question id."""

    def units(self, item, stage=0, prev=None):
        return [(q["qid"], {"messages": user(render(self.cfg.get("template", "squad_closed_book"),
                                                    title=item["metadata"]["title"].replace("_", " "),
                                                    question=q["question"]))})
                for q in item["reference"]]

    def grade_text(self, item, uid, text):
        q = next(q for q in item["reference"] if q["qid"] == uid)
        em, p, m = basic.grade_short(text, q["answers"])
        return em, p, m

    def oracle(self, item, uid, mode):
        q = next(q for q in item["reference"] if q["qid"] == uid)
        return {"content": "Answer: " + ("zqxj vortlemb" if mode == "wrong" else q["answers"][0])}


# ============================================================================= instruction following
class IFTask(Task):
    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(item["prompt"])})]

    def grade_text(self, item, uid, text):
        from .graders import ifeval as G
        fn = G.grade_ifeval if self.cfg["grader"] == "ifeval" else G.grade_ifbench
        r = fn(item["prompt"], text, item["reference"])
        return r["prompt_strict"], None, r

    def oracle(self, item, uid, mode):
        from pathlib import Path
        fx = json.load(open(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "if_handmade.json"))
        e = fx.get(item["id"])
        if e is None:
            return {"content": "(no hand-made oracle response for this item)"}
        return {"content": e["pass" if mode == "gold" else "fail"]}


class RLVRIFTask(Task):
    """IF training set `ifeval_train` (RLVR-IFeval): one constraint per prompt, verified by
    graders/rlvr_if.py. Oracle: hand-made pass/fail responses (tests/fixtures/rlvr_if_handmade.json)
    where available; other items (e.g. drawn pool items) are not oracle-gradable, like IFEval."""

    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(item["prompt"])})]

    def grade_text(self, item, uid, text):
        from .graders.rlvr_if import grade_rlvr_if
        r = grade_rlvr_if(text, item["reference"])
        r["constraint_type"] = item["metadata"].get("constraint_type")
        return r["score"], None, r

    def oracle(self, item, uid, mode):
        from pathlib import Path
        fx = json.load(open(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rlvr_if_handmade.json"))
        e = fx.get(item["id"])
        if e is None:
            return {"content": "(no hand-made oracle response for this item)"}
        return {"content": e["pass" if mode == "gold" else "fail"]}


# ============================================================================= code
class CodeTask(Task):
    def units(self, item, stage=0, prev=None):
        g = self.cfg["grader"]
        if g == "humanevalplus":
            msgs = user(render("humaneval", prompt=item["prompt"].strip()))
        elif g == "mbppplus":
            msgs = user(render("mbpp", prompt=item["prompt"].strip(), test=item["metadata"]["test_list"][0]))
        else:  # livecodebench / apps: official LCB generic prompt (lcb_runner/prompts/code_generation.py)
            m = item["metadata"]
            q = f"### Question:\n{item['prompt']}\n\n"
            if m["starter_code"]:
                q += ("### Format: You will use the following starter code to write the solution to the problem "
                      "and enclose your code within delimiters.\n")
                q += f"```python\n{m['starter_code']}\n```\n\n"
            else:
                q += ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do "
                      "not directly test on the sample inputs). Enclose your code within delimiters as follows. "
                      "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
                      "output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
            q += "### Answer: (use the provided format with backticks)\n\n"
            msgs = [{"role": "system", "content": TEMPLATES["lcb_system"]}, {"role": "user", "content": q}]
        return [("0", {"messages": msgs})]

    def grade(self, item, outputs):
        from .graders.code import extract_solution
        recs = []
        for uid, choices in outputs.items():
            codes = [extract_solution(self.cfg["grader"], item, strip_thinking(c["content"])) for c in choices]
            r = self.make_record(item, uid, choices, [None] * len(choices), codes, [None] * len(choices),
                                 status="pending_sandbox")
            r["_codes"] = codes
            recs.append(r)
        return recs

    def oracle(self, item, uid, mode):
        if mode == "wrong":
            return {"content": "```python\ndef solution():\n    return None\n```"}
        if self.cfg["grader"] == "humanevalplus":
            code = item["prompt"] + item["reference"]
        elif self.cfg["grader"] in ("mbppplus", "apps"):
            code = item["reference"]
        else:
            code = "# no gold solution shipped with LiveCodeBench\n"
        return {"content": f"Here is the solution.\n```python\n{code}\n```"}


# ============================================================================= tool calling
class BFCLTask(Task):
    def units(self, item, stage=0, prev=None):
        from .graders.bfcl import SUPPORTED, build_tools
        cat = item["metadata"]["bfcl_category"]
        if cat not in SUPPORTED:
            return []
        tools, _ = build_tools(item["metadata"]["function"], cat)
        return [("0", {"messages": item["messages"][0], "tools": tools})]

    def grade(self, item, outputs):
        from .graders.bfcl import grade_bfcl
        if not outputs:
            r = self.make_record(item, "0", [], [None], [None], [None], status="not_supported_use_official_cli")
            r["score"] = None
            r["bfcl_category"] = item["metadata"]["bfcl_category"]
            return [r]
        recs = []
        for uid, choices in outputs.items():
            scores, parsed, metrics = [], [], []
            for c in choices:
                s, info = grade_bfcl(item, c)
                scores.append(s)
                parsed.append(info.get("decoded"))
                metrics.append({k: v for k, v in info.items() if k != "decoded"})
            r = self.make_record(item, uid, choices, scores, parsed, metrics)
            r["tool_calls"] = [c.get("tool_calls") for c in choices]
            r["bfcl_category"] = item["metadata"]["bfcl_category"]
            r["decision"] = [m.get("decision") for m in metrics]
            recs.append(r)
        return recs

    def oracle(self, item, uid, mode):
        from .graders.bfcl import gold_tool_calls
        cat = item["metadata"]["bfcl_category"]
        funcs = item["metadata"]["function"] or [{"name": "some_function"}]
        fn0 = funcs[0]["name"].replace(".", "_")
        wrong_call = [{"id": "call_0", "type": "function", "function": {"name": fn0, "arguments": "{}"}}]
        if "irrelevance" in cat:
            return {"content": "None of the provided functions can answer this."} if mode == "gold" \
                else {"content": "", "tool_calls": wrong_call}
        if cat == "live_relevance":
            return {"content": "", "tool_calls": wrong_call} if mode == "gold" else {"content": "I cannot help."}
        if mode == "wrong":
            return {"content": "", "tool_calls": [{"id": "call_0", "type": "function",
                                                   "function": {"name": "nonexistent_function", "arguments": "{}"}}]}
        return {"content": "", "tool_calls": gold_tool_calls(item)}


class TEvalTask(Task):
    def units(self, item, stage=0, prev=None):
        subs = self.cfg.get("subsets")
        if subs and item["metadata"]["subset"] not in subs:
            return []
        return [("0", {"messages": item["messages"]})]

    def grade(self, item, outputs):
        if not outputs:
            return []
        return super().grade(item, outputs)

    def grade_text(self, item, uid, text):
        from .graders.tools import grade_teval
        s, m = grade_teval(item, text)
        return s, None, m

    def oracle(self, item, uid, mode):
        from .graders.tools import teval_gold_prediction
        return {"content": "{}" if mode == "wrong" else teval_gold_prediction(item)}


class MetaToolTask(Task):
    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(item["prompt"])})]

    def grade_text(self, item, uid, text):
        from .graders.tools import grade_metatool
        s, info = grade_metatool(item, text)
        return s, info.get("parsed"), None

    def oracle(self, item, uid, mode):
        t, ref = item["metadata"]["task"], item["reference"]
        if t == "Task1":
            ans = ref if mode == "gold" else ("no" if ref == "yes" else "yes")
            return {"content": f"{ans.capitalize()}. Explanation omitted."}
        if mode == "wrong":
            from .graders.tools import metatool_candidates
            if ref == "None":
                return {"content": "Tool: " + (metatool_candidates(item["prompt"]) or ["X"])[0]}
            return {"content": "I would not pick any specific tool; maybe."}
        if t == "Task2-Subtask4":
            return {"content": f"Tools: {ref[0]}, {ref[1]}. They cover both needs."}
        return {"content": f"{ref}. This is the most suitable choice."}


class SealToolTask(Task):
    """Seal-Tools (official finetune prompt: api_list + task_instruction; answer = api_calling list)."""

    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(item["prompt"])})]

    def grade_text(self, item, uid, text):
        from .graders.tools import grade_sealtool
        s, info = grade_sealtool(item, text)
        return s, info.pop("parsed"), info

    def oracle(self, item, uid, mode):
        if mode == "wrong":
            return {"content": "[{'api': 'nonexistentApi', 'parameters': {}, 'responses': ['API_call_0']}]"}
        return {"content": str(item["reference"])}


class APIBankTask(Task):
    def units(self, item, stage=0, prev=None):
        if item["metadata"]["task_type"] != "api_call":
            return []
        return [("0", {"messages": user(item["prompt"])})]

    def grade(self, item, outputs):
        if not outputs:
            return []
        return super().grade(item, outputs)

    def grade_text(self, item, uid, text):
        from .graders.tools import grade_apibank
        s, info = grade_apibank(item, text)
        return s, info.get("parsed"), None

    def oracle(self, item, uid, mode):
        return {"content": "API-Request: [WrongApi(x='1')]" if mode == "wrong" else item["reference"]}


# ============================================================================= judge (optional)
class MTBenchTask(Task):
    stages = 2

    def units(self, item, stage=0, prev=None):
        turns = [m["content"] for m in item["messages"]]
        if stage == 0:
            return [("turn1", {"messages": user(turns[0])})]
        a1 = prev.get("turn1")
        a1 = strip_thinking(a1[0]["content"]) if a1 else ""
        return [("turn2", {"messages": [{"role": "user", "content": turns[0]},
                                        {"role": "assistant", "content": a1},
                                        {"role": "user", "content": turns[1]}]})]

    def grade(self, item, outputs):
        recs = []
        for uid, choices in outputs.items():
            r = self.make_record(item, uid, choices, [None] * len(choices), [None] * len(choices),
                                 [None] * len(choices), status="needs_judge")
            recs.append(r)
        return recs

    def oracle(self, item, uid, mode):
        return {"content": "A helpful, well-structured answer." if mode == "gold" else "no."}


class AlpacaEvalTask(Task):
    def units(self, item, stage=0, prev=None):
        return [("0", {"messages": user(item["prompt"])})]

    def grade(self, item, outputs):
        return [self.make_record(item, uid, ch, [None] * len(ch), [None] * len(ch), [None] * len(ch),
                                 status="needs_judge") for uid, ch in outputs.items()]

    def oracle(self, item, uid, mode):
        return {"content": "A helpful, well-structured answer." if mode == "gold" else "no."}


KIND = {"math": MathTask, "mcq": MCQTask, "short_qa": ShortQATask, "squad_closed_book": SquadClosedBookTask,
        "ifeval": IFTask, "ifbench": IFTask, "rlvr_if": RLVRIFTask, "humanevalplus": CodeTask, "mbppplus": CodeTask,
        "livecodebench": CodeTask, "apps": CodeTask, "sealtool": SealToolTask, "bfcl": BFCLTask, "teval": TEvalTask, "metatool": MetaToolTask,
        "apibank": APIBankTask, "mt_bench_judge": MTBenchTask, "alpaca_eval_judge": AlpacaEvalTask}


def eval_target_tasks(cfg: dict) -> dict:
    """Task entries of the eval-set targets (configs/tasks.yaml `eval_targets`, objective eval_overfit): key K ->
    task K_eval = the first source task's entry with dataset K_eval and allowed_pools [pool]. Entries `tasks`
    defines itself (math_eval) are left out."""
    tasks, out = cfg.get("tasks") or {}, {}
    for key, sources in (cfg.get("eval_targets") or {}).items():
        name = f"{key}_eval"
        if name in tasks:
            continue
        src = [tasks[t] for t, _ in sources]
        if len({s["grader"] for s in src}) != 1:
            raise ValueError(f"eval target {key}: sources must share one grader")
        out[name] = dict(src[0], dataset=name, allowed_pools=["pool"])
    return out


def load_tasks(cfg: dict) -> dict:
    TEMPLATES.clear()
    TEMPLATES.update(cfg["templates"])
    entries = dict(cfg["tasks"], **eval_target_tasks(cfg))
    return {name: KIND[t["grader"]](name, t) for name, t in entries.items()}
