"""Normalize raw downloads into data/processed/<dataset>.jsonl (unified schema).

Unified schema (one JSON object per line):
  id         stable item id (official id when available, else "<dataset>/<split>/<row index>")
  dataset    processed dataset name (== file stem)
  category   math | code | knowledge | instruction_following | tool_calling | knowledge_injection
  split_pool null here; filled by make_splits.py (search | gate | test | reserve | train)
  prompt     user-facing task text (task-agnostic; eval/configs/tasks.yaml adds instructions), or null
  messages   list of chat messages when the item is natively multi-turn / has a system prompt, or null
  reference  gold answer(s) in a dataset-specific but documented shape
  metadata   everything else needed for grading / stratification

Run:  eval/.venv/bin/python -I data/scripts/normalize.py [--only a,b]
Nothing from data/raw is executed; only json/csv/parquet parsing.
"""
import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

DATA = Path(__file__).resolve().parents[1]
RAW = DATA / "raw"
OUT = DATA / "processed"
CFG = yaml.safe_load(open(DATA / "splits" / "split_config.yaml"))

csv.field_size_limit(1 << 30)


def h(s: str, n=16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def rec(id, dataset, category, prompt=None, messages=None, reference=None, **metadata):
    return {"id": str(id), "dataset": dataset, "category": category, "split_pool": None,
            "prompt": prompt, "messages": messages, "reference": reference, "metadata": metadata}


def tolist(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def jsonable(o):
    """Recursively convert numpy containers from parquet into plain python."""
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        o = o.tolist()
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def last_boxed(s: str):
    """Return content of the last \\boxed{...} / \\fbox{...} in s (brace-matched)."""
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


# ----------------------------------------------------------------------------- math
def gsm8k():
    out = {}
    for split, name in [("test", "gsm8k"), ("train", "gsm8k_train")]:
        df = pd.read_parquet(RAW / f"gsm8k/main/{split}-00000-of-00001.parquet")
        rows = []
        for i, r in df.iterrows():
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            rows.append(rec(f"gsm8k/{split}/{i}", name, "math", prompt=r["question"], reference=ans,
                            solution=r["answer"]))
        out[name] = rows
    return out


MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def math():
    m500 = [json.loads(l) for l in open(RAW / "math500/test.jsonl")]
    m500_by_problem = {norm_ws(x["problem"]): x for x in m500}
    out = {}
    for split, name in [("test", "math"), ("train", "math_train")]:
        rows = []
        for subj in MATH_SUBJECTS:
            df = pd.read_parquet(RAW / f"math/{subj}/{split}-00000-of-00001.parquet")
            for i, r in df.iterrows():
                ans = last_boxed(r["solution"])
                meta = dict(level=r["level"], type=r["type"], subject=subj, solution=r["solution"])
                if split == "test":
                    hit = m500_by_problem.get(norm_ws(r["problem"]))
                    meta["is_math500"] = hit is not None
                    if hit is not None:
                        meta["math500_unique_id"] = hit["unique_id"]
                        ans = hit["answer"]  # MATH-500 ships a curated final answer
                rows.append(rec(f"math/{split}/{subj}/{i}", name, "math", prompt=r["problem"],
                                reference=ans, **meta))
        out[name] = rows
    n500 = sum(r["metadata"].get("is_math500", False) for r in out["math"])
    assert n500 == 500, f"MATH-500 matched {n500} != 500"
    return out


def aime_amc():
    out = {}
    df = pd.read_parquet(RAW / "aime24/data/train-00000-of-00001.parquet")
    out["aime24"] = [rec(f"aime24/{r['id']}", "aime24", "math", prompt=r["problem"], reference=str(r["answer"]),
                         url=r["url"]) for _, r in df.iterrows()]
    df = pd.read_parquet(RAW / "aime25/data/train-00000-of-00001.parquet")
    out["aime25"] = [rec(f"aime25/{r['problem_idx']}", "aime25", "math", prompt=r["problem"],
                         reference=str(r["answer"]), problem_type=str(r["problem_type"]))
                     for _, r in df.iterrows()]
    df = pd.read_parquet(RAW / "amc23/test-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        a = str(r["answer"])
        a = a[:-2] if a.endswith(".0") else a
        rows.append(rec(f"amc23/{r['id']}", "amc23", "math", prompt=r["question"], reference=a, url=r["url"]))
    out["amc23"] = rows
    df = pd.read_parquet(RAW / "olympiadbench/OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet")
    rows = []
    for _, r in df.iterrows():
        fa = tolist(r["final_answer"])
        if isinstance(fa, str):
            try:
                fa = ast.literal_eval(fa)
            except Exception:
                fa = [fa]
        rows.append(rec(f"olympiadbench/{r['id']}", "olympiadbench", "math", prompt=r["question"],
                        reference=[str(x) for x in fa], is_multiple_answer=str(r["is_multiple_answer"]),
                        unit=None if str(r["unit"]) in ("nan", "None") else str(r["unit"]),
                        answer_type=str(r["answer_type"]), subfield=str(r["subfield"]),
                        context=None if str(r["context"]) in ("nan", "None") else str(r["context"])))
    out["olympiadbench"] = rows
    return out


# ----------------------------------------------------------------------------- code
def code():
    out = {}
    df = pd.read_parquet(RAW / "humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")
    out["humanevalplus"] = [rec(r["task_id"], "humanevalplus", "code", prompt=r["prompt"],
                                reference=r["canonical_solution"], entry_point=r["entry_point"])
                            for _, r in df.iterrows()]
    df = pd.read_parquet(RAW / "mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet")
    rows = []
    for _, r in df.iterrows():
        tl = tolist(r["test_list"])
        rows.append(rec(f"Mbpp/{r['task_id']}", "mbppplus", "code", prompt=r["prompt"], reference=r["code"],
                        test_list=jsonable(tl), test_imports=jsonable(tolist(r["test_imports"]))))
    out["mbppplus"] = rows
    # LiveCodeBench: keep only problems inside the configured window; tests stay in raw (large).
    start = str(CFG["datasets"]["livecodebench"]["window_start"])
    end = str(CFG["datasets"]["livecodebench"].get("window_end") or "9999-12-31")
    rows = []
    raw = RAW / "livecodebench/test6.jsonl"
    with open(raw) as f:
        for lineno, line in enumerate(f):
            r = json.loads(line)
            d = r["contest_date"][:10]
            if not (start <= d <= end):
                continue
            rows.append(rec(f"lcb/{r['question_id']}", "livecodebench", "code", prompt=r["question_content"],
                            reference=None, question_title=r["question_title"], platform=r["platform"],
                            contest_id=r["contest_id"], contest_date=r["contest_date"],
                            starter_code=r["starter_code"], difficulty=r["difficulty"],
                            public_test_cases=json.loads(r["public_test_cases"]),
                            tests_file="livecodebench/test6.jsonl", tests_line=lineno,
                            lcb_metadata=json.loads(r["metadata"]) if r["metadata"] else {}))
    out["livecodebench"] = rows
    return out


# ----------------------------------------------------------------------------- knowledge
LETTERS = "ABCDEFGHIJ"


def fmt_mcq(q, options):
    return q.strip() + "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def knowledge():
    out = {}
    df = pd.read_parquet(RAW / "mmlu_pro/data/test-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        opts = [str(o) for o in tolist(r["options"])]
        assert LETTERS[int(r["answer_index"])] == r["answer"]
        rows.append(rec(f"mmlu_pro/{r['question_id']}", "mmlu_pro", "knowledge", prompt=fmt_mcq(r["question"], opts),
                        reference=r["answer"], n_options=len(opts), subject=r["category"], src=r["src"]))
    out["mmlu_pro"] = rows

    # TriviaQA rc.nocontext: validation -> eval pools; train -> train
    for split, name in [("validation", "triviaqa"), ("train", "triviaqa_train")]:
        df = pd.read_parquet(RAW / f"triviaqa/rc.nocontext/{split}-00000-of-00001.parquet",
                             columns=["question", "question_id", "question_source", "answer"])
        # The HF rc.* configs contain each question once per evidence domain (web / wikipedia),
        # so question_id repeats; keep the first occurrence of each question_id.
        n_raw = len(df)
        df = df.drop_duplicates(subset="question_id", keep="first")
        print(f"  triviaqa {split}: {n_raw} rows -> {len(df)} unique question_id")
        rows = []
        for _, r in df.iterrows():
            a = r["answer"]
            aliases = list(dict.fromkeys([a["value"]] + list(tolist(a["aliases"]))))
            rows.append(rec(f"triviaqa/{r['question_id']}", name, "knowledge", prompt=r["question"],
                            reference=aliases, normalized_aliases=list(tolist(a["normalized_aliases"])),
                            answer_type=a["type"]))
        out[name] = rows

    d = json.load(open(RAW / "freebaseqa/FreebaseQA-eval.json"))
    rows = []
    for q in d["Questions"]:
        ans = []
        for p in q["Parses"]:
            for a in p["Answers"]:
                ans.extend(a["AnswersName"])
        ans = list(dict.fromkeys(ans))
        rows.append(rec(q["Question-ID"], "freebaseqa", "knowledge", prompt=q["RawQuestion"], reference=ans))
    out["freebaseqa"] = rows

    gp = RAW / "gpqa_diamond/gpqa_diamond.csv"
    if gp.exists():  # manual download only; see data/README.md
        import random
        rows = []
        with open(gp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row.get("Record ID") or h(row["Question"])
                opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                        row["Incorrect Answer 3"]]
                perm = list(range(4))
                random.Random(f"20261118-{rid}").shuffle(perm)
                shuffled = [opts[i].strip() for i in perm]
                gold = LETTERS[perm.index(0)]
                rows.append(rec(f"gpqa/{rid}", "gpqa_diamond", "knowledge", prompt=fmt_mcq(row["Question"], shuffled),
                                reference=gold, n_options=4, subdomain=row.get("Subdomain")))
        out["gpqa_diamond"] = rows
    return out


# ----------------------------------------------------------------------------- instruction following
def instr():
    out = {}
    rows = []
    for l in open(RAW / "ifeval/ifeval_input_data.jsonl"):
        r = json.loads(l)
        rows.append(rec(f"ifeval/{r['key']}", "ifeval", "instruction_following", prompt=r["prompt"],
                        reference={"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]},
                        key=r["key"]))
    out["ifeval"] = rows
    df = pd.read_parquet(RAW / "ifbench/data/train-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        kw = [{k: v for k, v in jsonable(x).items() if v is not None} for x in tolist(r["kwargs"])]
        rows.append(rec(f"ifbench/{r['key']}", "ifbench", "instruction_following", prompt=r["prompt"],
                        reference={"instruction_id_list": list(tolist(r["instruction_id_list"])), "kwargs": kw},
                        key=str(r["key"])))
    out["ifbench"] = rows
    refs = {}
    for l in open(RAW / "mt_bench/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"):
        r = json.loads(l)
        refs[r["question_id"]] = r["choices"][0]["turns"]
    rows = []
    for l in open(RAW / "mt_bench/fastchat/llm_judge/data/mt_bench/question.jsonl"):
        r = json.loads(l)
        rows.append(rec(f"mt_bench/{r['question_id']}", "mt_bench", "instruction_following", prompt=None,
                        messages=[{"role": "user", "content": t} for t in r["turns"]],
                        reference=refs.get(r["question_id"]), category_mt=r["category"],
                        question_id=r["question_id"], judge_optional=True))
    out["mt_bench"] = rows
    data = json.load(open(RAW / "alpaca_eval/alpaca_eval.json"))
    base = json.load(open(RAW / "alpaca_eval/alpaca_eval_gpt4_baseline.json"))
    assert len(base) == len(data)
    rows = []
    for i, (r, b) in enumerate(zip(data, base)):
        assert r["instruction"] == b["instruction"]
        rows.append(rec(f"alpaca_eval/{i}", "alpaca_eval", "instruction_following", prompt=r["instruction"],
                        reference=b["output"], baseline_generator=b["generator"], subset=r["dataset"],
                        judge_optional=True))
    out["alpaca_eval"] = rows
    return out


# ----------------------------------------------------------------------------- tool calling
BFCL_SINGLE_TURN = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
                    "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
                    "live_parallel_multiple", "irrelevance", "live_irrelevance", "live_relevance"]
BFCL_MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
BFCL_AGENTIC = ["memory", "web_search"]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def tools():
    out = {}
    rows = []
    for cat in BFCL_SINGLE_TURN + BFCL_MULTI_TURN + BFCL_AGENTIC:
        items = read_jsonl(RAW / f"bfcl/BFCL_v4_{cat}.json")
        pa_p = RAW / f"bfcl/possible_answer/BFCL_v4_{cat}.json"
        pa = {x["id"]: x["ground_truth"] for x in read_jsonl(pa_p)} if pa_p.exists() else {}
        for x in items:
            meta = {k: v for k, v in x.items() if k not in ("id", "question")}
            meta["bfcl_category"] = cat
            meta["group"] = ("single_turn" if cat in BFCL_SINGLE_TURN else
                             "multi_turn" if cat in BFCL_MULTI_TURN else "agentic")
            rows.append(rec(x["id"], "bfcl", "tool_calling", messages=x["question"],
                            reference=pa.get(x["id"]), **meta))
    out["bfcl"] = rows

    rows = []
    for sub in ["instruct_v2", "plan_json_v2", "plan_str_v2", "reason_retrieve_understand_json_v2",
                "reason_str_v2", "retrieve_str_v2", "review_str_v2", "understand_str_v2"]:
        d = json.load(open(RAW / f"teval/data/{sub}.json"))
        for k, v in d.items():
            rows.append(rec(f"teval/{sub}/{k}", "teval", "tool_calling", messages=v["origin_prompt"],
                            reference=v["ground_truth"], subset=sub, key=k,
                            template=v.get("template"), meta_data=v.get("meta_data", v.get("meta"))))
    out["teval"] = rows

    rows = []
    for task in ["Task1", "Task2-Subtask1", "Task2-Subtask2", "Task2-Subtask3", "Task2-Subtask4"]:
        d = json.load(open(RAW / f"metatool/dataset/tmp_dataset/{task}.json"))
        for i, x in enumerate(d):
            if task == "Task1":
                prompt, ref = x["thought_prompt"], ("yes" if x["label"] == "positive" else "no")
            else:
                prompt = x["action_prompt"].strip()
                if prompt.startswith('"') and prompt.endswith('"'):
                    prompt = prompt[1:-1]
                tool = x["tool"]
                if task == "Task2-Subtask4":
                    ref = list(tool) if isinstance(tool, list) else ast.literal_eval(tool)
                elif task == "Task2-Subtask3" and tool not in re.findall(
                        r"^\s*\d+\.\s*tool name:\s*(.+?),\s*tool description:", prompt, re.M):
                    ref = "None"  # reliability subtask: correct tool absent from the candidate list
                else:
                    ref = tool
            rows.append(rec(f"metatool/{task}/{i}", "metatool", "tool_calling", prompt=prompt, reference=ref,
                            task=task, query=x["query"], gold_tool=x.get("tool"), label=x.get("label")))
    out["metatool"] = rows

    rows = []
    for lvl, fn in [("level-1", "level-1-api.json"), ("level-2", "level-2-api.json")]:
        d = json.load(open(RAW / f"apibank/test-data/{fn}"))
        for i, x in enumerate(d):
            rows.append(rec(f"apibank/{lvl}-api/{i}", "apibank", "tool_calling",
                            prompt=x["instruction"].strip() + "\n" + x["input"], reference=x["expected_output"],
                            level=lvl, task_type="api_call", source_file=x.get("file"), source_id=x.get("id")))
    for i, x in enumerate(json.load(open(RAW / "apibank/test-data/level-3.json"))):
        rows.append(rec(f"apibank/level-3/{i}", "apibank", "tool_calling", prompt=x["requirement"].strip(),
                        reference={"apis": x["apis"], "response": x["response"]}, level="level-3",
                        task_type="plan_retrieve_call"))
    out["apibank"] = rows

    # training sets
    rows = []
    for lvl in ["lv1", "lv2", "lv3"]:
        d = json.load(open(RAW / f"apibank/training-data/{lvl}-api-train.json"))
        for i, x in enumerate(d):
            rows.append(rec(f"apibank_train/{lvl}/{i}", "apibank_train", "tool_calling",
                            prompt=x["instruction"].strip() + "\n" + x["input"], reference=x["output"], level=lvl))
    out["apibank_train"] = rows
    d = json.load(open(RAW / "toolace/data.json"))
    rows = []
    for i, x in enumerate(d):
        msgs = [{"role": {"user": "user", "assistant": "assistant", "tool": "tool"}[c["from"]], "content": c["value"]}
                for c in x["conversations"]]
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        rows.append(rec(f"toolace/{i}", "toolace", "tool_calling", prompt=first_user,
                        messages=[{"role": "system", "content": x["system"]}] + msgs, reference=None))
    out["toolace"] = rows
    return out


# ----------------------------------------------------------------------------- training sets (code / IF)
def train_sets():
    out = {}
    rows = []
    with open(RAW / "magicoder_oss/data-oss_instruct-decontaminated.jsonl") as f:
        for lineno, l in enumerate(f):  # 'index' is not unique across languages -> use line number
            x = json.loads(l)
            rows.append(rec(f"magicoder/{lineno}", "magicoder_oss", "code", prompt=x["problem"],
                            reference=x["solution"], lang=x["lang"], index=x["index"], raw_index=x["raw_index"]))
    out["magicoder_oss"] = rows
    df = pd.read_parquet(RAW / "tulu3_if/data/train-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        msgs = jsonable(r["messages"])
        rows.append(rec(r["id"], "tulu3_if", "instruction_following", prompt=r["prompt"], messages=msgs,
                        reference=next((m["content"] for m in msgs if m["role"] == "assistant"), None),
                        constraints=list(tolist(r["constraints"]))))
    out["tulu3_if"] = rows
    return out


# ----------------------------------------------------------------------------- IF training set (RLVR-IFeval)
def _rlvr_constraint_text(template: str, gt: dict) -> str:
    """Instantiate the plain-English constraint template of an RLVR-IFeval row with its arguments
    (exactly as it appears inside the prompt; checked for every row)."""
    s = template
    if gt.get("quantifier"):
        s = s.replace("at least / around / at most", gt["quantifier"])
    rep = {"{N}": gt.get("N"), "{word}": gt.get("word"), "{letter}": gt.get("letter"),
           "{end phrase}": gt.get("end_phrase"), "{options}": gt.get("options"),
           "{section splitter}": gt.get("section_splitter"), "{postscript marker}": gt.get("postscript_marker"),
           "{first word}": gt.get("first_word"), "{i}": gt.get("i"),
           "{forbidden words}": ", ".join(gt["forbidden_words"]) if gt.get("forbidden_words") else None,
           "{keyword1}, {keyword2}": ", ".join(gt["keyword_list"]) if gt.get("keyword_list") else None}
    for k, v in rep.items():
        if v is not None:
            s = s.replace(k, str(v))
    return s


def if_train():
    """allenai/RLVR-IFeval -> ifeval_train: one user turn = a Tulu-2 SFT prompt + one IFEval-style
    constraint (prepended or appended). reference = parsed ground_truth (verifier func_name + its
    non-null arguments), graded by eval/evallib/graders/rlvr_if.py. Exact duplicate rows (same
    prompt and ground_truth) are dropped, keeping the first occurrence."""
    df = pd.read_parquet(RAW / "rlvr_ifeval/data/train-00000-of-00001.parquet")
    rows, seen, n_dup = [], set(), 0
    for k, r in df.iterrows():
        msgs = jsonable(r["messages"])
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        prompt = msgs[0]["content"]
        if (prompt, r["ground_truth"]) in seen:
            n_dup += 1
            continue
        seen.add((prompt, r["ground_truth"]))
        gt = {kk: v for kk, v in json.loads(r["ground_truth"]).items() if v is not None}
        ctext = _rlvr_constraint_text(r["constraint"], gt)
        if prompt.startswith(ctext):
            pos, instruction = "prefix", prompt[len(ctext):]
        elif prompt.endswith(ctext):
            pos, instruction = "suffix", prompt[:-len(ctext)]
        else:
            raise AssertionError(f"rlvr_ifeval row {k}: constraint text not at prompt start/end")
        instruction = instruction.strip()
        rows.append(rec(f"rlvr_ifeval/{k}", "ifeval_train", "instruction_following", prompt=prompt,
                        messages=msgs, reference=gt, constraint_type=r["constraint_type"],
                        func_name=gt["func_name"], constraint_template=r["constraint"],
                        constraint_text=ctext, constraint_position=pos, instruction=instruction,
                        instruction_group=h(instruction), source_dataset=r["dataset"]))
    print(f"  ifeval_train: rows={len(df)} exact duplicates dropped={n_dup} kept={len(rows)} "
          f"distinct base instructions={len({x['metadata']['instruction_group'] for x in rows})}")
    return {"ifeval_train": rows}


# ----------------------------------------------------------------------------- SQuAD (knowledge injection)
def squad():
    """One record per passage. id = squad/<sha256(context)[:16]>. Questions are in reference."""
    out = {}
    seal = json.load(open(RAW / "seal_squad_subset/general-knowledge/data/synthetic_data/eval/base_val.json"))
    seal_ctx = [x["context"] for x in seal]
    for split, name in [("validation", "squad_val"), ("train", "squad_train")]:
        df = pd.read_parquet(RAW / f"squad/plain_text/{split}-00000-of-00001.parquet")
        by_ctx = {}
        for _, r in df.iterrows():
            e = by_ctx.setdefault(r["context"], {"title": r["title"], "qs": []})
            e["qs"].append({"qid": r["id"], "question": r["question"],
                            "answers": list(dict.fromkeys(tolist(r["answers"]["text"])))})
        rows = []
        for ctx, e in by_ctx.items():
            rows.append(rec(f"squad/{h(ctx)}", name, "knowledge_injection", prompt=ctx, reference=e["qs"],
                            title=e["title"], n_questions=len(e["qs"]), squad_split=split,
                            in_seal_eval=(ctx in set(seal_ctx)) if split == "validation" else False))
        out[name] = rows
    ids = {r["prompt"]: r["id"] for r in out["squad_val"]}
    assert all(c in ids for c in seal_ctx), "SEAL passage not found in SQuAD validation"
    return out


# ----------------------------------------------------------------------------- RSI-validated training sets (2026-11-18)
DAPO_PRE = ("Solve the following math problem step by step. The last line of your response should be of the form "
            "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
DAPO_SUF = '\n\nRemember to put your answer on its own line after "Answer:".'


def _apps_io_to_lcb(io: dict):
    """APPS input_output -> LiveCodeBench-runner format (inputs/outputs are strings).
    stdin problems: join list-valued inputs/outputs with newlines.
    call-based problems (fn_name): each input = one JSON value per line (one per argument);
    output = JSON of the expected return value (APPS wraps it in a 1-element list)."""
    fn = io.get("fn_name")
    ins, outs = [], []
    for i, o in zip(io["inputs"], io["outputs"]):
        if fn:
            args = i if isinstance(i, list) else [i]
            ins.append("\n".join(json.dumps(a) for a in args))
            outs.append(json.dumps(o[0] if isinstance(o, list) and len(o) == 1 else o))
        else:
            ins.append("\n".join(i) if isinstance(i, list) else str(i))
            outs.append("\n".join(o) if isinstance(o, list) else str(o))
    return {"inputs": ins, "outputs": outs, "fn_name": fn}


def rsi_train():
    out = {}
    # DAPO-Math-17K (WHALE D_weight): dedupe the repeated prompts, strip the instruction template.
    df = pd.read_parquet(RAW / "dapo_math/data/dapo-math-17k.parquet")
    rows, seen = [], set()
    for _, r in df.iterrows():
        t = r["prompt"][0]["content"]
        if t in seen:
            continue
        seen.add(t)
        assert t.startswith(DAPO_PRE) and t.endswith(DAPO_SUF)
        idx = r["extra_info"]["index"]
        rows.append(rec(f"dapo/{idx}", "dapo_math", "math", prompt=t[len(DAPO_PRE):-len(DAPO_SUF)],
                        reference=str(r["reward_model"]["ground_truth"]), source=r["data_source"]))
    out["dapo_math"] = rows

    # APPS train, all difficulties (introductory = ReST-EM's set; interview + competition added
    # 2026-11-18 so the per-generation draw pool is large enough). Keep problems with tests and
    # >=1 reference solution; difficulty is kept in metadata for stratified draws.
    aux = OUT / "aux"
    aux.mkdir(parents=True, exist_ok=True)
    rows, n_seen, n_fn_no_starter = [], 0, 0
    with open(RAW / "apps/train.jsonl") as f, open(aux / "apps_tests.jsonl", "w") as ft:
        for l in f:
            x = json.loads(l)
            n_seen += 1
            try:
                io = json.loads(x["input_output"])
                sols = json.loads(x["solutions"])
            except Exception:
                continue
            if not io.get("inputs") or not sols:
                continue
            lcb = _apps_io_to_lcb(io)
            starter = x["starter_code"] or ""
            if lcb["fn_name"] and not starter.strip():
                n_fn_no_starter += 1
            qid = f"apps/{x['id']}"
            ft.write(json.dumps({"id": qid, "input_output": lcb}) + "\n")
            rows.append(rec(qid, "apps", "code", prompt=x["question"], reference=sols[0],
                            n_solutions=len(sols), difficulty=x["difficulty"], starter_code=starter, fn_name=lcb["fn_name"],
                            n_tests=len(lcb["inputs"]), url=x["url"], tests_file="aux/apps_tests.jsonl"))
    from collections import Counter
    print(f"  apps: train={n_seen} kept(with tests+solution)={len(rows)} "
          f"by difficulty={dict(Counter(r['metadata']['difficulty'] for r in rows))} fn_name_without_starter={n_fn_no_starter}")
    out["apps"] = rows

    # ARC-Easy + ARC-Challenge train (Transformer2 trains on ARC-Easy).
    rows = []
    for sub in ["ARC-Easy", "ARC-Challenge"]:
        df = pd.read_parquet(RAW / f"arc/{sub}/train-00000-of-00001.parquet")
        for _, r in df.iterrows():
            texts = [str(t) for t in tolist(r["choices"]["text"])]
            labels = [str(t) for t in tolist(r["choices"]["label"])]
            gold = LETTERS[labels.index(str(r["answerKey"]))]
            rows.append(rec(f"arc/{sub}/{r['id']}", "arc", "knowledge", prompt=fmt_mcq(r["question"], texts),
                            reference=gold, n_options=len(texts), subset=sub))
    out["arc"] = rows

    # GPQA Main minus Diamond (WSP-Scope dev). Same option shuffling rule as gpqa_diamond.
    gm, gd = RAW / "gpqa_main/gpqa_main.csv", RAW / "gpqa_diamond/gpqa_diamond.csv"
    if gm.exists() and gd.exists():
        import random
        with open(gd, newline="", encoding="utf-8") as f:
            diamond_ids = {row["Record ID"] for row in csv.DictReader(f)}
        rows = []
        with open(gm, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row["Record ID"]
                if rid in diamond_ids:
                    continue
                opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                        row["Incorrect Answer 3"]]
                perm = list(range(4))
                random.Random(f"20261118-{rid}").shuffle(perm)
                rows.append(rec(f"gpqa/{rid}", "gpqa_main", "knowledge",
                                prompt=fmt_mcq(row["Question"], [opts[i].strip() for i in perm]),
                                reference=LETTERS[perm.index(0)], n_options=4, subdomain=row.get("Subdomain")))
        out["gpqa_main"] = rows

    # Seal-Tools (official finetune format: prompt carries the candidate api_list).
    def seal(fn, split):
        d = json.load(open(RAW / f"sealtool/Seal-Tools_Dataset/dataset_for_finetune/{fn}.json"))
        res = []
        for x in d:
            prompt, gold = x["conversations"][0]["value"], x["conversations"][1]["value"]
            calls = ast.literal_eval(gold)
            res.append(rec(f"sealtool/{x['id']}", None, "tool_calling", prompt=prompt, reference=calls,
                           seal_split=split, difficulty=x["id"].split("-")[1], n_calls=len(calls)))
        return res
    tr = seal("train", "train")
    for r in tr:
        r["dataset"] = "sealtool_train"
    out["sealtool_train"] = tr
    ev = seal("dev", "dev") + seal("test_in_domain", "test_in_domain") + seal("test_out_domain", "test_out_domain")
    for r in ev:
        r["dataset"] = "sealtool"
    out["sealtool"] = ev
    return out


# ----------------------------------------------------------------------------- TACO-verified (2026-11-19)
TACO_MIN_TESTS = 1   # decided 2026-11-19: follow EvoTrainer / ReVeal, which train on all verified TACO problems
TACO_MAX_TEST_BYTES = 1 << 20      # cap on the serialized tests of one problem (truncate, keep >= MIN)
TACO_MULTI_ANSWER = re.compile(
    r"print any|output any|any of them|any one of them|any such|if there are (several|multiple|many)"
    r"|multiple (possible |valid )?(answers|solutions)|several (possible |valid )?(answers|solutions)", re.I)


def _grams13(s):
    w = re.findall(r"\w+", s.lower())
    return {" ".join(w[i:i + 13]) for i in range(len(w) - 12)}


def _norm_url(u):
    u = (u or "").strip().lower().rstrip("/")
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u.replace("problemset/problem/", "contest/").replace("/problem/", "/") if u not in ("", "none") else ""


def taco():
    """TACO-verified: the ONLY code training set (decided 2026-11-19; APPS is no longer used, so there is
    no APPS de-duplication). Precedents: EvoTrainer (2606.03108) trains on 11,897 TACO-verified problems
    and evaluates on LiveCodeBench v6; ReVeal (2506.11442) trains on 11,151 TACO problems, LCB v6.
    Filters: tests parse, >= TACO_MIN_TESTS tests (after capping the serialized tests at
    TACO_MAX_TEST_BYTES), >=1 verified solution, no pictures, no 'print any valid answer' style problems
    (no checker), date < 2025-01-01, no duplicate inside TACO (first 200 normalized chars).
    Held-out decontamination (exact + 13-gram) is applied later by make_splits.py."""
    import pyarrow.parquet as pq
    from collections import Counter
    aux = OUT / "aux"
    aux.mkdir(parents=True, exist_ok=True)
    stats, rows, seen_heads = Counter(), [], set()
    f = pq.ParquetFile(RAW / "taco_verified/default/train/0000.parquet")
    cols = ["id", "question", "solutions", "starter_code", "input_output", "difficulty", "source", "url",
            "date", "picture_num", "tags", "time_limit"]
    with open(aux / "taco_verified_tests.jsonl", "w") as ft:
        for batch in f.iter_batches(batch_size=64, columns=cols):
            for x in batch.to_pylist():
                stats["rows"] += 1
                q = x["question"] or ""
                sols = [s for s in (x["solutions"] or []) if s and s.strip()]
                if not sols:
                    stats["drop_no_solution"] += 1
                    continue
                try:
                    io = json.loads(x["input_output"])
                except Exception:
                    stats["drop_bad_tests"] += 1
                    continue
                if not io.get("inputs"):
                    stats["drop_bad_tests"] += 1
                    continue
                lcb = _apps_io_to_lcb(io)
                keep, size = 0, 0
                for i_, o_ in zip(lcb["inputs"], lcb["outputs"]):
                    size += len(i_) + len(o_)
                    if size > TACO_MAX_TEST_BYTES:
                        break
                    keep += 1
                truncated = keep < len(lcb["inputs"])
                lcb = {"inputs": lcb["inputs"][:keep], "outputs": lcb["outputs"][:keep], "fn_name": lcb["fn_name"]}
                if keep < TACO_MIN_TESTS:
                    stats["drop_few_tests"] += 1
                    continue
                pic = str(x.get("picture_num") or "0")
                if (pic not in ("None", "0", "nan", "")) or "<image>" in q:
                    stats["drop_picture"] += 1
                    continue
                if TACO_MULTI_ANSWER.search(q):
                    stats["drop_multi_answer"] += 1
                    continue
                d = x.get("date")
                if d is not None and str(d)[:4] not in ("None", "NaT") and str(d)[:10] >= "2025-01-01":
                    stats["drop_date"] += 1
                    continue
                head = norm_ws(q)[:200].lower()
                url = _norm_url(x.get("url"))
                if head in seen_heads:
                    stats["drop_dup_internal"] += 1
                    continue
                seen_heads.add(head)
                qid = f"taco/{x['id']}"
                ft.write(json.dumps({"id": qid, "input_output": lcb}) + "\n")
                rows.append(rec(qid, "taco_verified", "code", prompt=q, reference=sols[0],
                                n_solutions=len(sols), starter_code=x.get("starter_code") or "",
                                fn_name=lcb["fn_name"], n_tests=keep, tests_truncated=truncated,
                                difficulty=str(x.get("difficulty")), source=str(x.get("source")),
                                url=None if url == "" else x.get("url"), date=None if d is None else str(d)[:10],
                                tests_file="aux/taco_verified_tests.jsonl"))
    stats["kept"] = len(rows)
    print("  taco_verified:", dict(stats))
    print("  by source:", dict(Counter(r["metadata"]["source"] for r in rows)))
    print("  by difficulty:", dict(Counter(r["metadata"]["difficulty"] for r in rows)))
    return {"taco_verified": rows}


BUILDERS = [gsm8k, math, aime_amc, code, knowledge, instr, tools, train_sets, squad, rsi_train, if_train, taco]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    OUT.mkdir(parents=True, exist_ok=True)
    for b in BUILDERS:
        if only and b.__name__ not in only:
            continue
        for name, rows in b().items():
            ids = [r["id"] for r in rows]
            assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
            with open(OUT / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(jsonable(r), ensure_ascii=False) + "\n")
            print(f"{name:20s} {len(rows):7d}")


if __name__ == "__main__":
    sys.exit(main())
