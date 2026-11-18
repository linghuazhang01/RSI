"""Write data/SAMPLES.md: 10 random items from each training set (draw pool) and 3 from each held-out
evaluation set (test pool), grouped by domain. Deterministic (seed 20261118). Long texts are truncated.
GPQA (gated; the authors ask that questions not be shown in plain text) is listed without examples.

Run: eval/.venv/bin/python -I data/scripts/show_samples.py
"""
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PROC = DATA / "processed"
SEED = 20261118

DOMAINS = [
    ("数学", [("gsm8k_train", "pool"), ("math_train", "pool"), ("dapo_math", "pool")],
             [("math", "test", "MATH-500"), ("gsm8k", "test", None), ("aime24", "test", None), ("aime25", "test", None)]),
    ("代码", [("taco_verified", "pool")],
             [("humanevalplus", "test", None), ("mbppplus", "test", None), ("livecodebench", "test", None)]),
    ("通用推理 / 知识", [("triviaqa_train", "pool"), ("arc", "pool"), ("gpqa_main", "gate")],
             [("gpqa_diamond", "test", None), ("mmlu_pro", "test", None), ("triviaqa", "test", None)]),
    ("指令遵循", [("ifeval_train", "pool"), ("ifeval", "gate")],
             [("ifeval", "test", None), ("ifbench", "test", None)]),
    ("工具调用", [("sealtool_train", "pool")],
             [("sealtool", "test", None), ("apibank", "test", None), ("bfcl", "test", None)]),
]
GATED = {"gpqa_main", "gpqa_diamond"}


def load(ds):
    with open(PROC / f"{ds}.jsonl") as f:
        return [json.loads(l) for l in f]


def cut(s, n):
    s = "" if s is None else str(s)
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + f" …（共 {len(s)} 字符，已截断）"


def text_of(r):
    if r.get("prompt"):
        return r["prompt"]
    m = r.get("messages")
    if isinstance(m, list) and m and isinstance(m[0], list):  # BFCL: list of turns
        m = m[0]
    if isinstance(m, list):
        return "\n".join(f"[{x.get('role')}] {x.get('content')}" for x in m if isinstance(x, dict))
    return json.dumps(m, ensure_ascii=False)


def ref_of(ds, r):
    ref, md = r["reference"], r["metadata"]
    if ds in ("ifeval_train",):
        return json.dumps(ref, ensure_ascii=False)
    if ds in ("ifeval", "ifbench"):
        return json.dumps(ref, ensure_ascii=False)
    if ds == "bfcl":
        funcs = [f.get("name") for f in (md.get("function") or [])]
        return f"类别 {md.get('bfcl_category')}；可用函数 {funcs}；标准答案 {json.dumps(ref, ensure_ascii=False)}"
    if ds == "livecodebench":
        return f"无参考解；公开测试示例 {json.dumps(md.get('public_test_cases', [])[:1], ensure_ascii=False)}"
    if ds == "taco_verified":
        extra = f"（{md.get('source')}，难度 {md.get('difficulty')}，{md.get('n_tests')} 个测试" + \
                (f"，函数式 {md.get('fn_name')}" if md.get("fn_name") else "，标准输入输出") + "）"
        return extra + "\n参考解：\n" + str(ref)
    if isinstance(ref, (list, dict)):
        return json.dumps(ref, ensure_ascii=False)
    return str(ref)


def block(ds, r, plen, rlen):
    code = ds in ("taco_verified", "humanevalplus", "mbppplus")
    lines = [f"**{r['id']}**", "", "题目：", "````", cut(text_of(r), plen), "````", "答案 / 判分依据：",
             "````" + ("python" if code else ""), cut(ref_of(ds, r), rlen), "````", ""]
    return "\n".join(lines)


def main():
    rng = random.Random(SEED)
    out = ["# 数据样例（训练集每个 10 道，评测集每个 3 道）", "",
           "由 `data/scripts/show_samples.py` 生成（随机种子 20261118）。训练集从抽样池（`pool`，GPQA Main 与 IFEval 从验收集 `gate`）里抽，"
           "评测集从留出池（`test`）里抽。长文本已截断。GPQA 按数据作者要求不展示题目。", ""]
    for dom, trains, evals in DOMAINS:
        out += [f"## {dom}", ""]
        for ds, pool in trains:
            rows = [r for r in load(ds) if r["split_pool"] == pool]
            out += [f"### 训练集：{ds}（{pool}，{len(rows)} 条）", ""]
            if ds in GATED:
                out += ["（GPQA：受访问条款限制，不展示题目。）", ""]
                continue
            k = 10 if ds != "ifeval" else 3
            for r in rng.sample(sorted(rows, key=lambda x: x["id"]), k):
                out.append(block(ds, r, 700, 600))
        for ds, pool, name in evals:
            rows = [r for r in load(ds) if r["split_pool"] == pool]
            if ds == "bfcl":
                rows = [r for r in rows if r["metadata"].get("group") == "single_turn"]
            if ds == "apibank":
                rows = [r for r in rows if r["metadata"].get("task_type") == "api_call"]
            out += [f"### 评测集：{name or ds}（{pool}，{len(rows)} 条）", ""]
            if ds in GATED:
                out += ["（GPQA：受访问条款限制，不展示题目。）", ""]
                continue
            for r in rng.sample(sorted(rows, key=lambda x: x["id"]), 3):
                out.append(block(ds, r, 900, 500))
    p = DATA / "SAMPLES.md"
    p.write_text("\n".join(out))
    print(f"wrote {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
