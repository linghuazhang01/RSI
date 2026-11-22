"""Print the markdown dataset table used in data/README.md (source, revision, license, pools, counts).
Run: eval/.venv/bin/python -I data/scripts/report_table.py
"""
import json
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parents[1]
src = yaml.safe_load(open(DATA / "scripts" / "sources.yaml"))
rep = json.load(open(DATA / "splits" / "check_report.json"))
man = json.load(open(DATA / "splits" / "MANIFEST.json"))

# processed dataset -> source entry
SRC = {"gsm8k": "gsm8k", "gsm8k_train": "gsm8k", "math": "math", "math_train": "math", "aime24": "aime24",
       "aime25": "aime25", "amc23": "amc23", "olympiadbench": "olympiadbench", "humanevalplus": "humanevalplus",
       "mbppplus": "mbppplus", "livecodebench": "livecodebench", "magicoder_oss": "magicoder_oss",
       "mmlu_pro": "mmlu_pro", "gpqa_diamond": "gpqa_diamond", "triviaqa": "triviaqa", "triviaqa_train": "triviaqa",
       "freebaseqa": "freebaseqa", "ifeval": "ifeval", "ifbench": "ifbench", "mt_bench": "mt_bench",
       "alpaca_eval": "alpaca_eval", "tulu3_if": "tulu3_if", "bfcl": "bfcl", "teval": "teval", "metatool": "metatool",
       "apibank": "apibank", "apibank_train": "apibank", "toolace": "toolace", "squad_val": "squad",
       "squad_train": "squad", "dapo_math": "dapo_math", "apps": "apps", "arc": "arc", "gpqa_main": "gpqa_main",
       "sealtool_train": "sealtool", "sealtool": "sealtool", "taco_verified": "taco_verified", "ifeval_train": "rlvr_ifeval"}
CAT = {}
for l in (DATA / "processed").glob("*.jsonl"):
    with open(l) as f:
        CAT[l.stem] = json.loads(f.readline())["category"]
CAT["gpqa_diamond"] = "knowledge"
CAT["gpqa_main"] = "knowledge"

print("| dataset | category | source | revision | license | pool（抽样池） | gate（验收） | test（留出） | reserve | train / search | expected | actual |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, c in rep["counts"].items():
    s = src[SRC[name]]
    rev = s.get("revision") or f"{s.get('package')}=={s.get('version')}"
    rev = rev[:12] if len(str(rev)) == 40 else rev
    repo = s.get("repo") or s.get("package")
    pools = rep["pools"].get(name, {})
    extra = "".join(f" (+{pools[k]} {k})" for k in ("train_excluded", "train_duplicate") if k in pools)
    cells = [str(pools.get(p, "")) for p in ("pool", "gate", "test", "reserve")]
    ts = " / ".join(f"{k} {pools[k]}" for k in ("train", "search") if k in pools)
    cells.append(ts + extra)
    print(f"| {name} | {CAT.get(name, '')} | [{repo}]({s['url']}) | `{rev}` | {s['license']} | " + " | ".join(cells)
          + f" | {c['expected']} | {c['actual'] if c['actual'] is not None else 'MISSING (manual)'} |")
