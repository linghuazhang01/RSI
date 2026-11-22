"""Generate the unified search(=signal)/gate/test(=held-out)/reserve/train pools from data/splits/split_config.yaml.

* Reads data/processed/<dataset>.jsonl (written by normalize.py).
* Writes data/splits/<dataset>/<pool>.json  (sorted JSON list of item ids)
* Training sets: eval-overlapping items -> train_excluded, internal duplicates -> train_duplicate (both
  before splitting, so no loop-visible pool contains them).
* Rewrites data/processed/<dataset>.jsonl with split_pool filled in.
* Writes data/splits/MANIFEST.json with sha256 of every split file (+ config + processed files).

Deterministic: depends only on item ids, the config and the seed (20261118).
Run: eval/.venv/bin/python -I data/scripts/make_splits.py
"""
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parents[1]
PROC = DATA / "processed"
SPL = DATA / "splits"
CFG_PATH = SPL / "split_config.yaml"
CFG = yaml.safe_load(open(CFG_PATH))
SEED = CFG["seed"]


def load(name):
    with open(PROC / f"{name}.jsonl") as f:
        return [json.loads(l) for l in f]


def shuffled(ids, tag):
    ids = sorted(ids)
    random.Random(f"{SEED}:{tag}").shuffle(ids)
    return ids


def take_sequential(ids, tag, sizes, rest):
    """sizes: list of (pool, n). Returns {pool: [ids]}."""
    order = shuffled(ids, tag)
    out, i = {}, 0
    for pool, n in sizes:
        assert i + n <= len(order), f"{tag}: not enough items for {pool} ({n})"
        out[pool] = order[i:i + n]
        i += n
    if rest and i < len(order):
        out.setdefault(rest, []).extend(order[i:])
    return out


def alloc(strata_sizes: dict, total: int) -> dict:
    """Largest-remainder proportional allocation of `total` over strata (capped by stratum size)."""
    N = sum(strata_sizes.values())
    raw = {k: total * v / N for k, v in strata_sizes.items()}
    base = {k: min(int(r), strata_sizes[k]) for k, r in raw.items()}
    left = total - sum(base.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), k)):
        if left <= 0:
            break
        if base[k] < strata_sizes[k]:
            base[k] += 1
            left -= 1
    assert sum(base.values()) == total
    return base


def stratified(items_by_stratum: dict, tag, sizes):
    """Allocate each pool proportionally across strata, then take sequentially within strata.
    sizes: list of (pool, n); the last pool may absorb the remainder if n is None."""
    remaining = {k: shuffled(v, f"{tag}:{k}") for k, v in items_by_stratum.items()}
    out = defaultdict(list)
    for pool, n in sizes:
        avail = {k: len(v) for k, v in remaining.items() if v}
        if n is None:
            for k in list(remaining):
                out[pool].extend(remaining[k])
                remaining[k] = []
            continue
        a = alloc(avail, n)
        for k, m in a.items():
            out[pool].extend(remaining[k][:m])
            remaining[k] = remaining[k][m:]
    leftover = [x for v in remaining.values() for x in v]
    return dict(out), leftover


def split_dataset(name, rows, c):
    ids = [r["id"] for r in rows]
    if "all" in c:
        return {c["all"]: ids}

    if name == "math":
        m500 = [r["id"] for r in rows if r["metadata"]["is_math500"]]
        other = [r["id"] for r in rows if not r["metadata"]["is_math500"]]
        sizes = [(p, c[p]) for p in ("search", "gate") if isinstance(c.get(p), int)]
        out = take_sequential(other, name, sizes, c["rest"])
        out["test"] = m500
        return out

    if name == "sealtool":
        return {"test": [r["id"] for r in rows if r["metadata"]["seal_split"] != "dev"],
                "reserve": [r["id"] for r in rows if r["metadata"]["seal_split"] == "dev"]}

    if name == "livecodebench":
        # cut on a contest-date boundary (problems of one date never straddle pools), choosing the
        # boundary whose search share is closest to search_frac_oldest (ties -> smaller search)
        rs = sorted(rows, key=lambda r: (r["metadata"]["contest_date"], r["id"]))
        target = len(rs) * c["search_frac_oldest"]
        cuts = [i for i in range(1, len(rs)) if rs[i]["metadata"]["contest_date"] != rs[i - 1]["metadata"]["contest_date"]]
        k = min(cuts, key=lambda i: (abs(i - target), i))
        return {"search": [r["id"] for r in rs[:k]], "test": [r["id"] for r in rs[k:]]}

    if name == "ifeval":
        strata = defaultdict(list)
        for r in rows:
            strata[r["reference"]["instruction_id_list"][0].split(":")[0]].append(r["id"])
        # order pool -> gate -> test keeps gate/test identical to the earlier search/gate/test split
        out, left = stratified(strata, name, [("pool", c["pool"]), ("gate", c["gate"]), ("test", c["test"])])
        assert not left
        return out

    if name == "bfcl":
        bycat = defaultdict(list)
        for r in rows:
            bycat[r["metadata"]["bfcl_category"]].append(r["id"])
        out = defaultdict(list)
        used = set()
        for cat, n in c["search"].items():
            pick = shuffled(bycat[cat], f"{name}:search:{cat}")[:n]
            out["search"].extend(pick)
            used.update(pick)
        strata = {cat: [i for i in bycat[cat] if i not in used] for cat in c["gate_stratify_over"]}
        g, _ = stratified(strata, f"{name}:gate", [("gate", c["gate"])])
        out["gate"] = g["gate"]
        used.update(out["gate"])
        out["test"] = [i for i in ids if i not in used]
        return dict(out)

    if name == "squad_val":
        test = [r["id"] for r in rows if r["metadata"]["in_seal_eval"]]
        assert len(test) == 200
        return {"test": test, "reserve": [r["id"] for r in rows if not r["metadata"]["in_seal_eval"]]}

    if name == "squad_train":
        bytitle = defaultdict(list)
        for r in rows:
            bytitle[r["metadata"]["title"]].append(r["id"])
        titles = shuffled(list(bytitle), f"{name}:titles")
        ns, ng, ppt = c["search_titles"], c["gate_titles"], c["passages_per_title"]
        out = {"search": [], "gate": [], "train": [], "reserve": []}
        for t in titles[:ns]:
            ps = shuffled(bytitle[t], f"{name}:{t}")
            out["search"] += ps[:ppt]
            out["reserve"] += ps[ppt:]   # other passages of search titles: never used for training
        for t in titles[ns:ns + ng]:
            ps = shuffled(bytitle[t], f"{name}:{t}")
            out["gate"] += ps[:ppt]
            out["reserve"] += ps[ppt:]
        for t in titles[ns + ng:]:
            out["train"] += bytitle[t]
        return out

    # generic: gate/test sizes with optional rest (training sets: rest = pool); optional
    # stratification of the fixed slices by a metadata key (e.g. APPS difficulty)
    sizes = [(p, c[p]) for p in ("search", "gate", "test") if isinstance(c.get(p), int)]
    if c.get("stratify_by"):
        strata = defaultdict(list)
        for r in rows:
            strata[str(r["metadata"][c["stratify_by"]])].append(r["id"])
        out, left = stratified(strata, name, sizes)
        if c.get("rest"):
            out.setdefault(c["rest"], []).extend(left)
        else:
            assert not left, f"{name}: {len(left)} items unassigned"
        return out
    return take_sequential(ids, name, sizes, c.get("rest"))


def text_of(r):
    """Text used for decontamination: prompt, else concatenated user turns."""
    if r.get("prompt"):
        return r["prompt"]
    msgs = r.get("messages") or []
    flat = []
    def walk(m):
        if isinstance(m, list):
            for x in m:
                walk(x)
        elif isinstance(m, dict) and m.get("role") == "user":
            flat.append(str(m.get("content", "")))
    walk(msgs)
    return "\n".join(flat)


def norm_text(s):
    return " ".join(str(s).lower().split())


def norm_strict(s):
    """Also strip LaTeX markup, punctuation and spaces: catches formatting variants of the same item
    (e.g. `Who wrote the book `A Brief History Of Time`?` vs the same question without backticks)."""
    s = str(s).lower()
    s = re.sub(r"\\(left|right|displaystyle|text|mathrm|mathbf|operatorname|dfrac|tfrac|frac)", "", s)
    s = s.replace("\\$", "").replace("$", "")
    return re.sub(r"[\W_]+", "", s)


NGRAM, NEAR_THR = 13, 0.5
# 13-grams found in >= TEMPLATE_DF eval items of these categories are template text (IF constraint
# sentences, tool-call format instructions) and do not count as shared in the near-duplicate check
TEMPLATE_DF, TEMPLATE_CATEGORIES = 5, {"instruction_following", "tool_calling"}


def near_text(r):
    """Text for near-duplicate checks. Seal-Tools prompts are a shared template + api_list, so only the
    task_instruction line is compared."""
    t = text_of(r)
    if str(r.get("dataset", "")).startswith("sealtool"):
        m = re.search(r'task_instruction = "(.*)"\s*\nOutput:', t, re.S)
        if m:
            return m.group(1)
    return t


def grams(s):
    w = re.findall(r"\w+", str(s).lower())
    return {" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def is_train_dataset(c):
    return c.get("role") == "train" or c.get("all") == "train"


def weight_alloc(weights: dict, total: int) -> dict:
    """Largest-remainder split of `total` over keys by (float) weight; ties broken by key."""
    W = sum(weights.values())
    raw = {k: total * w / W for k, w in weights.items()}
    out = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def gate_sizes():
    """Per-dataset gate sizes from the top-level `gate_domains` table."""
    sizes = {}
    for dom, g in (CFG.get("gate_domains") or {}).items():
        for ds, n in weight_alloc({k: float(v) for k, v in g["datasets"].items()}, int(g["size"])).items():
            assert ds not in sizes, f"{ds} appears in two gate domains"
            sizes[ds] = n
    return sizes


def resolve(name, c, sizes):
    if c.get("gate") == "domain":
        assert name in sizes, f"{name}: gate: domain but not listed in gate_domains"
        c = dict(c, gate=sizes[name])
    return c


def main():
    manifest = {"seed": SEED, "config_sha256": sha256(CFG_PATH), "splits": {}, "processed": {},
                "decontamination": {}}
    sizes = gate_sizes()
    present = {n: resolve(n, c, sizes) for n, c in CFG["datasets"].items() if (PROC / f"{n}.jsonl").exists()}
    manifest["gate_sizes"] = {n: c["gate"] for n, c in present.items() if isinstance(c.get("gate"), int)}
    for n in CFG["datasets"]:
        if n not in present:
            print(f"[skip] {n}: not in data/processed (manual / not downloaded)")
    # Exact-match decontamination set: normalized text of EVERY item of every evaluation dataset
    # (all pools incl. reserve). SQuAD: passages of squad_val (+ squad_train search/gate are
    # handled by title-disjoint construction).
    # Two normalizations (2026-11-21): whitespace/case on the full text, and strict (also LaTeX markup,
    # punctuation, spaces) on near_text, so a Seal-Tools task_instruction is compared on its own.
    eval_texts, eval_strict = set(), set()
    for n, c in present.items():
        if not is_train_dataset(c):
            for r in load(n):
                eval_texts.add(norm_text(text_of(r)))
                eval_strict.add(norm_strict(near_text(r)))
    eval_texts.discard("")
    eval_strict.discard("")
    # Near-duplicate index: 13-grams of EVERY eval item (all pools incl. reserve, since 2026-11-21; was test
    # only), per category. A training item is dropped if its shared 13-grams with a single eval item of the
    # same category cover >= 50% of either item's 13-grams. Template 13-grams of TEMPLATE_CATEGORIES are
    # left out of the index.
    near_idx = defaultdict(lambda: defaultdict(set))
    near_len = {}
    for n, c in present.items():
        if is_train_dataset(c):
            continue
        for r in load(n):
            G = grams(near_text(r))
            near_len[r["id"]] = len(G)
            for g in G:
                near_idx[r["category"]][g].add(r["id"])
    for cat in TEMPLATE_CATEGORIES & set(near_idx):
        near_idx[cat] = {g: s for g, s in near_idx[cat].items() if len(s) < TEMPLATE_DF}

    def near_dup(r):
        G = grams(near_text(r))
        idx = near_idx.get(r["category"])
        if not G or not idx:
            return None
        cnt = defaultdict(int)
        for g in G:
            for x in idx.get(g, ()):
                cnt[x] += 1
        if not cnt:
            return None
        # symmetric: shared 13-grams cover >= 50% of the training item OR of the held-out item
        # (the second catches long training problems that embed a held-out problem)
        frac = {x: max(k / len(G), k / max(near_len[x], 1)) for x, k in cnt.items()}
        best = max(frac, key=lambda x: (frac[x], x))
        return best if frac[best] >= NEAR_THR else None
    for name, c in present.items():
        p = PROC / f"{name}.jsonl"
        rows = load(name)
        if is_train_dataset(c):
            # training sets: drop items whose text exactly matches ANY held-out item BEFORE splitting,
            # so the signal / gate / train pools are all decontaminated
            exact = sorted(r["id"] for r in rows
                           if norm_text(text_of(r)) in eval_texts or norm_strict(near_text(r)) in eval_strict)
            exact_set = set(exact)
            near = {r["id"]: m for r in rows if r["id"] not in exact_set and (m := near_dup(r))}
            bad_set = exact_set | set(near)
            # internal duplicates (2026-11-21): same text under norm_strict(near_text) as an earlier id of the
            # same training set -> train_duplicate, keeping the smallest id. Otherwise one problem could be
            # drawn as both `train` and `signal` (Paper B) or into both the gate and a generation's draw.
            seen, dup = {}, {}
            for r in sorted(rows, key=lambda r: r["id"]):
                if r["id"] in bad_set:
                    continue
                k = norm_strict(near_text(r))
                if k in seen:
                    dup[r["id"]] = seen[k]
                else:
                    seen[k] = r["id"]
            pools = split_dataset(name, [r for r in rows if r["id"] not in bad_set and r["id"] not in dup], c)
            if bad_set:
                pools["train_excluded"] = sorted(bad_set)
            if dup:
                pools["train_duplicate"] = sorted(dup)
            manifest["decontamination"][name] = {
                "exact_match_removed": len(exact), "exact_examples": exact[:20],
                "near_dup_13gram_removed": len(near),
                "near_dup_examples": [[k, near[k]] for k in sorted(near)[:20]],
                "internal_duplicates_removed": len(dup),
                "internal_duplicate_examples": [[k, dup[k]] for k in sorted(dup)[:20]]}
        else:
            pools = split_dataset(name, rows, c)
        for old, new in (c.get("relabel") or {}).items():  # e.g. ifeval: pool/gate -> reserve
            if old in pools:
                pools.setdefault(new, []).extend(pools.pop(old))
        id2pool = {}
        for pool, ids in pools.items():
            for i in ids:
                assert i not in id2pool, f"{name}: {i} in {id2pool[i]} and {pool}"
                id2pool[i] = pool
        assert len(id2pool) == len(rows), f"{name}: {len(rows) - len(id2pool)} items unassigned"
        d = SPL / name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.json"):
            old.unlink()
        for pool, ids in pools.items():
            fp = d / f"{pool}.json"
            json.dump(sorted(ids), open(fp, "w"), indent=0)
            manifest["splits"][f"{name}/{pool}.json"] = {"n": len(ids), "sha256": sha256(fp)}
        with open(p, "w") as f:
            for r in rows:
                r["split_pool"] = id2pool[r["id"]]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["processed"][f"{name}.jsonl"] = {"n": len(rows), "sha256": sha256(p)}
        print(f"{name:16s} " + "  ".join(f"{k}={len(v)}" for k, v in sorted(pools.items())))
    json.dump(manifest, open(SPL / "MANIFEST.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
