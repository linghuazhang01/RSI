"""Self-test for data/scripts/draw.py (no model, no network). Draws several generations for Paper A
and Paper B into a temporary directory and checks the reuse policy (revised 2026-11-19):
  * sizes per domain / role; screen is a stratified subset of signal; roles disjoint within a generation
  * every drawn id is in the draw pool (never gate / test / reserve)
  * never-used items are taken first: an item repeats only once its dataset has no never-used items left
  * an item ever drawn as `train` or marked `shown` is never drawn again as signal / screen / kl
    (it may be drawn as `train` again)
  * determinism: the same sequence of draws (incl. --mark-shown) in a second directory gives identical manifests
  * --force redraws only the latest generation; a run refuses a different config, out-of-order
    generations, a changed target, and --mark-shown of ids not drawn in that generation

Run: eval/.venv/bin/python -I data/scripts/check_draw.py [--keep DIR]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
DRAW = ROOT / "data" / "scripts" / "draw.py"
FAIL = []
N_GEN = 6   # 6 x (1,000 code signal + KL) exceeds the TACO-verified pool (9,806): reuse must kick in


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL:", msg)


def draw(run, *args, ok=True):
    r = subprocess.run([PY, "-I", str(DRAW), "--run", str(run), *args], capture_output=True, text=True)
    if ok and r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"draw failed: {args}")
    return r


def manifest(run, g):
    return json.load(open(run / "draws" / f"gen_{g:02d}" / "manifest.json"))


def ids_of(info, role):
    return {(ds, i) for ds, v in info["roles"].get(role, {}).items() for i in v}


def gate_of(run):
    m = json.load(open(run / "draws" / "gate" / "manifest.json"))
    return {(ds, i) for g in m["domains"].values() for ds, ids in g.items() for i in ids}, m


def policy_checks(run, gens, shown_by_gen):
    """shown_by_gen: {gen: set((ds, id))} marked shown after drawing that generation."""
    led = json.load(open(run / "draws" / "ledger.json"))
    gate, _ = gate_of(run)
    for ds in led["items"]:
        check(not ({(ds, i) for i in led["items"][ds]} & gate), f"{ds}: a gate item was drawn")
    for ds in led["items"]:
        pool = set(json.load(open(ROOT / "data" / "splits" / ds / "pool.json")))
        check(set(led["items"][ds]) <= pool, f"{ds}: ledger has ids outside the pool")
    pool_size = {ds: len(json.load(open(ROOT / "data" / "splits" / ds / "pool.json"))) for ds in led["items"]}
    banned = {}          # (ds, id) -> reason: may never be signal / kl again
    used_before = set()  # any role, earlier generations
    n_reused = 0
    for g in gens:
        m = manifest(run, g)
        this_gen = set()
        for d, info in m["domains"].items():
            sig, scr, tr, kl = (ids_of(info, r) for r in ("signal", "screen", "train", "kl"))
            check(scr <= sig, f"gen {g} {d}: screen not subset of signal")
            check(not (sig & tr), f"gen {g} {d}: signal and train overlap")
            check(not (tr & kl), f"gen {g} {d}: train and kl overlap")
            for key in sig | kl:
                if key in banned:
                    check(False, f"gen {g} {d}: {key} used as signal/kl after being {banned[key]}")
            this_gen |= sig | tr | kl
        rep = this_gen & used_before
        n_reused += len(rep)
        for ds in {d for d, _ in rep}:
            fresh_before = pool_size[ds] - len({i for (d2, i) in used_before if d2 == ds})
            drawn_now = len({i for (d2, i) in this_gen if d2 == ds})
            check(drawn_now > fresh_before, f"gen {g} {ds}: reused items while never-used items remained")
        for info in m["domains"].values():
            for key in ids_of(info, "train"):
                banned.setdefault(key, f"train in gen {g}")
        for key in shown_by_gen.get(g, set()):
            banned.setdefault(key, f"shown in gen {g}")
        used_before |= this_gen
    return n_reused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="keep run directories under this path instead of a temp dir")
    a = ap.parse_args()
    base = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="check_draw_"))
    base.mkdir(parents=True, exist_ok=True)
    import yaml
    cfg_a = yaml.safe_load(open(ROOT / "data" / "draw" / "paper_a.yaml"))
    cfg_b = yaml.safe_load(open(ROOT / "data" / "draw" / "paper_b.yaml"))

    def run_a(run, mark_dir):
        mark_dir.mkdir(parents=True, exist_ok=True)
        shown = {}
        for g in range(N_GEN):
            if g == 0:
                draw(run, "--gen", "0", "--config", str(ROOT / "data/draw/paper_a.yaml"), "--export-jsonl")
            else:
                draw(run, "--gen", str(g))
            m = manifest(run, g)
            # pretend the editor was shown 20 code signal items of this generation as error examples
            code_ds = sorted(m["domains"]["code"]["roles"]["signal"])[0]
            ids = sorted(m["domains"]["code"]["roles"]["signal"][code_ds])[:20]
            f = mark_dir / f"shown_{g}.json"
            json.dump({code_ds: ids}, open(f, "w"))
            draw(run, "--gen", str(g), "--mark-shown", str(f))
            shown[g] = {(code_ds, i) for i in ids}
        return shown

    print(f"== Paper A: {N_GEN} generations")
    ra = base / "A_run"
    shown_a = run_a(ra, base / "marks_a")
    for g in range(N_GEN):
        m = manifest(ra, g)
        for d, info in m["domains"].items():
            want = cfg_a["domains"][d]["n"]
            check(info["counts"]["signal"] == want, f"A gen {g} {d}: signal {info['counts']['signal']} != {want}")
            check(info["counts"]["screen"] == min(cfg_a["screen_n"], want), f"A gen {g} {d}: screen size")
        check(sum(i["counts"]["kl"] for i in m["domains"].values()) == cfg_a["kl_contexts"], f"A gen {g}: kl total")
    n_rep = policy_checks(ra, range(N_GEN), shown_a)
    check(n_rep > 0, "expected reuse across generations (code pool smaller than total demand)")
    s0, s1 = (ids_of(manifest(ra, g)["domains"]["math"], "signal") for g in (0, 1))
    check(not (s0 & s1), "math: large pools must not repeat items between gen 0 and gen 1")
    print(f"  items reused across generations: {n_rep}")
    lines = sum(1 for _ in open(ra / "draws/gen_00/data/math.signal.jsonl"))
    check(lines == cfg_a["domains"]["math"]["n"], "export-jsonl line count")
    r = draw(ra, "--gen", "1", "--mark-shown", str(base / "marks_a" / "shown_0.json"), ok=False)
    check(r.returncode != 0, "--mark-shown with ids from another generation must fail")

    print("== fixed gate: sizes, independent of run name and experiment seed")
    gate_a, gm = gate_of(ra)
    for d, dc in cfg_a["domains"].items():
        check(gm["sizes"].get(d, 0) == int(dc.get("gate_n", 0)), f"gate {d}: {gm['sizes'].get(d)} != {dc.get('gate_n')}")
    cfg_seed = dict(cfg_a, seed=7)
    alt_cfg = base / "paper_a_seed7.yaml"
    yaml.safe_dump(cfg_seed, open(alt_cfg, "w"))
    r_alt = base / "other_name_seed7"
    draw(r_alt, "--gate-only", "--config", str(alt_cfg))
    check(gate_of(r_alt)[0] == gate_a, "gate must not depend on run name or experiment seed")
    draw(r_alt, "--gen", "0")
    s_alt = ids_of(manifest(r_alt, 0)["domains"]["math"], "signal")
    check(s_alt != ids_of(manifest(ra, 0)["domains"]["math"], "signal"), "a different seed should draw different items")

    print("== determinism (second directory, same run name, same mark-shown sequence)")
    rb = base / "copy" / "A_run"
    run_a(rb, base / "copy" / "marks_a")
    for g in range(N_GEN):
        check(manifest(ra, g)["domains"] == manifest(rb, g)["domains"], f"gen {g}: manifests differ")

    print("== --force and refusals")
    last = N_GEN - 1
    before = json.load(open(ra / "draws/ledger.json"))
    draw(ra, "--gen", str(last), "--force")
    after = json.load(open(ra / "draws/ledger.json"))

    def strip(led):  # a forced redraw drops that generation's shown marks; the rest must be identical
        return {ds: {i: [u for u in us if u != [last, "shown"]] for i, us in v.items()} for ds, v in led["items"].items()}
    check(strip(before) == strip(after), "--force changed the redraw")
    check(draw(ra, "--gen", "1", "--force", ok=False).returncode != 0, "--force on a non-latest generation must fail")
    check(draw(ra, "--gen", str(N_GEN + 2), ok=False).returncode != 0, "out-of-order generation must fail")
    check(draw(ra, "--gen", str(N_GEN), "--config", str(ROOT / "data/draw/paper_b.yaml"), ok=False).returncode != 0,
          "a different config must be refused")

    targets = ["math", "code", "tool_calling", "code", "math", "code"]
    print("== Paper B: targets " + " -> ".join(targets))
    rB = base / "B_run"
    draw(rB, "--gen", "0", "--config", str(ROOT / "data/draw/paper_b.yaml"), "--target", targets[0])
    for g, t in enumerate(targets[1:], start=1):
        draw(rB, "--gen", str(g), "--target", t)
    for g, t in enumerate(targets):
        m = manifest(rB, g)
        for d, info in m["domains"].items():
            n = cfg_b["domains"][d]["n"]
            if d == t:
                n_tr = round(n * cfg_b["target_train_frac"])
                check(info["counts"]["train"] == n_tr, f"B gen {g} {d}: train size")
                check(info["counts"]["signal"] == n - n_tr, f"B gen {g} {d}: signal size")
            else:
                check(info["counts"]["signal"] == min(cfg_b["retention_n"], n), f"B gen {g} {d}: retention size")
                check(info["counts"]["kl"] == cfg_b["retention_kl"], f"B gen {g} {d}: kl size")
    policy_checks(rB, range(len(targets)), {})
    tr = [ids_of(manifest(rB, g)["domains"]["code"], "train") for g in (1, 3, 5)]
    print(f"  code train items reused between code generations: {len(tr[0] & tr[1]) + len(tr[1] & tr[2]) + len(tr[0] & tr[2])}")
    r = draw(rB, "--gen", "3", "--target", "math", ok=False)
    check(r.returncode != 0 and "target" in (r.stdout + r.stderr), "changing the target of a drawn gen must fail")

    print(draw(ra, "--status").stdout)
    print("RESULT:", "DRAW SELF-TEST PASSED" if not FAIL else f"{len(FAIL)} FAILURES")
    print("run directories:", base)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
