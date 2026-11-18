"""Cross-check for the RLVR-IFeval verifiers (eval/evallib/graders/rlvr_if.py).

For every func_name that is dispatched to an official IFEval checker, compare that checker with our
own open-instruct-semantics implementation (rlvr_if.OWN) on synthetic responses built around the
arguments of real `ifeval_train` items (deterministic, seeded). Disagreements are expected where the
two definitions differ (e.g. IFEval accepts ```json fences, is case-insensitive for end phrases,
requires English for the case checkers); the report lists the agreement rate and a few examples.

Informational only (no pass/fail threshold). Run standalone:
  eval/.venv/bin/python -I eval/tests/rlvr_if_crosscheck.py [--per-func 300]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evallib.common import load_items  # noqa: E402
from evallib.graders import rlvr_if as R  # noqa: E402

WORDS = ("the quick brown fox jumps over a lazy dog while rain falls on quiet hills and the river "
         "keeps moving toward an open sea under grey clouds").split()


def sentence(rng, n=None):
    w = [rng.choice(WORDS) for _ in range(n or rng.randint(4, 12))]
    return " ".join(w).capitalize() + rng.choice([".", ".", "!", "?"])


def text(rng, k=None):
    return " ".join(sentence(rng) for _ in range(k or rng.randint(1, 4)))


def recase(rng, s):
    return rng.choice([s, s, s.lower(), s.upper(), s.title()])


def near(rng, N, spread=2):
    return max(0, N + rng.randint(-spread, spread))


def gen(fn, args, rng):
    """One synthetic response for func_name `fn` with item arguments `args`."""
    if fn == "validate_lowercase":
        return recase(rng, text(rng)) + rng.choice(["", " ok", " OK", " 123"])
    if fn == "validate_uppercase":
        return recase(rng, text(rng)) + rng.choice(["", " ok", " OK", " 123"])
    if fn == "validate_no_commas":
        s = text(rng)
        return s if rng.random() < 0.5 else s.replace(" ", ", ", 1)
    if fn == "validate_quotation":
        s = text(rng)
        return rng.choice(['"%s"' % s, ' "%s" ' % s, '"%s' % s, s, '""', '"', "'%s'" % s])
    if fn == "validate_end":
        e = args["end_phrase"]
        s = text(rng)
        return rng.choice([f"{s} {e}", f"{s} {e.lower()}", f'"{s} {e}"', f"{s} {e} Bye.", f"{s}\n{e}  ", f"{e} {s}"])
    if fn == "validate_title":
        s = text(rng)
        return rng.choice([f"<<{s[:20]}>>\n{s}", f"<<>> {s}", f"<< >> {s}", f"<<a\nb>> {s}", s, f"<{s[:10]}> {s}"])
    if fn == "validate_json_format":
        obj = {"answer": text(rng, 1), "n": rng.randint(0, 9)}
        j = json.dumps(obj)
        return rng.choice([j, f"```json\n{j}\n```", f"```\n{j}\n```", f"Here: {j}", j[:-1], f"[{j}]", f" {j} "])
    if fn == "validate_two_responses":
        a, b = text(rng), text(rng)
        return rng.choice([f"{a}\n******\n{b}", f"{a}******{a}", f"{a}\n******\n{b}\n******\n{a}", f"{a}\n*****\n{b}",
                           f"******\n{a}\n******\n{b}", a, f"{a}\n******\n"])
    if fn == "verify_postscript":
        m = args["postscript_marker"]
        s = text(rng)
        return rng.choice([f"{s}\n\n{m} {text(rng, 1)}", f"{s}\n{m.lower()} {text(rng, 1)}", f"{s}\n{m}",
                           f"{s} {m} {text(rng, 1)}", s, f"{s}\nSteps: {text(rng, 1)}", f"{m.upper()} {s}"])
    if fn == "validate_placeholders":
        n = near(rng, args["N"])
        return text(rng) + " " + " ".join(rng.choice(["[x]", "[name]", "[]", "[a b]"]) for _ in range(n))
    if fn == "verify_bullet_points":
        n = near(rng, args["N"])
        lines = [rng.choice(["* ", "- ", "*", "**bold** ", "  * ", "-"]) + sentence(rng) for _ in range(n)]
        if rng.random() < 0.3:
            lines.insert(rng.randint(0, len(lines)), rng.choice(["* * *", "***", "---"]))
        return "Intro.\n" + "\n".join(lines)
    if fn == "validate_highlighted_sections":
        n = near(rng, args["N"])
        hs = [rng.choice(["*%s*", "**%s**", "* %s *", "*%s\n*"]) % " ".join(rng.sample(WORDS, 2)) for _ in range(n)]
        if rng.random() < 0.3:
            hs.append(rng.choice(["**", "* *", "****"]))
        return " ".join(hs)
    if fn == "validate_paragraphs":
        N, i, fw = args["N"], args["i"], args["first_word"]
        n = near(rng, N, 1)
        paras = [text(rng, 1) for _ in range(n)]
        if 1 <= i <= n:
            paras[i - 1] = rng.choice([fw, fw.capitalize(), f'"{fw}', f"{fw}ly", "other", f"{fw}."]) + " " + paras[i - 1]
        sep = rng.choice(["\n\n", "\n\n", "\n\n\n", "\n"])
        out = sep.join(paras)
        return out + rng.choice(["", "\n\n", "\n\n "])
    if fn == "verify_keywords":
        ks = args["keyword_list"]
        s = text(rng)
        mode = rng.choice(["all", "some", "none", "inside", "upper"])
        if mode == "all":
            return s + " " + " ".join(ks)
        if mode == "some":
            return s + " " + ks[0]
        if mode == "inside":
            return s + " " + " ".join("x" + k + "y" for k in ks)
        if mode == "upper":
            return s + " " + " ".join(k.upper() for k in ks)
        return s
    if fn == "validate_forbidden_words":
        fw = args["forbidden_words"]
        s = text(rng)
        mode = rng.choice(["none", "whole", "inside", "upper", "plural"])
        w = rng.choice(fw)
        return {"none": s, "whole": f"{s} {w}.", "inside": f"{s} un{w}ed.", "upper": f"{s} {w.upper()}",
                "plural": f"{s} {w}s"}[mode]
    raise KeyError(fn)


def crosscheck(per_func=300, seed=20261119, items_per_func=30):
    rows = load_items("ifeval_train")
    by = defaultdict(list)
    for r in rows:
        if r["split_pool"] in ("pool", "gate"):
            by[r["reference"]["func_name"]].append(r)
    out = {}
    for fn in sorted(R.IFEVAL_MAP):
        rng = random.Random(f"{seed}:{fn}")
        its = sorted(by[fn], key=lambda r: r["id"])
        its = rng.sample(its, min(items_per_func, len(its)))
        agree, n, ex = 0, 0, []
        n_pass_ifeval = n_pass_own = 0
        while n < per_func:
            it = its[n % len(its)]
            args = R._args(it["reference"])
            resp = gen(fn, args, rng)
            if not resp.strip():
                continue
            a = R.check_ifeval(fn, resp, args)
            b = R.check_own(fn, resp, args)
            n += 1
            agree += a == b
            n_pass_ifeval += a
            n_pass_own += b
            if a != b and len(ex) < 3:
                ex.append({"id": it["id"], "response": resp[:160], "ifeval": a, "own": b})
        out[fn] = {"ifeval_checker": R.IFEVAL_MAP[fn][0], "n": n, "agreement": agree / n,
                   "pass_rate_ifeval": n_pass_ifeval / n, "pass_rate_own": n_pass_own / n, "disagreement_examples": ex}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-func", type=int, default=300)
    a = ap.parse_args()
    res = crosscheck(a.per_func)
    for fn, r in res.items():
        print(f"{fn:30s} {r['ifeval_checker']:45s} n={r['n']} agreement={r['agreement']:.3f} "
              f"pass(ifeval)={r['pass_rate_ifeval']:.2f} pass(own)={r['pass_rate_own']:.2f}")
    tot = sum(r["n"] for r in res.values())
    print(f"overall agreement: {sum(r['agreement'] * r['n'] for r in res.values()) / tot:.3f} over {tot} responses")


if __name__ == "__main__":
    sys.exit(main())
