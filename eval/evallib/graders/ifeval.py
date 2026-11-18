"""IFEval (google-research official checker) and IFBench (allenai/IFBench official checker).

Both are loaded from eval/third_party at pinned commits. The only patch: the google-research
checker loads the *pickled* punkt tokenizer (`nltk:tokenizers/punkt/english.pickle`) for
count_sentences; we avoid loading pickles and substitute the equivalent pickle-free
`PunktTokenizer("english")` (punkt_tab, same parameters) -- see eval/README.md.
"""
import importlib.util
import sys

from ..common import THIRD_PARTY

_G = None
_B = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def google_lib():
    global _G
    if _G is None:
        root = THIRD_PARTY / "google-research-ifeval"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import nltk
        from instruction_following_eval import instructions_util

        instructions_util._get_sentence_tokenizer.cache_clear()
        instructions_util._get_sentence_tokenizer = lambda: nltk.tokenize.PunktTokenizer("english")
        from instruction_following_eval import evaluation_lib
        _G = evaluation_lib
    return _G


def ifbench_lib():
    global _B
    if _B is None:
        _B = _load_module("ifbench_evaluation_lib", THIRD_PARTY / "IFBench" / "evaluation_lib.py")
    return _B


def grade_if(lib, prompt: str, response: str, instruction_id_list, kwargs):
    inp = lib.InputExample(key=0, instruction_id_list=list(instruction_id_list), prompt=prompt,
                           kwargs=[dict(k) for k in kwargs])
    p2r = {prompt: response}
    strict = lib.test_instruction_following_strict(inp, p2r)
    loose = lib.test_instruction_following_loose(inp, p2r)
    return {
        "prompt_strict": float(strict.follow_all_instructions),
        "prompt_loose": float(loose.follow_all_instructions),
        "inst_strict": [bool(x) for x in strict.follow_instruction_list],
        "inst_loose": [bool(x) for x in loose.follow_instruction_list],
    }


def grade_ifeval(prompt, response, ref):
    return grade_if(google_lib(), prompt, response, ref["instruction_id_list"], ref["kwargs"])


def grade_ifbench(prompt, response, ref):
    return grade_if(ifbench_lib(), prompt, response, ref["instruction_id_list"], ref["kwargs"])
