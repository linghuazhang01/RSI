"""Deterministic verifiers for the instruction-following TRAINING set `ifeval_train`
(allenai/RLVR-IFeval, pinned in data/scripts/sources.yaml).

Each item carries exactly one constraint, given as `reference = {"func_name": ..., <arguments>}`
(the dataset's `ground_truth` JSON with null arguments dropped). `grade_rlvr_if(response, ref)`
returns a dict whose "score" is 1.0 if the response satisfies the constraint, else 0.0.

Semantics reference: open-instruct `open_instruct/if_functions.py` (IF_FUNCTIONS_MAP) and the
`IFEvalVerifierOld` dispatch in `open_instruct/ground_truth_utils.py`, read (never imported or
executed) at allenai/open-instruct commit 92776a730a4e53cf08b0ddd69faa8e27d2537686. The functions
in `OWN` below are our own clean re-implementations of those semantics, with the deviations listed
in the table. Like open-instruct, the response is stripped of surrounding whitespace first (the
task layer has already removed <think> blocks); unlike it, an empty response scores 0 for every
constraint (open-instruct's newer IFEval verifier does the same).

Where a func_name is the same IFEval instruction (same wording, arguments map one-to-one) we call
the official IFEval checker vendored in eval/third_party/google-research-ifeval (strict mode), so
the training reward for that constraint equals the IFEval held-out metric. langdetect (used by the
two case checkers) is seeded for determinism.

func_name -> implementation
  IFEval checker (argument mapping):
    validate_lowercase            change_case:english_lowercase          {}
    validate_uppercase            change_case:english_capital            {}
    validate_no_commas            punctuation:no_comma                   {}
    validate_quotation            startend:quotation                     {}
    validate_end                  startend:end_checker                   {end_phrase}
    validate_title                detectable_format:title                {}
    validate_json_format          detectable_format:json_format          {}
    validate_two_responses        combination:two_responses              {}
    verify_postscript             detectable_content:postscript          {postscript_marker}
    validate_placeholders         detectable_content:number_placeholders {N -> num_placeholders}
    verify_bullet_points          detectable_format:number_bullet_lists  {N -> num_bullets}
    validate_highlighted_sections detectable_format:number_highlighted_sections {N -> num_highlights}
    validate_paragraphs           length_constraints:nth_paragraph_first_word
                                  {N -> num_paragraphs, i -> nth_paragraph, first_word}
    verify_keywords               keywords:existence                     {keyword_list -> keywords}
    validate_forbidden_words      keywords:forbidden_words               {forbidden_words}
  own implementation (no exact IFEval equivalent):
    verify_keyword_frequency      word appears EXACTLY N times (case-insensitive whole \\w+ tokens);
                                  IFEval keywords:frequency only has "less than" / "at least"
    verify_letter_frequency       letter appears EXACTLY N times. Deviation: case-insensitive
                                  (open-instruct counts case-sensitively; IFEval lowercases)
    verify_paragraph_count        exactly N non-empty paragraphs separated by "* * *" (the divider
                                  the prompt names; IFEval number_paragraphs splits on "***")
    validate_word_constraint      whitespace words; at least >= N, at most <= N,
                                  around |n - N| <= max(round(0.1 N), 1)   (IFEval: no around/at most)
    verify_sentence_constraint    open-instruct sentence regex; around = within +-1
    validate_frequency_capital_words  \\b[A-Z]+\\b words; around as for words (IFEval: no around)
    validate_sections             exactly N sections: occurrences of the splitter, plus 1 if there is
                                  non-blank text before the first one (IFEval multiple_sections uses
                                  ">= N" and needs a digit after the splitter)
    validate_choice               at least one option appears as a standalone token
                                  (case-insensitive). The `options` argument is a STRING
                                  ("yes/no/maybe", "I know or I don't know", "a), b), c), d)");
                                  open-instruct iterates over its characters, which accepts almost
                                  any text -- fixed by splitting it into the options first
    validate_repeat_prompt        response starts with the request (case-insensitive, IFEval
                                  repeat_prompt semantics). Deviation: the dataset's original_prompt
                                  is the FULL prompt including the "First, repeat the request..."
                                  sentence, which the prompt says not to repeat; that sentence is
                                  removed before comparing
"""
import json
import re

from . import ifeval as _ifeval

REPEAT_SENTENCE = ("First, repeat the request without change, then give your answer (do not say anything "
                   "before repeating the request; the request you need to repeat does not include this sentence)")

# func_name -> (IFEval instruction id, {our argument name: IFEval argument name})
IFEVAL_MAP = {
    "validate_lowercase": ("change_case:english_lowercase", {}),
    "validate_uppercase": ("change_case:english_capital", {}),
    "validate_no_commas": ("punctuation:no_comma", {}),
    "validate_quotation": ("startend:quotation", {}),
    "validate_end": ("startend:end_checker", {"end_phrase": "end_phrase"}),
    "validate_title": ("detectable_format:title", {}),
    "validate_json_format": ("detectable_format:json_format", {}),
    "validate_two_responses": ("combination:two_responses", {}),
    "verify_postscript": ("detectable_content:postscript", {"postscript_marker": "postscript_marker"}),
    "validate_placeholders": ("detectable_content:number_placeholders", {"N": "num_placeholders"}),
    "verify_bullet_points": ("detectable_format:number_bullet_lists", {"N": "num_bullets"}),
    "validate_highlighted_sections": ("detectable_format:number_highlighted_sections", {"N": "num_highlights"}),
    "validate_paragraphs": ("length_constraints:nth_paragraph_first_word",
                            {"N": "num_paragraphs", "i": "nth_paragraph", "first_word": "first_word"}),
    "verify_keywords": ("keywords:existence", {"keyword_list": "keywords"}),
    "validate_forbidden_words": ("keywords:forbidden_words", {"forbidden_words": "forbidden_words"}),
}


def _around(n, N):
    return abs(n - N) <= max(round(N * 0.1), 1)


def _quant(n, N, quantifier, around=_around):
    if quantifier == "at least":
        return n >= N
    if quantifier == "at most":
        return n <= N
    if quantifier == "around":
        return around(n, N)
    raise ValueError(f"unknown quantifier {quantifier!r}")


def _token_re(opt):
    return re.compile(r"(?<!\w)" + re.escape(opt) + r"(?!\w)", re.IGNORECASE)


def parse_options(options):
    """'yes/no/maybe' | "I know or I don't know" | 'a), b), c), d)' -> list of options."""
    if isinstance(options, (list, tuple)):
        return [str(o).strip() for o in options if str(o).strip()]
    s = str(options).strip()
    for sep in (" or ", "/", ", "):
        if sep in s:
            return [o.strip() for o in s.split(sep) if o.strip()]
    return [s]


def request_to_repeat(original_prompt):
    return original_prompt.replace(REPEAT_SENTENCE, "").strip()


def _paragraphs_star(t, N):
    t = "\n".join(line.strip() for line in t.splitlines()).strip()
    parts = t.split("* * *")
    return all(p.strip() for p in parts) and len(parts) == N


def _sections(t, N, section_splitter):
    parts = t.split(section_splitter)
    n = len(parts) - 1 + (1 if parts[0].strip() else 0)
    return n == N


def _two_responses(t):
    if t.count("******") != 1:
        return False
    a, b = (x.strip() for x in t.split("******"))
    return a != b


def _json(t):
    try:
        json.loads(t)
    except (ValueError, RecursionError):
        return False
    return True


def _postscript(t, postscript_marker):
    k = t.find(postscript_marker)
    return k >= 0 and len(t[k:].strip()) > len(postscript_marker)


def _nth_paragraph(t, N, i, first_word):
    paras = t.split("\n\n")
    return len(paras) == N and 1 <= i <= N and paras[i - 1].strip().startswith(first_word)


# Own implementations (open-instruct semantics, cleaned). Signature: (stripped_text, **arguments) -> bool.
# Entries for func_names in IFEVAL_MAP are only used for the IFEval cross-check in tests/test_graders.py.
OWN = {
    "verify_keywords": lambda t, keyword_list: all(k.lower() in t.lower() for k in keyword_list),
    "verify_keyword_frequency": lambda t, word, N: sum(
        1 for w in re.findall(r"\b\w+\b", t.lower()) if w == word.lower()) == N,
    "validate_forbidden_words": lambda t, forbidden_words: not any(w.lower() in t.lower() for w in forbidden_words),
    "verify_letter_frequency": lambda t, letter, N: t.lower().count(letter.lower()) == N,
    "verify_paragraph_count": _paragraphs_star,
    "validate_word_constraint": lambda t, N, quantifier: _quant(len(t.split()), N, quantifier),
    "verify_sentence_constraint": lambda t, N, quantifier: _quant(
        len(re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", t)), N, quantifier,
        around=lambda n, N: abs(n - N) <= 1),
    "validate_paragraphs": _nth_paragraph,
    "verify_postscript": _postscript,
    "validate_placeholders": lambda t, N: len(re.findall(r"\[(.*?)\]", t)) >= N,
    "verify_bullet_points": lambda t, N: sum(
        1 for line in t.split("\n") if line.strip().startswith(("*", "-"))) == N,
    "validate_title": lambda t: bool(re.search(r"<<(.*?)>>", t)),
    "validate_choice": lambda t, options: any(_token_re(o).search(t) for o in parse_options(options)),
    "validate_highlighted_sections": lambda t, N: len(re.findall(r"\*(.*?)\*", t)) >= N,
    "validate_sections": _sections,
    "validate_json_format": _json,
    "validate_repeat_prompt": lambda t, original_prompt: t.lower().startswith(
        request_to_repeat(original_prompt).lower()),
    "validate_two_responses": _two_responses,
    "validate_uppercase": lambda t: t == t.upper(),
    "validate_lowercase": lambda t: t == t.lower(),
    "validate_frequency_capital_words": lambda t, N, quantifier: _quant(
        len(re.findall(r"\b[A-Z]+\b", t)), N, quantifier),
    "validate_end": lambda t, end_phrase: t.endswith(end_phrase),
    "validate_quotation": lambda t: t.startswith('"') and t.endswith('"'),
    "validate_no_commas": lambda t: "," not in t,
}

FUNC_NAMES = sorted(OWN)


def _args(ref):
    return {k: v for k, v in ref.items() if k != "func_name" and v is not None}


def check_own(func_name, text, args):
    return bool(OWN[func_name](text.strip(), **args))


def check_ifeval(func_name, text, args):
    """Official IFEval checker for a mapped func_name (strict: response as given, stripped)."""
    iid, amap = IFEVAL_MAP[func_name]
    _ifeval.google_lib()  # puts the vendored package on sys.path (and patches the sentence tokenizer)
    import langdetect
    from instruction_following_eval import instructions_registry

    inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
    inst.build_description(**{amap[k]: v for k, v in args.items()})
    old = langdetect.DetectorFactory.seed
    langdetect.DetectorFactory.seed = 0
    try:
        return bool(inst.check_following(text.strip()))
    finally:
        langdetect.DetectorFactory.seed = old


def grade_rlvr_if(response, ref):
    fn = ref["func_name"]
    if fn not in OWN:
        raise KeyError(f"unknown RLVR-IFeval func_name {fn!r}")
    args = _args(ref)
    text = (response or "").strip()
    impl = f"ifeval:{IFEVAL_MAP[fn][0]}" if fn in IFEVAL_MAP else "own"
    if not text:
        ok = False
    elif fn in IFEVAL_MAP:
        ok = check_ifeval(fn, text, args)
    else:
        ok = check_own(fn, text, args)
    return {"score": float(ok), "func_name": fn, "impl": impl}
