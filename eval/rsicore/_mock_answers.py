"""Constructed "gold" replies for the IF training set ifeval_train (RLVR-IFeval) used by the mock server.
Each RLVR-IFeval item has exactly one constraint ({"func_name", args}); we build a reply satisfying it.
The mock server checks the reply with the real grader once and caches the outcome.
"""
from evallib.graders.rlvr_if import parse_options, request_to_repeat

EN = ("the internet protocol is designed to provide a much larger address space and better routing for "
      "modern networks, and this answer explains the main ideas in plain english for every reader")
EN_NC = EN.replace(",", "")


def _filler(letter):
    return "qqq" if letter.lower() != "q" else "zzz"


def rlvr_if_gold(ref: dict) -> str:
    fn = ref["func_name"]
    a = {k: v for k, v in ref.items() if k != "func_name" and v is not None}
    N = int(a.get("N", 1) or 1)
    if fn == "validate_lowercase":
        return EN + "."
    if fn == "validate_uppercase":
        return EN.upper() + "."
    if fn == "validate_no_commas":
        return EN_NC.capitalize() + "."
    if fn == "verify_paragraph_count":
        return "\n* * *\n".join(f"Paragraph {i + 1} text." for i in range(N))
    if fn == "verify_keyword_frequency":
        return " ".join([a["word"]] * N) + " end"
    if fn == "validate_quotation":
        return '"' + EN_NC.capitalize() + '."'
    if fn == "validate_highlighted_sections":
        return " ".join(f"*part {i + 1}*" for i in range(N))
    if fn == "validate_repeat_prompt":
        return request_to_repeat(a["original_prompt"]) + "\nHere is my answer."
    if fn == "validate_paragraphs":
        i = int(a["i"])
        return "\n\n".join((f"{a['first_word']} begins this paragraph." if k + 1 == i else f"Paragraph {k + 1} text.")
                           for k in range(N))
    if fn == "validate_json_format":
        return '{"answer": "ok"}'
    if fn == "verify_letter_frequency":
        return a["letter"].lower() * N + " " + _filler(a["letter"])
    if fn == "validate_frequency_capital_words":
        if a.get("quantifier") == "at most":
            return "all words here are lowercase."
        return " ".join(["WORD"] * N) + " end."
    if fn == "verify_sentence_constraint":
        return " ".join(["This is a sentence."] * N)
    if fn == "validate_end":
        return "Here is my answer. " + a["end_phrase"]
    if fn == "validate_sections":
        sp = a["section_splitter"]
        return "\n".join(f"{sp} {k + 1}\nSome content." for k in range(N))
    if fn == "validate_forbidden_words":
        bad = [w.lower() for w in a.get("forbidden_words", [])]
        for t in ("Okay.", "Sure thing.", "Yes indeed.", "Fine."):
            if not any(w in t.lower() for w in bad):
                return t
        return "Hm."
    if fn == "validate_choice":
        return "My choice: " + parse_options(a["options"])[0]
    if fn == "validate_title":
        return "<<A Title>>\nBody text."
    if fn == "validate_word_constraint":
        return " ".join(["word"] * N)
    if fn == "verify_postscript":
        return "Main answer.\n\n" + a["postscript_marker"] + " an extra note."
    if fn == "validate_placeholders":
        return " ".join(f"[p{k}]" for k in range(N))
    if fn == "verify_bullet_points":
        return "\n".join(f"* item {k + 1}" for k in range(N))
    if fn == "validate_two_responses":
        return "First response here.\n******\nSecond, different response here."
    if fn == "verify_keywords":
        return " ".join(a["keyword_list"]) + "."
    return ""
