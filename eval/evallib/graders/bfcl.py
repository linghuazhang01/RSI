"""BFCL single-turn grading with the official bfcl-eval AST checker (function-calling / FC mode).

Generation side mirrors bfcl-eval's OpenAI-compatible FC handler:
  functions -> language-specific hints (_func_doc_language_specific_pre_processing)
            -> convert_to_tool(GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)  ('.' -> '_' in names)
Grading side: decode tool_calls to [{name: args}], map sanitized names back to the original
(dotted) names, then
  * irrelevance / live_irrelevance: success iff no function call was emitted
  * live_relevance: success iff at least one non-empty call was emitted
  * everything else: bfcl_eval ast_checker(original function docs, decoded calls, possible_answer)
Multi-turn and agentic categories (memory, web_search) are NOT graded here: they need BFCL's
stateful executable environments (and, for web_search, a SerpAPI key). Use the official
`bfcl generate/evaluate` CLI against the vLLM endpoint for those (see eval/README.md).
"""
import copy
import json

from bfcl_eval.constants.enums import Language, ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import _func_doc_language_specific_pre_processing, is_function_calling_format_output

# A registered BFCL FC model whose config has underscore_to_dot=False: names are already mapped back
# to the original dotted form before checking, so the checker must not convert them again.
CHECKER_MODEL_NAME = "Qwen_Qwen3-8B-FC"

SUPPORTED = {"simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
             "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "irrelevance",
             "live_irrelevance", "live_relevance"}


def build_tools(functions: list, category: str):
    funcs = _func_doc_language_specific_pre_processing(copy.deepcopy(functions), category)
    tools = convert_to_tool(funcs, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    tools = [{"type": "function", "function": t} for t in tools]
    name_map = {f["name"].replace(".", "_"): f["name"] for f in functions}
    return tools, name_map


def decode_tool_calls(tool_calls: list, name_map: dict):
    """OpenAI tool_calls -> BFCL decoded format [{name: {args}}]. Raises on malformed JSON args."""
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function", tc)
        name = name_map.get(fn["name"], fn["name"])
        args = fn.get("arguments", "{}")
        args = json.loads(args) if isinstance(args, str) else (args or {})
        out.append({name: args})
    return out


def language_of(category: str):
    if "java" in category and "javascript" not in category:
        return Language.JAVA
    if "javascript" in category:
        return Language.JAVASCRIPT
    return Language.PYTHON


def grade_bfcl(item: dict, choice: dict):
    """Returns (score, info). info has decision=call|no_call and error details."""
    cat = item["metadata"]["bfcl_category"]
    if cat not in SUPPORTED:
        return None, {"status": "not_supported_use_official_cli", "decision": None}
    _, name_map = build_tools(item["metadata"]["function"], cat)
    try:
        decoded = decode_tool_calls(choice.get("tool_calls"), name_map)
        decode_err = None
    except Exception as e:
        decoded, decode_err = None, f"{type(e).__name__}: {e}"
    has_call = bool(decoded) and any(decoded)
    decision = "call" if has_call else "no_call"
    info = {"decision": decision, "decoded": decoded}
    if "irrelevance" in cat:
        return float(not has_call), info
    if cat == "live_relevance":
        return float(has_call), info
    if decode_err:
        info["error"] = decode_err
        return 0.0, info
    if not decoded or not is_function_calling_format_output(decoded):
        info["error"] = "no/invalid function call"
        return 0.0, info
    try:
        res = ast_checker(item["metadata"]["function"], decoded, item["reference"], language_of(cat), cat,
                          CHECKER_MODEL_NAME)
    except Exception as e:
        info["error"] = f"checker exception {type(e).__name__}: {e}"
        return 0.0, info
    if not res["valid"]:
        info["error"] = res.get("error")
        info["error_type"] = res.get("error_type")
    return float(res["valid"]), info


def gold_tool_calls(item: dict):
    """Construct an OpenAI tool_calls list from the possible_answer (first acceptable value, optional
    params with "" as first option are omitted). Used by the oracle server and grader self-tests."""
    cat = item["metadata"]["bfcl_category"]
    calls = []
    for k, call in enumerate(item["reference"] or []):
        (name, params), = call.items()
        args = {}
        for p, vals in params.items():
            v = vals[0] if isinstance(vals, list) and vals else vals
            if v == "" and isinstance(vals, list):
                nonempty = [x for x in vals if x != ""]
                if not nonempty:
                    continue
                v = nonempty[0] if p in _required(item, name) else None
                if v is None:
                    continue
            args[p] = _first_dict_vals(v)
            if cat in ("simple_java", "simple_javascript"):
                args[p] = _lang_string(args[p], _param_schema(item, name, p), cat)
        calls.append({"id": f"call_{k}", "type": "function",
                      "function": {"name": name.replace(".", "_"), "arguments": json.dumps(args)}})
    return calls


def _param_schema(item, name, p):
    for f in item["metadata"]["function"]:
        if f["name"] == name:
            return f["parameters"]["properties"].get(p, {})
    return {}


def _lang_string(v, schema, cat):
    """Java/JS FC calls pass every argument as a string in the language's own syntax (the checker
    converts it back with java_type_converter / js_type_converter)."""
    t = schema.get("type")
    if isinstance(v, bool):
        return "true" if v else "false"
    if cat == "simple_java":
        nested = (schema.get("items") or {}).get("type")
        if t == "long":
            return f"{v}L"
        if t == "float":
            return f"{v}f"
        if t in ("ArrayList", "Array") and isinstance(v, list):
            el = ", ".join(f'"{x}"' if nested == "String" else (f"'{x}'" if nested == "char" else str(x)) for x in v)
            return f"new ArrayList<>(Arrays.asList({el}))" if t == "ArrayList" else f"new {nested}[]{{{el}}}"
        return v if isinstance(v, str) else str(v)
    if t == "Bigint":
        return f"{v}n"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v if isinstance(v, str) else str(v)


def _required(item, name):
    for f in item["metadata"]["function"]:
        if f["name"] == name:
            return set(f["parameters"].get("required", []))
    return set()


def _first_dict_vals(v):
    """possible_answer nests acceptable values inside dicts/lists of dicts: pick first everywhere."""
    if isinstance(v, dict):
        out = {}
        for k, vals in v.items():
            if isinstance(vals, list):
                nonempty = [x for x in vals if x != ""]
                if not nonempty:
                    continue
                out[k] = _first_dict_vals(nonempty[0])
            else:
                out[k] = _first_dict_vals(vals)
        return out
    if isinstance(v, list):
        return [_first_dict_vals(x) for x in v]
    return v
