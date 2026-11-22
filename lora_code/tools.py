"""The editor's tool surface for one generation: a Workspace holding tensors behind
string handles, registered on an rsicore.sandbox.ToolAPI so programs call `api.<tool>(**kwargs)`.

Handles: L* = LoRA, W* = weight edit (mode=weights), S* = SVD of a weight, A* = activation statistics,
H<g> = adapter accepted in generation g (the stack), C:<cid> = an earlier candidate of this generation.
merge.mode adapter (scratch, 2026-11-22): the current model is M_0 + the cumulative adapter C_g (base_lora, never
merged); store reads W_0 + B_C A_C (OverlayStore), HF passes (acts / train_loss) run with C_g applied, and
evaluator.register serves a candidate (a delta on M_g) as C_g (+) delta.
Vector references (for rank1 / from_direction): "S3.u0", "S3.v1", "A2.in_mean", "A2.out_mean", "A2.pc0", and
"T5.<name>" (a vector returned by tensor_compute).

Budgets (RSI_LOOP_DESIGN §3.3): `evaluate` calls (an identical LoRA is evaluated once and only charged
once) and `gpu_seconds` for acts / train_loss (measured on gpu1; estimated in mock).
All accounting is done by the workspace's rsicore.sandbox.ToolAPI (`ws.api`, created with the workspace):
evaluate = cost_fn (1 per non-empty LoRA) + dedupe_key_fn (LoRA name = digest; identical LoRAs are free,
also between program calls and the search's screen evaluation, which goes through the same tool);
GPU tools = cost {"gpu_seconds": 0} (refused once exhausted) + cost_fn (measured seconds, capped at what
was left before the call so a successful call keeps its result; the uncapped total is ws.gpu_measured).
search.charge_screens false (2026-11-21; search_settings): the automatic screen is free
(screen_candidate) and the `evaluate` budget is only the pool of the programs' api.evaluate calls
(search.program_evaluate_total); an api.evaluate of a LoRA that was already evaluated (screened) is free.
Data access: acts / train_example only accept this generation's `train` ids (TrainData).
The editor never trains and never computes gradients (2026-11-20: no sgd / grad / from_grad tools in any
mode). Tool set by method.mode (tools_for_mode): edit = every LoRA tool plus raw_lora (the raw LoRA under handle
"RAW"); scratch = every LoRA tool without raw_lora; weights = every LoRA tool without raw_lora plus the weight-edit
tools (WEIGHT_TOOLS: structured, any-rank edits of the dense weights, a WeightEdit / MockWeightEdit with
kind "weights"; programs return {"edit": handle}; per-module cap ||W'-W||_F/||W||_F <= weights.norm_cap_tau).
Unregistered tools are refused by the ToolAPI.
tensor_compute (2026-11-21, every mode and the exploration api; tools.tensor_compute: false removes it and `tensors`):
CPU torch code in a fresh sandbox (rsicore.sandbox.run_program, no tools) on fp32 copies of up to
tensor_compute.max_tensors CURRENT model tensors (any language-model tensor, WeightStore.tensor_names(); the child
reads the shard files itself). Outputs: `result` (JSON, cut to result_chars), `vectors` (handle T<n>, refs
"T<n>.<name>"), `lora` (handle L<n>, linear modules, rank <= lora.rank_cap) and, in mode=weights, `writes` (dense
"delta" ops on a new W handle; caps change_errors / no uniform rescale, never embed_tokens / final norm / lm_head).
It runs WITHOUT the workspace lock (only resolving inputs / registering outputs take it), under a process-wide
semaphore (max_concurrent), with timeout min(timeout_s, the calling program's / exploration call's time left).
The simple flow (2026-11-21, base.yaml; search_settings / paper_tools_enabled / screen_tools_enabled): no evaluate
tool (search.program_evaluate_calls 0); screen_results / screen_output read the current model's own answers on the
generation's screen items (ws.screen, set by run.py); paper tools save_edit / load_edit / list_edits / drop_edit
(out_dir/saved/<root cid>/), output_matrices, unit, direction, project_out, acts(source='own'),
tensor_compute(vectors=...).
Exploration (2026-11-21): ws.explore_api(tag) is the `api` of the editor's run_python chat tool (rsicore.chat_tools)
while it writes a candidate: a separate ToolAPI without evaluate / raw_lora and (unless editor.explore.gpu_tools)
without the GPU tools; it never charges the generation's budgets, and its handles are freed after each call.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

from . import mock as M
from .common import stable_hash, write_json

TOOL_DOCS = {
    "modules": "modules(layer=None, leaf=None, group=None) -> list of {module, layer, layer_type, leaf, group, out, in} (the only valid module names)",
    "diagnostics": "diagnostics(module=None, layer=None, top=20, sort='w_norm') -> rows of the fixed diagnostics table (a list; can be EMPTY for a module or layer without a row: check before indexing)",
    "train_ids": "train_ids(n=None) -> ids of this generation's train items (the only items activation tools accept)",
    "train_example": "train_example(id) -> {prompt, target} of one train item (truncated)",
    "svd": "svd(module, k=8) -> {handle, S, shape}: top-k singular triplets of the CURRENT weight W",
    "acts": "acts(module, ids=None, k=4) -> {handle, in_norm, out_norm, pc_var}: activation stats on train items",
    "new_lora": "new_lora() -> {lora}: an empty LoRA",
    "rank1": "rank1(lora, module, u, v, s) -> {lora}: add s * u v^T (normalised vector refs of the SAME module: u output side, length out: S.u*, A.out_mean; v input side, length in: S.v*, A.in_mean, A.pc*)",
    "from_direction": "from_direction(lora, module, direction, key, alpha) -> {lora}: dW = alpha d k^T/|k|^2 so dW k = alpha d (direction d: output side, length out: S.u*, A.out_mean; key k: input side, length in: S.v*, A.in_mean, A.pc*)",
    "scale_singular": "scale_singular(lora, module, svd, idx, factors) -> {lora}: dW = sum_i (f_i-1) s_i u_i v_i^T",
    "scale": "scale(lora, s, modules=None) -> {lora}",
    "interpolate": "interpolate(a, b, lam) -> {lora}: (1-lam) a + lam b",
    "combine": "combine(loras, weights=None) -> {lora}: weighted sum",
    "drop": "drop(lora, modules) -> {lora}",
    "truncate_rank": "truncate_rank(lora, k, modules=None) -> {lora}",
    "intruder_scale": "intruder_scale(lora, lam=0.5, eps=0.3, k0=32, modules=None) -> {lora}",
    "relchange_prune": "relchange_prune(lora, frac=0.03, modules=None) -> {lora}",
    "orthogonalize": "orthogonalize(lora, against='stack') -> {lora}",
    "ties": "ties(loras, weights=None, density=0.2) -> {lora}",
    "inspect": "inspect(lora) -> {modules, max_rank, check, top, intruder_mean}",
    "stack": "stack() -> list of {handle, gen, modules, max_rank} (adapters merged in earlier generations)",
    "raw_lora": "raw_lora() -> {lora: 'RAW', modules, max_rank, train}: the LoRA you start from (trained once, "
                "in generation 0, with the fixed raw recipe; edited since)",
    "candidate": "candidate(cid) -> {lora} of an earlier candidate of this generation",
    "evaluate": "evaluate(lora) -> structured screen feedback (costs 1 evaluate; identical LoRAs are free)",
    "compare": "compare(cid, other='parent', level='screen') -> per dataset: acc and acc_other, broken (other right, "
               "cid wrong) / fixed counts and ids, flips by output length, truncation; ids and statistics only, both "
               "must already be evaluated (free). cid / other: a candidate id or 'parent' (the current model)",
    "train_loss": "train_loss(lora, ids=None, n=64) -> {loss, parent_loss, delta, n}: mean answer NLL of train items "
                  "under the current model + lora vs without it (forward pass only, no training; costs gpu_seconds)",
    "budget": "budget() -> {spent, left, program_seconds_left, program_evaluate_calls_left}: generation budgets, and the wall clock / evaluate calls this program has left",
    "tensors": "tensors(layer=None, pattern=None) -> list of {name, shape, dtype, editable}: EVERY language-model "
               "tensor (not only the linear modules: input/post_attention_layernorm, self_attn.q_norm/k_norm, "
               "linear_attn.dt_bias / A_log / conv1d / norm, '<module>.weight'); pattern = substring or glob; "
               "embed_tokens, the final norm and lm_head are readable but never editable",
    "tensor_compute": "tensor_compute(code, tensors) -> {json, vectors, lora, lora_check, seconds, stdout}: run CPU "
                      "torch code in a fresh sandbox on fp32 copies of at most 4 CURRENT model tensors (tensors = "
                      "names from api.tensors() or module names). The code sees T = {name: tensor} and torch "
                      "(changing T changes nothing) and may set: result = a JSON-able value (returned as json, cut "
                      "to ~4000 chars); vectors = {name: 1-D tensor} (returned as refs 'T5.name', usable as rank1 / "
                      "from_direction vectors); lora = {module: (A [r, in], B [out, r])} (a new LoRA handle; linear "
                      "modules, rank <= the LoRA rank cap). No GPU, a few threads, at most ~300 s and never more than "
                      "your program has left: prefer torch.svd_lowrank to a full SVD of a large matrix",
    # ---- mode=weights only (WEIGHT_TOOLS): edits of the dense weights, applied IN ORDER to the CURRENT weights
    "new_edit": "new_edit() -> {edit}: an empty weight edit (handle 'W..'); every w_* op below returns a NEW edit "
                "with the op appended; ops apply in order to the CURRENT weights W; per module "
                "||W'-W||_F/||W||_F <= weights.norm_cap_tau",
    "w_scale": "w_scale(edit, modules, s) -> {edit}: W <- s * W on every listed module (keep s near 1)",
    "w_scale_rows": "w_scale_rows(edit, module, factors, rows=None) -> {edit}: scale OUTPUT channels (rows of W, "
                    "shape [out, in]) of one module: factors = one float or a list (one per row in rows; rows=None "
                    "= all rows)",
    "w_scale_cols": "w_scale_cols(edit, module, factors, cols=None) -> {edit}: scale INPUT channels (columns of W) "
                    "of one module: factors = one float or a list (one per column in cols; cols=None = all columns)",
    "w_add_lora": "w_add_lora(edit, lora, alpha=1.0) -> {edit}: add alpha * dW of a LoRA built with the LoRA tools "
                  "(new_lora, scale_singular, rank1, ...) as a dense update",
    "w_toward_original": "w_toward_original(edit, modules, alpha) -> {edit}: W <- W + alpha (W_0 - W) on the listed "
                         "modules (W_0 = the original model M_0's weights, 0 < alpha <= 1); undoes earlier merges "
                         "and earlier ops of this edit there",
    "inspect_edit": "inspect_edit(edit) -> {modules, check_ok, check_errors, top_rel_norm, total_rel_norm, ops}",
}
WEIGHT_TOOLS = ("new_edit", "w_scale", "w_scale_rows", "w_scale_cols", "w_add_lora", "w_toward_original",
                "inspect_edit")
WEIGHTS_DOC_OVERRIDES = {   # mode=weights: tools that also take an edit handle
    "tensor_compute": "tensor_compute(code, tensors, edit=None) -> {json, vectors, lora, edit, writes, seconds, "
                      "stdout}: as in the LoRA modes (T = {name: fp32 copy of the CURRENT tensor, after the ops of "
                      "`edit` if given}; result / vectors / lora as usual), plus writes = {tensor name: NEW full tensor, "
                      "same shape} -> appended to `edit` (or a new edit) as dense deltas D = new - T[name] (any rank, "
                      "any tensor in api.tensors() with editable true: linear weights, norms, dt_bias, A_log, conv1d). "
                      "Caps per tensor: >= 2-D ||D||_F/||W||_F <= weights.norm_cap_tau; 1-D rms(D) <= "
                      "weights.vector_cap_tau * max(rms(W), 1) (Qwen3.5 RMSNorm weights are zero-centred: the layer "
                      "multiplies by 1 + w); a write that is (almost) a uniform rescale of W is refused (use "
                      "w_scale for that); embed_tokens / final norm / lm_head are never editable",
    "candidate": "candidate(cid) -> {edit} (or {lora}) of an earlier candidate of this generation",
    "evaluate": "evaluate(edit=None, lora=None) -> structured screen feedback for an edit (a LoRA is evaluated as "
                "the edit that adds it; costs 1 evaluate; identical edits are free)",
    "train_loss": "train_loss(edit=None, lora=None, ids=None, n=64) -> {loss, parent_loss, delta, n}: mean answer NLL "
                  "of train items under the current model + edit vs without it (forward pass only; costs gpu_seconds)",
}


# Paper-style tools (2026-11-21; paper_tools_enabled): saved objects that survive between the
# exploration session and the final program of one candidate, the output-side matrices of a layer, vector algebra on
# refs, activation statistics on the current model's OWN screen outputs, and projecting a direction out of a module.
PAPER_DOCS = {
    "save_edit": "save_edit(handle, name) -> {name, kind, modules, max_rank}: save a LoRA / weight edit under `name` "
                 "for THIS candidate (it survives the end of run_python calls; your final program can return "
                 "api.load_edit(name)); the same name overwrites",
    "load_edit": "load_edit(name) -> {lora} or {edit}: a new handle of an object saved with save_edit for this candidate",
    "list_edits": "list_edits() -> list of {name, kind, modules, max_rank, created}: the objects saved for this candidate",
    "drop_edit": "drop_edit(name) -> {dropped}: delete a saved object of this candidate",
    "output_matrices": "output_matrices(layer) -> list of {module, kind, shape}: the modules of `layer` that write into "
                       "the residual stream (kind o_proj | out_proj | down_proj; shape [out, in], out = hidden size)",
    "unit": "unit(ref) -> {vector: 'T5.unit', size}: the vector ref normalised to length 1",
    "direction": "direction(a, b) -> {vector: 'T5.dir', size, norm}: (a - b) / ||a - b|| of two vector refs of equal "
                 "length (S / A / T refs, e.g. two acts handles' 'A3.out_mean' and 'A4.out_mean')",
    "project_out": "project_out(target, module, u, alpha=1.0) -> {lora} or {edit}: W <- W - alpha u u^T W on one linear "
                   "module (u = an OUTPUT-side vector ref of length out, normalised; W = the current weight with "
                   "target's change on that module): a LoRA target gets the rank-1 term B = -alpha u, A = u^T W; a "
                   "weight-edit target (mode weights) a low-rank op; the usual caps apply",
}
ACTS_OWN_DOC = ("acts(module, ids=None, k=4, source='prompt', sample=0) -> {handle, in_norm, out_norm, pc_var, tokens}: "
                "activation stats (vectors 'A.in_mean', 'A.out_mean', 'A.pc*') over all tokens of the given items. "
                "source='prompt': the item prompts (train ids, or screen ids); source='own': the prompt PLUS your own "
                "sampled answer on a screen item (thinking included; sample=0..n-1 or 'all'), ids must be screen ids "
                "(api.screen_results); contrast groups, e.g. right vs wrong items (costs gpu_seconds)")
TC_VECTORS_DOC = (" vectors=[refs] passes vector refs ('S3.u0', 'A2.out_mean', 'T5.dir', ...) to the code as "
                  "V = {ref: 1-D tensor}.")
# The current model's own answers on this generation's screen items (2026-11-21, screen_tools_enabled; free)
SCREEN_DOCS = {
    "screen_results": "screen_results(ids=None, filter='all', limit=50, offset=0) -> list of {id, prompt, gold, scores, "
                      "answers, finish, lengths, truncated_thinking}: YOUR OWN (the current model's) sampled answers "
                      "on this generation's screen items (training-pool items; prompt / gold may be read), one row "
                      "per item with one score / extracted answer per sample; filter all | wrong (no sample right) | "
                      "unstable (some right) | right (all right); free",
    "screen_output": "screen_output(id, sample=0, offset=0) -> {text, total_chars, next_offset, score, answer, finish, "
                     "has_thinking}: the full text of one of your sampled answers on a screen item (thinking "
                     "included), in pages of ~4000 characters (pass next_offset for the next page); free",
}


def tools_for_mode(mode: str, cfg: dict | None = None) -> dict:
    """TOOL_DOCS available to the editor in method.mode: edit = every LoRA tool incl. raw_lora; scratch = no
    raw_lora; weights = no raw_lora, plus WEIGHT_TOOLS (and edit-aware docs of candidate / evaluate / train_loss).
    cfg["tools"][name] = false removes a switchable tool (compare, train_loss, tensor_compute: also removes tensors).
    tensor_compute / tensors are on unless tensor_compute_enabled(cfg) is false."""
    out = dict(TOOL_DOCS)
    if not tensor_compute_enabled(cfg):
        out.pop("tensor_compute", None)
        out.pop("tensors", None)
    if mode == "weights":
        out.update({k: v for k, v in WEIGHTS_DOC_OVERRIDES.items() if k in out})
    else:
        for name in WEIGHT_TOOLS:
            out.pop(name, None)
    for name, on in ((cfg or {}).get("tools") or {}).items():
        if on is False:
            out.pop(name, None)
            if name == "tensor_compute":           # the info tool goes with it (2026-11-21)
                out.pop("tensors", None)
    if mode != "edit":
        out.pop("raw_lora", None)
    if mode == "scratch" and str(((cfg or {}).get("merge") or {}).get("mode") or "merge") == "adapter" and "stack" in out:
        out["stack"] = ("stack() -> list of {handle, gen, modules, max_rank} (adapters accepted in earlier generations, "
                        "kept as one cumulative adapter, not merged; already part of the current model)")
    ss = search_settings(cfg or {})
    if not ss["evaluate_tool"]:                # program_evaluate_calls 0 with uncharged screens (2026-11-21)
        out.pop("evaluate", None)
    if ss["gate_every_candidate"] and "compare" in out:
        out["compare"] = out["compare"].replace("level='screen') ->", "level='gate') ->") + \
            " (level 'gate': the gate evaluations of earlier candidates; ids and counts only)"
    if paper_tools_enabled(cfg):
        out.update(PAPER_DOCS)
        if "acts" in out:
            out["acts"] = ACTS_OWN_DOC
        if "tensor_compute" in out:
            out["tensor_compute"] = out["tensor_compute"].replace("tensor_compute(code, tensors", "tensor_compute(code, "
                                                                  "tensors, vectors=None", 1) + TC_VECTORS_DOC
    if screen_tools_enabled(cfg):
        out.update(SCREEN_DOCS)
    for name, on in ((cfg or {}).get("tools") or {}).items():   # the switches also remove the tools added here
        if on is False:
            out.pop(name, None)
    return out


def tensor_compute_enabled(cfg: dict | None) -> bool:
    """tools.tensor_compute (2026-11-21; base.yaml sets it). Absent key: on (the simple flow, 2026-11-22)."""
    on = ((cfg or {}).get("tools") or {}).get("tensor_compute")
    return bool(on) if on is not None else True


class BudgetError(Exception):
    pass


TC_DEFAULTS = {"max_tensors": 4, "timeout_s": 300, "mem_gb": 16, "threads": 4, "max_concurrent": 2,
               "result_chars": 4000, "explore": True, "max_input_mb": 8192}
_TC_SEMS: dict = {}
_TC_SEMS_LOCK = threading.Lock()


def tensor_compute_settings(cfg: dict) -> dict:
    """tensor_compute.* with code defaults (2026-11-21; absent keys keep runs locked before this resumable):
    max_tensors 4, timeout_s 300, mem_gb 16, threads 4, max_concurrent 2 (process-wide), result_chars 4000,
    explore true (also in the run_python exploration api), max_input_mb 8192 (fp32 MB of all input tensors)."""
    out = dict(TC_DEFAULTS)
    out.update({k: v for k, v in ((cfg or {}).get("tensor_compute") or {}).items() if v is not None})
    return out


def _tc_semaphore(n: int):
    with _TC_SEMS_LOCK:
        if n not in _TC_SEMS:
            _TC_SEMS[n] = threading.BoundedSemaphore(n)
        return _TC_SEMS[n]


def program_evaluate_cap(cfg: dict) -> int | None:
    """search.program_evaluate_calls (2026-11-21; code default 2, null = no cap): NEW api.evaluate calls one program
    may make (an identical, already evaluated LoRA is free; the automatic evaluation of the returned candidate is
    not a program call)."""
    v = (cfg.get("search") or {}).get("program_evaluate_calls", 2)
    return None if v is None else int(v)


def search_settings(cfg: dict) -> dict:
    """The search keys with code defaults that reproduce the runs before 2026-11-21 (base.yaml sets the simple flow's):
    threads, steps; charge_screens (default true: the automatic screen of a returned candidate costs one `evaluate`);
    pool = the generation's `evaluate` budget (charge_screens: search.max_evaluate; else search.program_evaluate_total,
    null = threads x program_evaluate_calls: the api.evaluate calls of all programs); program_cap (per program, None =
    no cap); max_candidates (null = max_evaluate while screens are charged, else threads x steps); parallel_builds."""
    sc = cfg.get("search") or {}
    threads, steps = int(sc.get("threads", 4)), int(sc.get("steps", 8))
    charge = sc.get("charge_screens", True)
    charge = True if charge is None else bool(charge)
    max_ev = int(sc.get("max_evaluate", 32))
    cap = program_evaluate_cap(cfg)
    if charge:
        pool = max_ev
    elif sc.get("program_evaluate_total") is not None:
        pool = int(sc["program_evaluate_total"])
    else:
        pool = threads * cap if cap is not None else max_ev
    # gate-per-candidate flow (2026-11-21, base.yaml; code defaults = the flow above): screen_candidates false =
    # candidates are not screened (the current model's screen is still computed once per generation and shown through
    # api.screen_results); gate_every_candidate true = every built candidate gets ONE gate evaluation paired vs the
    # current model and the generation accepts the passing one with the largest paired gain; program_evaluate_calls 0
    # with uncharged screens = no api.evaluate at all (evaluate_tool false)
    screen_c = sc.get("screen_candidates", True)
    screen_c = True if screen_c is None else bool(screen_c)
    charge = charge and screen_c
    evaluate_tool = charge or cap != 0
    if not charge:
        pool = 0 if not evaluate_tool else (int(sc["program_evaluate_total"]) if sc.get("program_evaluate_total")
                                            is not None else (threads * cap if cap is not None else max_ev))
    mc = sc.get("max_candidates")
    mc = int(mc) if mc is not None else (max_ev if charge else threads * steps)
    return {"threads": threads, "steps": steps, "charge_screens": charge, "pool": pool, "program_cap": cap,
            "max_candidates": mc, "parallel_builds": bool(sc.get("parallel_builds", False)),
            "screen_candidates": screen_c, "gate_every_candidate": bool(sc.get("gate_every_candidate", False)),
            "evaluate_tool": evaluate_tool}


def paper_tools_enabled(cfg: dict | None) -> bool:
    """tools.paper_tools (2026-11-21): save_edit / load_edit / list_edits / drop_edit, output_matrices, unit, direction,
    project_out, acts(source='own') and tensor_compute(vectors=...). base.yaml sets it; absent key: on (2026-11-22)."""
    on = ((cfg or {}).get("tools") or {}).get("paper_tools")
    return bool(on) if on is not None else True


def screen_tools_enabled(cfg: dict | None) -> bool:
    """tools.screen_results (2026-11-21): api.screen_results / api.screen_output over the current model's own answers
    on this generation's screen items. Absent key: on when search.screen_candidates is false (base.yaml)."""
    on = ((cfg or {}).get("tools") or {}).get("screen_results")
    return bool(on) if on is not None else not search_settings(cfg or {})["screen_candidates"]


def _budget_exc():
    try:
        from rsicore.sandbox import BudgetExceeded
        return BudgetExceeded
    except Exception:  # noqa: BLE001
        return BudgetError


class Workspace:
    def __init__(self, *, mock: bool, store, train, cfg: dict, out_dir: Path, evaluator=None, hf=None,
                 stack=None, budget: dict | None = None, name_prefix: str = "c",
                 diagnostics: dict | None = None, seed: int = 42, mode: str = "scratch", raw=None,
                 raw_info: dict | None = None, base_store=None, base_lora=None):
        self.mock = mock
        # merge.mode adapter: the cumulative LoRA C_g (current model = M_0 + C_g); None = the store / HF model is
        # the current model itself
        self.base_lora = base_lora if (base_lora is not None and base_lora.modules()) else None
        self._overlays: dict = {}
        # (2026-11-22, gpu_layout.tools_on_replicas) one tool device per main replica: [{"hf", "store", "lock",
        # "server"}]; a session (cid ..._t<k>_...) uses device k mod n, the same replica its editor runs on. Empty =
        # every tool on the train GPU's model under the workspace lock (as before).
        self.devs: list[dict] = []
        self._local = threading.local()
        self._xbudgets: dict[str, tuple[dict, dict]] = {}   # exploration GPU budget per session (cid)
        self._tl_n = 0
        self.mode = mode                        # method.mode: which tools are registered (tools_for_mode)
        self.raw = raw                          # mode=edit: the raw LoRA (handle "RAW")
        self.raw_info = dict(raw_info or {})
        self.store = store
        self.base_store = base_store if base_store is not None else store   # M_0's weights (w_toward_original)
        self.train = train
        self.cfg = cfg
        self.lcfg = cfg.get("lora", {})
        self.out_dir = Path(out_dir)
        self.evaluator = evaluator
        self.hf = hf
        self.stack = list(stack or [])          # [(gen, LoRA|MockLoRA)]
        from rsicore.sandbox import ToolAPI
        self.api = ToolAPI(dict(budget or {"evaluate": 32, "gpu_seconds": 7200}))   # the budget accountant
        self.budget = self.api.budget
        self.gpu_measured = 0.0
        self.name_prefix = name_prefix
        self.diag = diagnostics
        self.seed = seed
        self._h: dict[str, object] = {}
        self._owner: dict[str, str] = {}
        self._n = 0
        self._lock = threading.RLock()
        self._evaluated: dict[str, tuple] = {}
        self._prog_evals: dict[str, int] = {}     # cid -> new api.evaluate calls made by that program
        # (2026-11-21, set by run.py for the main search) charge_screens false: screen_candidate is free and the
        # `evaluate` budget is only the programs' api.evaluate pool; parallel_eval: candidates are built / screened
        # concurrently, so evaluate runs without the workspace lock in every mode (like mode=weights)
        self.charge_screens = True
        self.parallel_eval = False
        # (2026-11-21) the current model's own answers on this generation's screen items (dict or a
        # loader; Workspace.screen_data) for screen_results / screen_output / acts(source='own'); None = not available
        self.screen = None
        self.program_timeout_s = int((cfg.get("sandbox", {}) or {}).get("timeout_s", 1800))
        self.tool_log: dict[str, list] = {}
        self._registered = False
        self.rank_cap = int(self.lcfg.get("rank_cap", 64))
        self.tau = float(self.lcfg.get("norm_cap_tau", 0.1))
        self.allowed = self.lcfg.get("allowed_leaves")
        wcfg = cfg.get("weights", {}) or {}                # mode=weights: per-module relative change cap
        self.tau_w = float(wcfg.get("norm_cap_tau") if wcfg.get("norm_cap_tau") is not None else 0.05)
        self.max_modules_w = int(wcfg["max_modules"]) if wcfg.get("max_modules") is not None else None
        # 1-D tensors (norms, dt_bias, A_log): rms(D) <= vector_cap_tau * max(rms(W), 1); delta ops within rescale_tol
        # of a uniform rescale are refused (2026-11-21, code defaults 0.5 / 0.05)
        self.vtau_w = float(wcfg.get("vector_cap_tau") if wcfg.get("vector_cap_tau") is not None else 0.5)
        self.rescale_tol = float(wcfg.get("rescale_tol") if wcfg.get("rescale_tol") is not None else 0.05)
        self.tc = tensor_compute_settings(cfg)
        self._tc_n = 0
        for g, lo in self.stack:
            self._h[f"H{g}"] = lo
        if raw is not None:
            self._h["RAW"] = raw

    # ------------------------------------------------------------------ bookkeeping
    @property
    def spent(self) -> dict:
        return self.api.spent

    def left(self, kind: str) -> float:
        return self.budget.get(kind, math.inf) - self.api.spent.get(kind, 0)

    # ------------------------------------------------------------------ tool devices (2026-11-22)
    @property
    def store(self):
        """The weight store of the calling session's device (tools_on_replicas), else the workspace's."""
        d = self._dev()
        return self._overlay(d["store"] if d is not None and d.get("store") is not None else self._store0)

    def _overlay(self, s):
        """merge.mode adapter: the device's store seen through C_g (W_0 + B_C A_C, fp32); else the store itself."""
        if self.base_lora is None or self.mock or s is None:
            return s
        o = self._overlays.get(id(s))
        if o is None:
            from .primitives.weights import OverlayStore
            o = self._overlays.setdefault(id(s), OverlayStore(s, self.base_lora))
        return o

    def _hf_current(self):
        """Context in which the HF model computes the CURRENT model: merge.mode adapter applies C_g (forward hooks,
        entered before a tool's own statistic hooks so they see its output); otherwise nothing to do."""
        from contextlib import nullcontext
        if self.base_lora is None or self.mock or self.hf is None:
            return nullcontext()
        return self.hf.applied(self.base_lora)

    @store.setter
    def store(self, v):
        self._store0 = v

    @property
    def hf(self):
        """The HF model of the calling session's device (tools_on_replicas), else the workspace's (train GPU)."""
        d = self._dev()
        return d["hf"] if d is not None and d.get("hf") is not None else self._hf0

    @hf.setter
    def hf(self, v):
        self._hf0 = v

    def set_devices(self, devs: list[dict]) -> None:
        """One tool device per main replica, in the order of the editor endpoints (thread k -> replica k mod n):
        {"hf": HFModel on that replica's GPU | None, "store": WeightStore there | None, "server": the replica}."""
        self.devs = [dict(d, lock=d.get("lock") or threading.RLock()) for d in devs]

    def dev_index(self, cid) -> int | None:
        """Device of a session / candidate id (g01_t3_s00, g01_t3_s00~x...): thread mod the number of devices."""
        if not self.devs or not cid:
            return None
        m = re.search(r"_t(\d+)_", str(cid))
        return int(m.group(1)) % len(self.devs) if m else None

    def _dev(self) -> dict | None:
        i = getattr(self._local, "dev", None)
        return self.devs[i] if i is not None and i < len(self.devs) else None

    def _tool_lock(self):
        """Lock of a tool call: the session's device lock with tool devices (sessions on different replicas run
        their tools at once; shared bookkeeping keeps the workspace lock), else the workspace lock."""
        d = self._dev()
        return d["lock"] if d is not None else self._lock

    def warm_devices(self) -> None:
        """Load every device's HF model before the search (not charged to any budget). The loads run one after
        another (HFModel.model serialises from_pretrained process-wide); the first failure is raised, so a search
        never starts with a replica whose tools cannot run."""
        for i, d in enumerate(self.devs):
            if d.get("hf") is not None:
                try:
                    d["hf"].model
                except Exception as e:  # noqa: BLE001 - re-raised with the device named
                    raise RuntimeError(f"tool device {i} ({getattr(d['hf'], 'device', '?')}): HF copy failed to "
                                       f"load: {type(e).__name__}: {e}") from e

    def begin_program(self, cid: str, timeout_s: float | None = None) -> None:
        """timeout_s (2026-11-21): the program's wall clock (default sandbox.timeout_s); tools that start their own
        subprocess (tensor_compute) never run past it."""
        self._local.cid = cid
        self._local.dev = self.dev_index(cid)
        self._local.gpu = 0.0
        self._local.t0 = time.time()
        self._local.deadline = self._local.t0 + float(timeout_s if timeout_s is not None else self.program_timeout_s)
        self.tool_log[cid] = []

    def end_program(self, cid: str, keep: tuple = ()) -> None:
        with self._lock:
            for h in [h for h, o in self._owner.items() if o == cid and h not in keep]:
                self._h.pop(h, None)
                self._owner.pop(h, None)
        self._local.cid = None
        self._local.dev = None
        self._local.t0 = None
        self._local.deadline = None

    def _cid(self):
        return getattr(self._local, "cid", None) or "_"

    def _new(self, prefix: str, obj) -> str:
        with self._lock:
            self._n += 1
            h = f"{prefix}{self._n}"
            self._h[h] = obj
            self._owner[h] = self._cid()
            return h

    def _get(self, h, kind: str | None = None):
        if h is None:
            return None
        if isinstance(h, dict) and "lora" in h:
            h = h["lora"]
        elif isinstance(h, dict) and "edit" in h:
            h = h["edit"]
        if isinstance(h, str) and h.startswith("C:"):
            obj = self.load_candidate(h[2:])
            if obj is not None and kind == "edit" and not self._is_edit(obj):
                raise TypeError(f"candidate {h!r} is not a weight edit")
            if obj is not None and kind == "lora" and not self._is_lora(obj):
                raise TypeError(f"candidate {h!r} is not a LoRA")
            return obj
        if not isinstance(h, str) or h not in self._h:
            raise KeyError(f"unknown handle {h!r}")
        obj = self._h[h]
        if kind == "lora" and not self._is_lora(obj):
            raise TypeError(f"handle {h!r} is not a LoRA" + (" (it is a weight edit)" if self._is_edit(obj) else ""))
        if kind == "edit" and not self._is_edit(obj):
            raise TypeError(f"handle {h!r} is not a weight edit (use new_edit / w_add_lora)")
        return obj

    @staticmethod
    def is_edit_obj(obj) -> bool:
        """A weight edit (mode=weights): WeightEdit / MockWeightEdit, kind 'weights'."""
        return getattr(obj, "kind", "lora") == "weights"

    def _is_edit(self, obj):
        if self.mock:
            return isinstance(obj, M.MockWeightEdit)
        return self.is_edit_obj(obj) and hasattr(obj, "apply")

    def _is_lora(self, obj):
        if self.is_edit_obj(obj):
            return False
        from_mock = isinstance(obj, M.MockLoRA)
        if self.mock:
            return from_mock
        from .primitives.lora import LoRA
        return isinstance(obj, LoRA)

    def _empty(self):
        if self.mock:
            return M.MockLoRA()
        from .primitives.lora import LoRA
        return LoRA()

    def _empty_edit(self):
        if self.mock:
            return M.MockWeightEdit()
        from .primitives.wedit import WeightEdit
        return WeightEdit()

    def _as_edit(self, lo):
        """mode=weights: a LoRA as the weight edit that adds it (an edit is returned unchanged)."""
        if self.is_edit_obj(lo):
            return lo
        if self.mock:
            return M.m_w_add_lora(M.MockWeightEdit(), lo, 1.0) if lo.modules() else M.MockWeightEdit()
        from .primitives.wedit import lora_to_edit
        return lora_to_edit(lo)

    def _handle_of(self, lo) -> tuple[str, str]:
        """(result key, handle prefix) for an object: ('edit', 'W') for weight edits, ('lora', 'L') otherwise."""
        return ("edit", "W") if self.is_edit_obj(lo) else ("lora", "L")

    def _mods(self, modules) -> list[str]:
        if modules is None:
            return []
        if isinstance(modules, str):
            modules = [modules]
        allm = self.store.modules()
        out = []
        for pat in modules:
            if pat in ("all", "*"):
                out += allm
            elif self.store.has(pat):
                out.append(pat)
            else:
                hits = [m for m in allm if m.endswith("." + pat) or pat in m]
                if not hits:
                    raise KeyError(f"no module matches {pat!r}")
                out += hits
        return sorted(set(out), key=allm.index)

    def _ids(self, ids, default_n=None):
        if ids is None:
            ids = self.train.ids()
            if default_n:
                ids = ids[:default_n]
        return self.train.check(ids)

    def _log(self, name, kwargs):
        lst = self.tool_log.setdefault(self._cid(), [])
        lst.append(name)

    def _xbudget(self) -> tuple[dict, dict]:
        """(budget, spent) of the calling session's exploration GPU budget (editor.explore.gpu_seconds per session)."""
        return self._xbudgets.get(getattr(self._local, "xkey", None) or "", ({}, {}))

    def _gpu(self, fn, est_seconds: float, self_timed: bool = False):
        """Run fn and record its gpu seconds (mock: the estimate) for the tool's cost_fn. Inside explore_api the
        session's exploration budget is checked instead of the generation's. self_timed (2026-11-22): fn adds its
        own seconds to self._local.gpu (train_loss on a vLLM replica: the replica's request time only)."""
        if getattr(self._local, "explore", False):
            xb, xs = self._xbudget()
            if xs.get("gpu_seconds", 0.0) >= xb.get("gpu_seconds", 0.0) - 1e-9:
                raise _budget_exc()("exploration gpu_seconds budget of this session exhausted "
                                    "(editor.explore.gpu_seconds)")
        elif self.left("gpu_seconds") <= 0:
            raise _budget_exc()("gpu_seconds budget exhausted")
        if self.mock:
            self._local.gpu = getattr(self._local, "gpu", 0.0) + est_seconds
            return fn()
        if self_timed:
            return fn()
        t0 = self.hf.gpu_seconds
        try:
            return fn()
        finally:
            self._local.gpu = getattr(self._local, "gpu", 0.0) + (self.hf.gpu_seconds - t0)

    def _gpu_cost(self, kwargs, result) -> dict:
        """cost_fn of the GPU tools: seconds measured by _gpu during this call, capped at the remaining budget.
        (2026-11-22) The cap and the charge happen in one step under the ToolAPI lock: programs on several tool
        devices finish GPU calls at once, and a cap read before another call's charge landed would push spent past
        the budget and throw away a finished result."""
        used = float(getattr(self._local, "gpu", 0.0))
        self._local.gpu = 0.0
        with self.api._lock:
            self.gpu_measured += used
            self.api.charge("gpu_seconds", min(used, max(self.left("gpu_seconds"), 0.0)))
        return {}

    def _eval_obj(self, kwargs):
        """The object an evaluate / train_loss call refers to: kwargs edit (a weight edit) or lora (a LoRA; in
        mode=weights converted to the edit that adds it, so it matches the returned candidate's name)."""
        if kwargs.get("edit") is not None:
            if kwargs.get("lora") is not None:
                raise ValueError("pass either lora or edit, not both")
            return self._get(kwargs["edit"], "edit")
        if kwargs.get("lora") is None:
            raise ValueError("pass lora=<handle>" + (" or edit=<handle>" if self.mode == "weights" else ""))
        lo = self._get(kwargs["lora"], "lora")
        return self._as_edit(lo) if self.mode == "weights" else lo

    def _eval_key(self, kwargs):
        """dedupe_key_fn of evaluate: the LoRA's / edit's served name (digest), or "__keep__" when empty."""
        lo = self._eval_obj(kwargs)
        return self.lora_name(lo) if lo.modules() else "__keep__"

    def _eval_cost(self, kwargs, result) -> dict:
        """cost_fn of evaluate: 1 per non-empty LoRA (evaluating "keep" is free; so is a LoRA that was already
        evaluated before this call, e.g. screened automatically, 2026-11-21)."""
        cached, self._local.eval_cached = getattr(self._local, "eval_cached", False), False
        if cached:
            return {}
        return {} if self._eval_key(kwargs) == "__keep__" else {"evaluate": 1}

    def _note_screened(self, name: str | None, fb) -> None:
        """A free screen-level evaluation of `name` (charge_screens false): later api.evaluate calls of the same
        LoRA get it from the evaluate tool's dedupe cache, free and even when the pool is used up (2026-11-21)."""
        if name is None or self.charge_screens:
            return
        with self.api._lock:
            self.api._dedupe.setdefault(("evaluate", name), fb)

    # ------------------------------------------------------------------ registration
    GPU_TOOLS = ("acts", "train_loss")

    def build_api(self):
        if self._registered:
            return self.api
        docs = tools_for_mode(self.mode, cfg=self.cfg)
        if self.mode == "edit" and self.raw is None:
            docs.pop("raw_lora", None)
        for name, doc in docs.items():
            fn = self._wrap(name, getattr(self, "t_" + name))
            if name == "evaluate":
                self.api.register(name, fn, cost={"evaluate": 0}, cost_fn=self._eval_cost,
                                  dedupe_key_fn=self._eval_key, doc=doc)
            elif name in self.GPU_TOOLS:
                self.api.register(name, fn, cost={"gpu_seconds": 0}, cost_fn=self._gpu_cost, doc=doc)
            else:
                self.api.register(name, fn, doc=doc)
        self._registered = True
        return self.api

    # ------------------------------------------------------------------ exploration (editor chat tools)
    EXPLORE_NEVER = ("evaluate", "raw_lora")   # evaluate spends the generation's evaluate budget; raw_lora: the raw LoRA

    def explore_tool_names(self, gpu_tools: bool = False) -> list[str]:
        """Tools of explore_api: the program's tool set without EXPLORE_NEVER, and without GPU_TOOLS unless
        gpu_tools."""
        docs = tools_for_mode(self.mode, cfg=self.cfg)
        drop = set(self.EXPLORE_NEVER) | (set() if gpu_tools else set(self.GPU_TOOLS))
        if not self.tc.get("explore", True):     # tensor_compute.explore false: programs only
            drop |= {"tensor_compute", "tensors"}
        return [n for n in docs if n not in drop]

    def explore_api(self, tag: str, gpu_tools: bool = False, gpu_seconds: float = 300.0, persistent: bool = False):
        """(ToolAPI, cleanup) for one run_python call of the editor's tool loop (rsicore.chat_tools; 2026-11-21).

        A separate ToolAPI over the same tensors and handles (the _wrap'ed t_* functions, serialised by ws._lock),
        without evaluate / raw_lora (the handle 'RAW' still works) and, unless gpu_tools, without the GPU tools
        (acts / train_loss). It never charges the generation's budgets (ws.api.spent,
        gpu_measured): with gpu_tools the GPU tools are charged to a separate exploration budget of gpu_seconds per
        session (2026-11-22: each proposal / cid has its own; their parent-loss caches are shared with the
        programs). Runs under
        begin_program(tag) in the calling thread (the sandbox serves tool calls in it), so the handles made here
        are freed by cleanup() and the tool log lives under the '~x' tag, which classify() never reads.
        persistent (2026-11-21, editor.explore.persistent): -> (api, cleanup, before_call) for ONE api per proposal
        (rsicore.chat_tools keeps a sandbox kernel per proposal): handles live until cleanup() (the end of the
        proposal); before_call(timeout_s) starts each run_python call's wall clock (the tensor_compute deadline), and
        every tool call binds this tag's thread-local state first, so it works in whichever thread serves it."""
        from rsicore.sandbox import ToolAPI
        xkey = str(tag).split("~")[0]
        with self._lock:
            xb = self._xbudgets.setdefault(xkey, ({"gpu_seconds": float(gpu_seconds)}, {"gpu_seconds": 0.0}))
        api = ToolAPI({})
        if gpu_tools:
            api.budget, api.spent = xb          # this session's own budget (every run_python call of the proposal)
        docs = tools_for_mode(self.mode, cfg=self.cfg)
        clock = {"t0": time.time(), "deadline": time.time() + _explore_timeout(self.cfg)}

        def bind():
            """This tag's thread-local state in the calling thread (persistent: before every tool call)."""
            self._local.cid, self._local.t0, self._local.deadline = tag, clock["t0"], clock["deadline"]
            self._local.dev, self._local.xkey = self.dev_index(tag), xkey
            self._local.explore = True

        def bound(fn):
            def call(**kwargs):
                bind()
                return fn(**kwargs)
            call.__name__ = fn.__name__
            return call
        for name in self.explore_tool_names(gpu_tools):
            fn = self._wrap(name, getattr(self, "t_" + name))
            if persistent:
                fn = bound(fn)
            if name in self.GPU_TOOLS:
                api.register(name, fn, cost={"gpu_seconds": 0}, cost_fn=self._gpu_cost_explore, doc=docs[name])
            else:
                api.register(name, fn, doc=docs[name])
        self.begin_program(tag, timeout_s=_explore_timeout(self.cfg))
        self._local.explore = True
        self._local.xkey = xkey
        self._local.gpu = 0.0

        def cleanup():
            self._local.explore = False
            self._local.xkey = None
            self._local.gpu = 0.0
            self.end_program(tag)
        if not persistent:
            return api, cleanup

        def before_call(timeout_s=None):
            clock["t0"] = time.time()
            clock["deadline"] = clock["t0"] + float(timeout_s if timeout_s is not None else _explore_timeout(self.cfg))
            bind()
            self._local.gpu = 0.0
        return api, cleanup, before_call

    def _gpu_cost_explore(self, kwargs, result) -> dict:
        """cost_fn of the GPU tools in explore_api: the measured seconds go to the exploration budget (never to
        ws.api.spent or gpu_measured)."""
        with self._lock:
            used = float(getattr(self._local, "gpu", 0.0))
            self._local.gpu = 0.0
            xs = self._xbudget()[1]
            xs["gpu_seconds"] = xs.get("gpu_seconds", 0.0) + used
        return {}

    UNLOCKED_TOOLS = ("evaluate",)   # mode=weights: lock only their own bookkeeping (concurrent evaluations)
    ALWAYS_UNLOCKED = ("tensor_compute",)   # every mode: a CPU subprocess of up to minutes (2026-11-21)

    def _wrap(self, name, fn):
        def call(**kwargs):
            if name in self.ALWAYS_UNLOCKED or ((self.mode == "weights" or self.parallel_eval)
                                                and name in self.UNLOCKED_TOOLS):
                with self._lock:
                    self._log(name, kwargs)
                return fn(**kwargs)
            with self._tool_lock():
                self._log(name, kwargs)
                return fn(**kwargs)
        call.__name__ = name
        return call

    # ------------------------------------------------------------------ info tools
    def t_modules(self, layer=None, leaf=None, group=None):
        out = []
        for m in self.store.modules():
            inf = self.store.info(m)
            if layer is not None and inf["layer"] != int(layer):
                continue
            if leaf is not None and inf["leaf"] != leaf:
                continue
            if group is not None and inf["group"] != group:
                continue
            out.append(inf)
        return out

    def t_diagnostics(self, module=None, layer=None, top=20, sort="w_norm"):
        rows = (self.diag or {}).get("rows", [])
        if module is not None:
            rows = [r for r in rows if r["module"] == module]
        if layer is not None:
            rows = [r for r in rows if r["layer"] == int(layer)]
        if sort:
            def key(r):
                v = r.get(sort)
                if v is None and r.get("stack"):
                    v = r["stack"][-1].get(sort)
                return -(v if isinstance(v, (int, float)) else 0)
            rows = sorted(rows, key=key)
        return rows[: int(top)] if top else rows

    def t_train_ids(self, n=None):
        ids = self.train.ids()
        return ids[: int(n)] if n else ids

    def t_train_example(self, id):  # noqa: A002
        ex = self.train.get([id])[0]
        msgs = ex.get("messages") or []
        return {"prompt": "\n".join(m.get("content", "") for m in msgs)[:2000],
                "target": (ex.get("target") or "")[:2000], "source": ex.get("source")}

    def budget_status(self) -> dict:
        """Budgets of this generation for the step context (2026-11-21). charge_screens false: `evaluate` is the
        programs' api.evaluate pool (the automatic screens are free)."""
        return {"evaluate_left": self.left("evaluate"), "evaluate": self.budget.get("evaluate"),
                "gpu_seconds_left": round(self.left("gpu_seconds"), 1), "gpu_seconds": self.budget.get("gpu_seconds"),
                "program_evaluate_calls": program_evaluate_cap(self.cfg), "charge_screens": bool(self.charge_screens),
                "evaluate_tool": "evaluate" in self.api.names() if self._registered else None}

    def t_budget(self):
        out = {"spent": dict(self.api.spent), "left": self.api.remaining()}
        cid, t0 = getattr(self._local, "cid", None), getattr(self._local, "t0", None)
        if cid and "~x" not in str(cid) and t0:          # a candidate program (not run_python exploration)
            out["program_seconds_left"] = round(max(0.0, self.program_timeout_s - (time.time() - t0)), 1)
            cap = program_evaluate_cap(self.cfg)
            if cap is not None:
                left = max(0, cap - self._prog_evals.get(cid, 0))
                if not self.charge_screens:             # the generation's pool can be the tighter limit
                    left = int(max(0, min(left, self.left("evaluate"))))
                out["program_evaluate_calls_left"] = left
        if getattr(self._local, "explore", False):      # (2026-11-22) run_python: the session's own GPU budget
            xb, xs = self._xbudget()
            if xb:
                out["explore_gpu_seconds_left"] = round(max(0.0, xb.get("gpu_seconds", 0.0)
                                                            - xs.get("gpu_seconds", 0.0)), 1)
                out["explore_note"] = ("GPU tools in run_python spend explore_gpu_seconds_left (this session only); "
                                       "'spent' / 'left' are your final program's budgets")
        if not self.charge_screens and self._registered and "evaluate" not in self.api.names():
            out["note"] = ("no api.evaluate in this run: every returned candidate gets one gate evaluation "
                           "automatically (shown to later steps / generations)")
        elif not self.charge_screens:
            out["note"] = ("returned candidates are scored automatically at no cost; 'evaluate' is the api.evaluate "
                           "pool of all programs of this generation")
        return out

    def t_stack(self):
        return [{"handle": f"H{g}", "gen": g, "modules": len(lo.modules()), "max_rank": lo.max_rank()}
                for g, lo in self.stack]

    def t_raw_lora(self):
        if self.raw is None:
            raise KeyError("no raw LoRA in this session")
        return {"lora": "RAW", "modules": len(self.raw.modules()), "max_rank": self.raw.max_rank(),
                "train": self.raw_info}

    def t_candidate(self, cid):
        lo = self.load_candidate(cid)
        if lo is None:
            raise KeyError(f"candidate {cid!r} has no saved LoRA / edit")
        key, prefix = self._handle_of(lo)
        return {key: self._new(prefix, lo)}

    # ------------------------------------------------------------------ weight-space tools
    def t_svd(self, module, k=8):
        k = int(min(max(1, k), 64))
        if self.mock:
            s1 = 10 + (int(stable_hash(module), 16) % 100) / 20
            S = [round(s1 * 0.85 ** i, 4) for i in range(k)]
            return {"handle": self._new("S", {"module": module, "S": S}), "S": S, "shape": list(self.store.shape(module))}
        U, S, V = self.store.svd(module, k)
        return {"handle": self._new("S", {"module": module, "U": U, "S": S, "V": V}), "S": [float(x) for x in S],
                "shape": list(self.store.shape(module))}

    def t_acts(self, module, ids=None, k=4, source="prompt", sample=0):
        """source (2026-11-21, paper tools): 'prompt' = the item prompts (train ids; screen ids too when the screen
        tools are on); 'own' = the prompt + the current model's own sampled answer(s) on screen items."""
        if source not in ("prompt", "own"):
            raise ValueError("source must be 'prompt' or 'own'")
        if not self.store.has(module):
            raise KeyError(f"unknown module {module!r}")
        texts = None
        if source == "own":
            texts = self._screen_texts(ids, own=True, sample=sample)
        else:
            try:
                ids = self._ids(ids, default_n=64)
            except PermissionError:
                if getattr(self, "screen", None) is None:
                    raise
                texts = self._screen_texts(ids, own=False)
        n = len(texts) if texts is not None else len(ids)
        if self.mock:
            # screen-item groups get distinct (deterministic) stats; train-id calls keep the old mock values
            salt = 0.0 if texts is None else 0.1 * (int(stable_hash(str(source), str(sample), str(ids)), 16) % 10)
            def run():
                return {"module": module, "in_norm": 12.0 + salt, "out_norm": 3.0 + salt,
                        "pc_var": [4.0 / (i + 1) for i in range(k)]}
            st = self._gpu(run, est_seconds=2 + 0.05 * n * (4 if source == "own" else 1))
            return {"handle": self._new("A", st), "in_norm": st["in_norm"], "out_norm": st["out_norm"],
                    "pc_var": st["pc_var"], "tokens": (800 if source == "own" else 200) * n}
        if texts is None:
            texts = self.train.texts(ids)
        def run_acts():
            with self._hf_current():
                return self.hf.acts(texts, [module], k=int(k))[module]
        st = self._gpu(run_acts, est_seconds=0)
        st["module"] = module
        return {"handle": self._new("A", st), "in_norm": st["in_norm"], "out_norm": st["out_norm"],
                "pc_var": st["pc_var"], "tokens": st["tokens"]}

    def _examples(self, ids):
        ex = [e for e in self.train.get(ids) if e.get("target")]
        if not ex:
            raise ValueError("none of these train items has a training target")
        return ex

    def _vec(self, ref: str, dim: int):
        """Resolve a vector reference to a tensor of size dim (mock: None after validation)."""
        if not isinstance(ref, str) or "." not in ref:
            if isinstance(ref, list) and len(ref) == dim:
                if self.mock:
                    return None
                import torch
                return torch.tensor(ref, dtype=torch.float32)
            raise ValueError(f"bad vector reference {ref!r}")
        h, rest = ref.split(".", 1)
        with self._lock:
            obj = self._get(h)
        if h.startswith("T"):                   # tensor_compute vectors (2026-11-21; mock: validated, then None)
            vs = obj.get("vectors", {}) if isinstance(obj, dict) else {}
            if rest not in vs:
                raise ValueError(f"handle {h!r} has no vector {rest!r} (it has {sorted(vs)[:10]})")
            v = vs[rest]
            if v.numel() != dim:
                raise ValueError(f"vector {ref!r} has size {v.numel()}, need {dim}")
            return None if self.mock else v
        if self.mock:
            return None
        if h.startswith("S"):
            side, i = rest[0], int(rest[1:])
            v = obj["U"][:, i] if side == "u" else obj["V"][:, i]
        elif h.startswith("A"):
            if rest in ("in_mean", "out_mean"):
                v = obj[rest]
            elif rest.startswith("pc"):
                v = obj["pcs"][int(rest[2:])]
            else:
                raise ValueError(f"bad activation field {rest!r}")
        else:
            raise ValueError(f"handle {h!r} has no vectors")
        if v.numel() != dim:
            raise ValueError(f"vector {ref!r} has size {v.numel()}, need {dim}")
        return v

    # ------------------------------------------------------------------ constructors
    def t_new_lora(self):
        return {"lora": self._new("L", self._empty())}

    def t_rank1(self, lora, module, u, v, s):
        base = self._get(lora, "lora")
        out_f, in_f = self.store.shape(module)
        uu, vv = self._vec(u, out_f), self._vec(v, in_f)
        if self.mock:
            return {"lora": self._new("L", M.m_construct(base, module, s, "rank1"))}
        from .primitives.construct import rank1
        lo = base.copy()
        rank1(lo, module, uu, vv, float(s), shape=(out_f, in_f))
        return {"lora": self._new("L", lo)}

    def t_from_direction(self, lora, module, direction, key, alpha):
        base = self._get(lora, "lora")
        out_f, in_f = self.store.shape(module)
        d, kk = self._vec(direction, out_f), self._vec(key, in_f)
        if self.mock:
            return {"lora": self._new("L", M.m_construct(base, module, alpha, "from_direction"))}
        from .primitives.construct import from_direction
        lo = base.copy()
        from_direction(lo, module, d, kk, float(alpha), shape=(out_f, in_f))
        return {"lora": self._new("L", lo)}

    def t_scale_singular(self, lora, module, svd, idx, factors):
        base = self._get(lora, "lora")
        s = self._get(svd)
        if s["module"] != module:
            raise ValueError(f"svd handle is for {s['module']}, not {module}")
        if self.mock:
            f = factors if isinstance(factors, list) else [factors] * len(idx)
            strength = sum(abs(float(x) - 1) for x in f) * 5
            return {"lora": self._new("L", M.m_construct(base, module, strength, "scale_singular"))}
        from .primitives.construct import scale_singular
        lo = base.copy()
        scale_singular(lo, module, s["U"], s["S"], s["V"], idx, factors)
        return {"lora": self._new("L", lo)}

    # ------------------------------------------------------------------ operators
    def _op(self, mock_fn, real_fn, *los):
        if self.mock:
            return {"lora": self._new("L", mock_fn(*los))}
        out = real_fn(*los)
        if out.max_rank() > self.rank_cap:
            out.recompress(self.rank_cap)
        return {"lora": self._new("L", out)}

    def t_scale(self, lora, s, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        return self._op(lambda a: M.m_scale(a, float(s)), lambda a: ops.scale(a, float(s), modules), lo)

    def t_interpolate(self, a, b, lam):
        from .primitives import ops
        A, B = self._get(a, "lora"), self._get(b, "lora")
        return self._op(lambda x, y: M.m_combine([x, y], [1 - float(lam), float(lam)]),
                        lambda x, y: ops.interpolate(x, y, float(lam), cap=self.rank_cap), A, B)

    def t_combine(self, loras, weights=None):
        from .primitives import ops
        los = [self._get(h, "lora") for h in loras]
        w = [float(x) for x in (weights or [1.0] * len(los))]
        if self.mock:
            return {"lora": self._new("L", M.m_combine(los, w))}
        return {"lora": self._new("L", ops.combine(los, w, cap=self.rank_cap))}

    def t_drop(self, lora, modules):
        from .primitives import ops
        lo = self._get(lora, "lora")
        if self.mock:
            out = lo.copy()
            for m in self._mods(modules):
                out.modules_.pop(m, None)
            out.ops.append(["drop", modules])
            return {"lora": self._new("L", out)}
        return {"lora": self._new("L", ops.drop(lo, modules))}

    def t_truncate_rank(self, lora, k, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        return self._op(lambda a: M.m_truncate(a, int(k)), lambda a: ops.truncate_rank(a, int(k), modules), lo)

    def t_intruder_scale(self, lora, lam=0.5, eps=0.3, k0=32, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        return self._op(lambda a: M.m_intruder(a, float(lam)),
                        lambda a: ops.intruder_scale(a, self.store, float(lam), float(eps), int(k0), modules), lo)

    def t_relchange_prune(self, lora, frac=0.03, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        frac = float(min(max(frac, 0.0), 0.5))
        return self._op(lambda a: M.m_prune(a, frac),
                        lambda a: ops.relchange_prune(a, self.store, frac, modules,
                                                      device=getattr(self.hf, "device", "cpu") if self.hf else "cpu"), lo)

    def t_orthogonalize(self, lora, against="stack"):
        from .primitives import ops
        lo = self._get(lora, "lora")
        if against == "stack":
            others = [x for _, x in self.stack]
        else:
            others = [self._get(h, "lora") for h in (against if isinstance(against, list) else [against])]
        if self.mock:
            return {"lora": self._new("L", M.m_orth(lo, len(others)))}
        return {"lora": self._new("L", ops.orthogonalize(lo, others))}

    def t_ties(self, loras, weights=None, density=0.2):
        from .primitives import ops
        los = [self._get(h, "lora") for h in loras]
        w = [float(x) for x in (weights or [1.0] * len(los))]
        if self.mock:
            return {"lora": self._new("L", M.m_ties(los, w, float(density)))}
        return {"lora": self._new("L", ops.ties(los, w, float(density), cap=self.rank_cap,
                                                device=getattr(self.hf, "device", "cpu") if self.hf else "cpu"))}

    # ------------------------------------------------------------------ weight edits (mode=weights)
    def _wnew(self, ed):
        return {"edit": self._new("W", ed)}

    def _one_module(self, module) -> str:
        if not isinstance(module, str) or not self.store.has(module):
            raise KeyError(f"unknown module {module!r} (give one exact module name from api.modules())")
        return module

    @staticmethod
    def _finite(x, what: str) -> float:
        v = float(x)
        if not math.isfinite(v):
            raise ValueError(f"{what} must be finite, got {x!r}")
        return v

    def _channels(self, module, factors, idx, dim, what):
        """Validate a channel scaling: idx (None = all `dim` channels) and factors (float or list)."""
        if idx is not None:
            if isinstance(idx, int):
                idx = [idx]
            idx = [int(i) for i in idx]
            if not idx:
                raise ValueError(f"{what}: empty index list")
            bad = [i for i in idx if not 0 <= i < dim]
            if bad:
                raise ValueError(f"{what}: indices {bad[:5]} out of range for {module} ({dim} channels)")
            if len(set(idx)) != len(idx):
                raise ValueError(f"{what}: duplicate indices")
        if isinstance(factors, (list, tuple)):
            f = [self._finite(x, "factors") for x in factors]
            n = dim if idx is None else len(idx)
            if len(f) != n:
                raise ValueError(f"{what}: {len(f)} factors for {n} channels")
        else:
            f = self._finite(factors, "factors")
        return idx, f

    def t_new_edit(self):
        return self._wnew(self._empty_edit())

    def t_w_scale(self, edit, modules, s):
        ed = self._get(edit, "edit")
        mods = self._mods(modules)
        if not mods:
            raise ValueError("w_scale needs at least one module")
        s = self._finite(s, "s")
        if self.mock:
            return self._wnew(M.m_w_scale(ed, mods, s))
        out = ed.copy()
        for m in mods:
            out.add(m, {"op": "scale", "s": s})
        return self._wnew(out)

    def _w_channels(self, kind, edit, module, factors, idx):
        ed = self._get(edit, "edit")
        module = self._one_module(module)
        out_f, in_f = self.store.shape(module)
        dim = out_f if kind == "rows" else in_f
        idx, f = self._channels(module, factors, idx, dim, f"w_scale_{kind}")
        if self.mock:
            fn = M.m_w_rows if kind == "rows" else M.m_w_cols
            return self._wnew(fn(ed, module, f, idx=idx, dim=dim))
        out = ed.copy()
        out.add(module, {"op": kind, "idx": idx, "f": f})
        return self._wnew(out)

    def t_w_scale_rows(self, edit, module, factors, rows=None):
        return self._w_channels("rows", edit, module, factors, rows)

    def t_w_scale_cols(self, edit, module, factors, cols=None):
        return self._w_channels("cols", edit, module, factors, cols)

    def t_w_add_lora(self, edit, lora, alpha=1.0):
        ed = self._get(edit, "edit")
        lo = self._get(lora, "lora")
        alpha = self._finite(alpha, "alpha")
        if self.mock:
            return self._wnew(M.m_w_add_lora(ed, lo, alpha))
        out = ed.copy()
        for m in lo.modules():
            A, B = lo.factors[m]
            out.add(m, {"op": "lowrank", "A": A.detach().clone(), "B": (B * alpha).detach().clone()})
        return self._wnew(out)

    def _base_rel(self, mods) -> dict:
        """mock: ||W - W_0|| / ||W|| per module from the merged stack (earlier generations)."""
        out = {}
        for m in mods:
            r2 = sum(lo.modules_[m]["rel"] ** 2 for _, lo in self.stack if m in getattr(lo, "modules_", {}))
            if r2 > 0:
                out[m] = math.sqrt(r2)
        return out

    def t_w_toward_original(self, edit, modules, alpha):
        ed = self._get(edit, "edit")
        mods = self._mods(modules)
        if not mods:
            raise ValueError("w_toward_original needs at least one module")
        alpha = self._finite(alpha, "alpha")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if self.mock:
            undo = 0.0
            for _, lo in self.stack:            # mock: the merged gain on these modules is (partly) undone
                lm = getattr(lo, "modules_", {})
                if lm:
                    undo += alpha * lo.gain * len(set(mods) & set(lm)) / len(lm)
            return self._wnew(M.m_w_toward(ed, mods, alpha, base_rel=self._base_rel(mods), undo_gain=undo))
        out = ed.copy()
        for m in mods:
            out.add(m, {"op": "toward_base", "alpha": alpha})
        return self._wnew(out)

    def t_inspect_edit(self, edit):
        return self.report(self._get(edit, "edit"))

    # ------------------------------------------------------------------ tensor tools (2026-11-21)
    def t_tensors(self, layer=None, pattern=None):
        import fnmatch

        from .primitives.naming import tensor_layer
        glob = isinstance(pattern, str) and any(c in pattern for c in "*?[")
        out = []
        for n in self.store.tensor_names():
            if layer is not None and tensor_layer(n) != int(layer):
                continue
            if pattern and not (fnmatch.fnmatchcase(n, pattern) if glob else str(pattern) in n):
                continue
            inf = self.store.tensor_info(n)
            out.append({k: inf[k] for k in ("name", "shape", "dtype", "editable")})
        return out

    def _time_left(self) -> float | None:
        dl = getattr(self._local, "deadline", None)
        return None if dl is None else dl - time.time()

    def _tc_current(self, name: str):
        """fp32 CPU value of a tensor of the CURRENT model (mock: the seeded synthetic tensor)."""
        if self.mock:
            return M.synth_tensor(self.store.synth_spec(name))
        return self.store.tensor(name, device="cpu")

    def t_tensor_compute(self, code, tensors=None, edit=None, vectors=None):
        """Run `code` on fp32 copies of current tensors in a fresh sandbox; register its outputs (see TOOL_DOCS).
        vectors (2026-11-21, paper tools): vector refs passed to the code as V = {ref: 1-D fp32 tensor}."""
        import shutil

        from rsicore.sandbox import ToolAPI, run_program
        tc = self.tc
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string of Python")
        names = [tensors] if isinstance(tensors, str) else list(tensors or [])
        vrefs = [vectors] if isinstance(vectors, str) else list(vectors or [])
        if vrefs and not paper_tools_enabled(self.cfg):
            raise ValueError("vectors= is not available in this run (tools.paper_tools)")
        if len(vrefs) > 16:
            raise ValueError("at most 16 vectors per call")
        if not names and not vrefs:
            raise ValueError("tensors: give at least one tensor name (api.tensors())")
        if edit is not None and self.mode != "weights":
            raise ValueError("edit= is only available in mode=weights")
        with self._lock:                                   # resolve inputs
            canon, aliases = [], {}
            for n in names:
                if not isinstance(n, str) or not self.store.has_tensor(n):
                    raise KeyError(f"unknown tensor {n!r} (names from api.tensors(), or a module name)")
                c = self.store.tensor_info(n)["name"]
                if c not in canon:
                    canon.append(c)
                if n != c:
                    aliases[n] = c                         # a module name also keys T (same tensor)
            if len(canon) > int(tc["max_tensors"]):
                raise ValueError(f"at most {tc['max_tensors']} tensors per call (tensor_compute.max_tensors)")
            shapes = {n: tuple(self.store.tensor_shape(n)) for n in canon}
            mb = sum(4 * math.prod(s) for s in shapes.values()) / 2 ** 20
            if mb > float(tc["max_input_mb"]):
                raise ValueError(f"inputs are {mb:.0f} MB in fp32 > tensor_compute.max_input_mb {tc['max_input_mb']}")
            base_ed = self._get(edit, "edit") if edit is not None else None
            vin = {f"V|{r}": self._vec_t(r).contiguous() for r in vrefs}
            self._tc_n += 1
            n_call = self._tc_n
        cur = {}                                           # tensors the given edit changes: T = W after its ops
        if base_ed is not None and not self.mock:
            for n in canon:
                en = self.store.edit_name(n)
                if en in base_ed.ops:
                    wb = None
                    if base_ed.needs_base(en):
                        wb = (self.base_store.weight(en) if self.base_store.has(en) else self.base_store.tensor(en))
                    cur[n] = base_ed.apply(en, self._tc_current(n), wb).contiguous()
        if not self.mock and hasattr(self.store, "is_overlaid"):   # merge.mode adapter: W_0 + B_C A_C, not the disk W_0
            for n in canon:
                if n not in cur and self.store.is_overlaid(n):
                    cur[n] = self._tc_current(n).contiguous()
        left = self._time_left()
        timeout = float(tc["timeout_s"]) if left is None else min(float(tc["timeout_s"]), left - 5.0)
        if timeout < 5:
            raise _budget_exc()("tensor_compute: not enough time left in this program / exploration call")
        sem = _tc_semaphore(max(1, int(tc["max_concurrent"])))
        t_wait = time.time()
        if not sem.acquire(timeout=timeout):
            raise _budget_exc()(f"tensor_compute: {tc['max_concurrent']} jobs already running for {timeout:.0f} s "
                                "(tensor_compute.max_concurrent)")
        timeout -= time.time() - t_wait
        wd = self.out_dir / "work" / "tensor_compute" / f"{self._cid().replace('/', '_')}_{n_call}"
        try:
            if timeout < 5:
                raise _budget_exc()("tensor_compute: not enough time left after waiting for a free slot")
            if wd.exists():
                shutil.rmtree(wd, ignore_errors=True)
            wd.mkdir(parents=True, exist_ok=True)
            thr = max(1, int(tc["threads"]))
            spec = {"code": code, "threads": thr, "names": canon, "aliases": aliases, "inputs": sorted(cur),
                    "vin": list(vrefs)}
            allow = []
            if self.mock:
                spec["synth_src"] = M.SYNTH_SRC
                spec["synth"] = {n: self.store.synth_spec(n) for n in canon if n not in cur}
            else:
                spec["sources"] = {n: list(self.store.tensor_source(n)) for n in canon if n not in cur}
                allow = sorted({p for p, _ in spec["sources"].values()})
            if cur or vin:
                from safetensors.torch import save_file
                save_file({**cur, **vin}, str(wd / "in.safetensors"))
            (wd / "tc_spec.json").write_text(json.dumps(spec))
            res = run_program(TC_WRAPPER, ToolAPI({}), timeout_s=max(5, int(timeout)), mem_gb=float(tc["mem_gb"]),
                              workdir=wd, extra_allow_read=allow, cpu_s=int(timeout * thr) + 30,
                              extra_env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(thr),
                                         "MKL_NUM_THREADS": str(thr), "OPENBLAS_NUM_THREADS": str(thr)})
            if not res.ok:
                err = (res.error or "failed").replace(str(wd) + "/", "")
                raise RuntimeError(f"tensor_compute: {err[-2500:]}" + (f"\nstderr: {res.stderr[-800:]}"
                                                                         if res.stderr and not res.error else ""))
            man = res.result or {}
            outs = {}
            if (wd / "out.safetensors").exists():
                from safetensors.torch import load_file
                outs = load_file(str(wd / "out.safetensors"))
            return self._tc_register(man, outs, canon, shapes, cur, base_ed, res)
        finally:
            sem.release()
            shutil.rmtree(wd, ignore_errors=True)

    def _tc_register(self, man, outs, canon, shapes, cur, base_ed, res):
        """Validate the job's outputs and register them (vectors -> T handle, lora -> L handle, writes -> W handle);
        nothing is registered when any output is invalid."""
        import torch

        from .primitives.wedit import change_errors, rescale_ratio
        out = {"seconds": round(res.seconds, 2)}
        rtxt = man.get("result")
        if rtxt is not None:
            lim = int(self.tc["result_chars"])
            if len(rtxt) <= lim:
                out["json"] = json.loads(rtxt)
            else:
                out["json"], out["json_truncated"] = rtxt[:lim], True
        if res.stdout:
            out["stdout"] = res.stdout[-1500:]

        def finite(t, what):
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"tensor_compute: {what} has non-finite values")
            return t
        vecs = {}
        for k in man.get("vectors") or []:
            v = outs[f"v|{k}"]
            if v.dim() != 1 or not 0 < v.numel() <= 2 ** 20:
                raise ValueError(f"tensor_compute: vectors[{k!r}] must be 1-D (got shape {tuple(v.shape)})")
            vecs[k] = finite(v, f"vectors[{k!r}]")
        if len(vecs) > 64:
            raise ValueError("tensor_compute: at most 64 vectors per call")
        factors, info = {}, {}
        for m in man.get("lora") or []:
            if not self.store.has(m):
                raise KeyError(f"tensor_compute: lora module {m!r} is not a linear module of api.modules()")
            A, B = outs[f"A|{m}"], outs[f"B|{m}"]
            out_f, in_f = self.store.shape(m)
            if A.dim() != 2 or B.dim() != 2 or A.shape[0] != B.shape[1] or A.shape[1] != in_f or B.shape[0] != out_f:
                raise ValueError(f"tensor_compute: lora[{m!r}] needs A [r, {in_f}] and B [{out_f}, r], got "
                                 f"A{tuple(A.shape)} B{tuple(B.shape)}")
            if A.shape[0] > self.rank_cap:
                raise ValueError(f"tensor_compute: lora[{m!r}] rank {A.shape[0]} > rank cap {self.rank_cap}")
            factors[m] = (finite(A, f"lora[{m!r}] A"), finite(B, f"lora[{m!r}] B"))
            if self.mock:
                info[m] = (int(A.shape[0]), float((B @ A).norm()) / self.store.norm(m))
        writes = {}
        names = man.get("writes") or []
        if names and self.mode != "weights":
            raise ValueError("tensor_compute: writes are only available in mode=weights (build a lora instead)")
        for raw in names:
            n = self.store.tensor_info(raw)["name"] if self.store.has_tensor(raw) else raw
            if n not in canon:
                raise ValueError(f"tensor_compute: writes[{n!r}]: write only tensors listed in tensors= ({canon})")
            if not self.store.editable(n):
                raise PermissionError(f"tensor_compute: {n} is never editable (embed_tokens, final norm, lm_head)")
            new = finite(outs[f"w|{raw}"], f"writes[{n!r}]")
            if tuple(new.shape) != shapes[n]:
                raise ValueError(f"tensor_compute: writes[{n!r}] has shape {tuple(new.shape)}, need {shapes[n]}")
            W = cur[n] if n in cur else self._tc_current(n)
            D = (new - W).contiguous()
            if not bool(D.abs().max() > 0):
                continue                                   # unchanged: nothing to write
            r = rescale_ratio(D, W)
            if r is not None and r < self.rescale_tol:
                raise ValueError(f"tensor_compute: writes[{n!r}] is (almost) a uniform rescale of the tensor "
                                 f"(||D - cW||/||D|| = {r:.4f} < weights.rescale_tol {self.rescale_tol}); use w_scale")
            errs, stats = change_errors(n, D, W, tau=self.tau_w, vector_tau=self.vtau_w)
            if errs:
                raise ValueError("tensor_compute: " + errs[0])
            writes[n] = (D, stats)
        with self._lock:                                   # register outputs
            if vecs:
                h = self._new("T", {"vectors": vecs})
                out["vectors"] = {k: f"{h}.{k}" for k in vecs}
                out["vector_sizes"] = {k: int(v.numel()) for k, v in vecs.items()}
            if factors:
                if self.mock:
                    lo = M.m_tensor_lora(info)
                else:
                    from .primitives.lora import LoRA
                    lo = LoRA()
                    for m, (A, B) in factors.items():
                        lo.add(m, A, B)
                ok, rep = self.check(lo)
                out["lora"] = self._new("L", lo)
                out["lora_check"] = {"ok": ok, "errors": rep.get("errors", [])[:5]}
                self._log("tensor_compute.lora", {})
            if writes:
                ed = base_ed if base_ed is not None else self._empty_edit()
                if self.mock:
                    for n, (D, st) in writes.items():
                        rel = st["rel_norm"] if D.dim() >= 2 else st["rms_change"] * self.vtau_w / st["rms_cap"]
                        ed = M.m_w_delta(ed, self.store.edit_name(n), rel, D.dim(), tau_ratio=self.tau_w / self.vtau_w)
                else:
                    ed = ed.copy()
                    for n, (D, _) in writes.items():
                        ed.add(self.store.edit_name(n), {"op": "delta", "D": D})
                out["edit"] = self._new("W", ed)
                out["writes"] = {n: {k: round(v, 6) for k, v in st.items()} for n, (_, st) in writes.items()}
                self._log("tensor_compute.writes", {})
        return out

    # ------------------------------------------------------------------ paper-style tools (2026-11-21)
    MAX_SAVED = 32                          # saved objects per candidate
    _SAVE_TOOLS = ("save_edit", "load_edit", "list_edits", "drop_edit")

    def _saved_dir(self) -> Path:
        """out_dir/saved/<root cid>/: the candidate's saved objects (root cid = the program's cid, or the cid of the
        exploration tag '<cid>~x...'), shared by its exploration session and its final program; kept on disk."""
        root = str(self._cid()).split("~", 1)[0]
        if root in ("", "_") or "/" in root or "\\" in root or root.startswith("."):
            raise RuntimeError("save_edit / load_edit work only inside a candidate's program or exploration")
        return self.out_dir / "saved" / root

    def _saved_paths(self, name) -> tuple[Path, Path]:
        import re
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("name: 1-64 characters (letters, digits, '_', '-', '.'; not starting with '.' or '-')")
        d = self._saved_dir()
        return d / (f"{name}.json" if self.mock else f"{name}.safetensors"), d / f"{name}.meta.json"

    def t_save_edit(self, handle, name):
        obj = self._get(handle)
        if not (self._is_lora(obj) or self._is_edit(obj)):
            raise TypeError(f"handle {handle!r} is not a LoRA or a weight edit")
        p, pm = self._saved_paths(name)
        if not pm.exists() and len(list(p.parent.glob("*.meta.json"))) >= self.MAX_SAVED:
            raise ValueError(f"at most {self.MAX_SAVED} saved objects per candidate: drop_edit one first")
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.mock:
            write_json(p, obj.to_json())
        else:
            obj.save_raw(p)
        kind = "edit" if self.is_edit_obj(obj) else "lora"
        tools = [t for t in self.tool_log.get(self._cid(), []) if t not in self._SAVE_TOOLS]
        meta = {"name": name, "kind": kind, "modules": len(obj.modules()), "max_rank": obj.max_rank(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tools": tools, "saved_by": str(self._cid())}
        write_json(pm, meta)
        return {k: meta[k] for k in ("name", "kind", "modules", "max_rank")}

    def t_load_edit(self, name):
        p, pm = self._saved_paths(name)
        if not (p.exists() and pm.exists()):
            have = [x.name[:-len(".meta.json")] for x in sorted(p.parent.glob("*.meta.json"))] if p.parent.exists() else []
            raise KeyError(f"no saved object {name!r} for this candidate (saved: {have[:20]})")
        meta = json.loads(pm.read_text())
        if self.mock:
            d = json.loads(p.read_text())
            obj = (M.MockWeightEdit if d.get("kind") == "weights" else M.MockLoRA).from_json(d)
        elif meta.get("kind") == "edit":
            from .primitives.wedit import WeightEdit
            obj = WeightEdit.load_raw(p)
        else:
            from .primitives.lora import LoRA
            obj = LoRA.load_raw(p)
        # the program's strategy (classify) counts the tools that built the saved object (saved elsewhere, e.g. in
        # the exploration session; the program's own earlier calls are already in its log)
        if meta.get("saved_by") != str(self._cid()):
            self.tool_log.setdefault(self._cid(), []).extend(meta.get("tools") or [])
        key, prefix = self._handle_of(obj)
        return {key: self._new(prefix, obj)}

    def t_list_edits(self):
        d = self._saved_dir()
        out = []
        for pm in sorted(d.glob("*.meta.json")) if d.exists() else []:
            try:
                m = json.loads(pm.read_text())
            except (OSError, ValueError):
                continue
            out.append({k: m.get(k) for k in ("name", "kind", "modules", "max_rank", "created")})
        return out

    def t_drop_edit(self, name):
        p, pm = self._saved_paths(name)
        if not pm.exists():
            raise KeyError(f"no saved object {name!r} for this candidate")
        for x in (p, pm):
            x.unlink(missing_ok=True)
        return {"dropped": name}

    OUTPUT_LEAVES = ("o_proj", "out_proj", "down_proj")

    def t_output_matrices(self, layer):
        out = []
        for m in self.store.modules():
            inf = self.store.info(m)
            if inf["layer"] == int(layer) and inf["leaf"] in self.OUTPUT_LEAVES:
                out.append({"module": m, "kind": inf["leaf"], "shape": [int(inf["out"]), int(inf["in"])]})
        return out

    def _vec_size(self, ref) -> int:
        """Length of a vector ref (S / A / T handle field) without materialising it."""
        if isinstance(ref, list):
            return len(ref)
        if not isinstance(ref, str) or "." not in ref:
            raise ValueError(f"bad vector reference {ref!r}")
        h, rest = ref.split(".", 1)
        obj = self._get(h)
        if h.startswith("T"):
            vs = obj.get("vectors", {}) if isinstance(obj, dict) else {}
            if rest not in vs:
                raise ValueError(f"handle {h!r} has no vector {rest!r} (it has {sorted(vs)[:10]})")
            return int(vs[rest].numel())
        if not isinstance(obj, dict) or "module" not in obj:
            raise ValueError(f"handle {h!r} has no vectors")
        out_f, in_f = self.store.shape(obj["module"])
        if h.startswith("S"):
            return out_f if rest[:1] == "u" else in_f
        if h.startswith("A"):
            return out_f if rest == "out_mean" else in_f
        raise ValueError(f"handle {h!r} has no vectors")

    def _vec_t(self, ref):
        """fp32 CPU tensor of a vector ref (mock: T vectors as they are, S / A refs as seeded synthetic vectors of
        the right length)."""
        import torch
        n = self._vec_size(ref)
        if isinstance(ref, str) and ref.split(".", 1)[0].startswith("T"):
            return self._get(ref.split(".", 1)[0])["vectors"][ref.split(".", 1)[1]].float().cpu()
        v = self._vec(ref, n)
        if v is None:                               # mock: validated, no real vector behind S / A refs
            g = torch.Generator().manual_seed(int(stable_hash("vec", str(ref)), 16) % (2 ** 31))
            v = torch.randn(n, generator=g)
        return v.detach().float().cpu().reshape(-1)

    def t_unit(self, ref):
        v = self._vec_t(ref)
        nv = float(v.norm())
        if not nv > 0 or not math.isfinite(nv):
            raise ValueError(f"vector {ref!r} has norm {nv}: cannot normalise it")
        h = self._new("T", {"vectors": {"unit": v / nv}})
        return {"vector": f"{h}.unit", "size": int(v.numel()), "norm": round(nv, 6)}

    def t_direction(self, a, b):
        va, vb = self._vec_t(a), self._vec_t(b)
        if va.numel() != vb.numel():
            raise ValueError(f"{a!r} has length {va.numel()} and {b!r} length {vb.numel()}: they must be equal (the "
                             "same side of the same module, or two acts handles of one module)")
        d = va - vb
        nd = float(d.norm())
        if not nd > 0 or not math.isfinite(nd):
            raise ValueError(f"direction({a!r}, {b!r}): the two vectors are identical (norm {nd})")
        h = self._new("T", {"vectors": {"dir": d / nd}})
        return {"vector": f"{h}.dir", "size": int(d.numel()), "norm": round(nd, 6)}

    def t_project_out(self, target, module, u, alpha=1.0):
        obj = self._get(target)
        if not self.store.has(module):
            raise KeyError(f"unknown module {module!r} (a linear module of api.modules())")
        is_edit = self._is_edit(obj)
        if not (is_edit or self._is_lora(obj)):
            raise TypeError(f"target {target!r} is not a LoRA or a weight edit")
        a = self._finite(alpha, "alpha")
        out_f, in_f = self.store.shape(module)
        if self._vec_size(u) != out_f:
            raise ValueError(f"u {u!r} has length {self._vec_size(u)}, need the OUTPUT side of {module} ({out_f})")
        if self.mock:
            if is_edit:
                self._log("project_out.edit", {})
                ed = M.m_w_add_lora(obj, M.m_construct(M.MockLoRA(), module, a, "project_out"), 1.0)
                return {"edit": self._new("W", ed)}
            return {"lora": self._new("L", M.m_construct(obj, module, a, "project_out"))}
        uu = self._vec_t(u)
        uu = uu / max(float(uu.norm()), 1e-12)
        W = self.store.weight(module, device="cpu").float()
        if is_edit:
            wb = self.base_store.weight(module, device="cpu").float() if obj.needs_base(module) else None
            W = obj.apply(module, W, wb).cpu()
        elif module in obj.factors:
            W = W + obj.delta(module).float().cpu()
        A = (W.T @ uu).reshape(1, in_f).contiguous()
        B = (-a * uu).reshape(out_f, 1).contiguous()
        if is_edit:
            ed = obj.copy()
            ed.add(module, {"op": "lowrank", "A": A, "B": B})
            self._log("project_out.edit", {})
            return {"edit": self._new("W", ed)}
        lo = obj.copy()
        lo.add(module, A, B)
        return {"lora": self._new("L", lo)}

    # ---- the current model's own answers on this generation's screen items (2026-11-21)
    def screen_data(self) -> dict:
        """{"records": {dataset: [run_eval records]}, "items": {dataset: {id: {prompt, reference, messages}}}} set by
        run.py (Run.reference_screen) before the search; RuntimeError when this session has none."""
        sd = getattr(self, "screen", None)
        if callable(sd):
            with self._lock:            # sessions on several tool devices may ask at once (2026-11-22)
                sd = self.screen
                if callable(sd):
                    sd = self.screen = sd()
        if not sd or not sd.get("records"):
            raise RuntimeError("no screen results of the current model in this session")
        return sd

    def _screen_rec(self, id_):
        d = self.screen_data()
        for ds in sorted(d["records"]):
            for r in d["records"][ds]:
                if str(r.get("id")) == str(id_):
                    return ds, r, (d.get("items") or {}).get(ds, {}).get(r["id"]) or {}
        raise PermissionError(f"{id_!r} is not one of this generation's screen ids (api.screen_results)")

    def t_screen_results(self, ids=None, filter="all", limit=50, offset=0):  # noqa: A002
        from rsicore import screen_view as SV
        d = self.screen_data()
        return SV.rows(d["records"], d.get("items"), ids=ids, filter=filter, limit=min(max(int(limit), 0), 200),
                       offset=offset)

    def t_screen_output(self, id, sample=0, offset=0):  # noqa: A002
        from rsicore import screen_view as SV
        return SV.output(self._screen_rec(id)[1], sample=sample, offset=offset)

    def _screen_texts(self, ids, own: bool, sample=0) -> list[str]:
        """acts texts of screen items: the prompt (own False) or the prompt + one / every sampled answer (thinking
        included) of the current model (own True)."""
        from rsicore import screen_view as SV
        if ids is None:
            d = self.screen_data()
            ids = [r["id"] for ds in sorted(d["records"]) for r in d["records"][ds] if r.get("score") is not None][:64]
        ids = [ids] if isinstance(ids, str) else list(ids)
        if not ids:
            raise ValueError("ids: give at least one screen id")
        texts = []
        for i in ids:
            _, rec, item = self._screen_rec(i)
            msgs = [m for m in (item.get("messages") or []) if isinstance(m, dict)]
            prompt = "\n\n".join(str(m.get("content") or "") for m in msgs) or SV.prompt_text(item)
            if not own:
                texts.append(prompt)
                continue
            outs, reas = SV._samples(rec)[:2]
            n = len(outs)
            pick = range(n) if sample == "all" else [int(sample)]
            for k in pick:
                if not 0 <= k < n:
                    raise ValueError(f"sample must be 0..{n - 1} or 'all'")
                think = f"<think>\n{reas[k]}\n</think>\n\n" if reas[k] else ""
                texts.append(prompt + "\n\n" + think + str(outs[k] or ""))
        return texts

    # ------------------------------------------------------------------ inspection / checks
    def _store_device(self):
        return getattr(self.store, "device", None) or (getattr(self.hf, "device", "cpu") if self.hf else "cpu")

    def check_edit(self, ed) -> tuple[bool, dict]:
        """Checks of a weight edit: existing modules, finite values, ||W'-W||_F/||W||_F <= weights.norm_cap_tau per
        module, at most weights.max_modules modules."""
        if self.mock:
            from .primitives.naming import is_edit_excluded
            errs, mods = [], {}
            n_ops = _ops_per_module(ed.ops)
            for m, v in ed.modules_.items():
                linear = self.store.has(m)
                if not linear and not (hasattr(self.store, "has_tensor") and self.store.has_tensor(m)):
                    errs.append(f"{m}: not in model")
                elif not linear and is_edit_excluded(m):
                    errs.append(f"{m}: never editable (embed_tokens, the final norm and lm_head are excluded)")
                one_d = v.get("ndim", 2) < 2         # tensor_compute writes of 1-D tensors (2026-11-21)
                if not math.isfinite(v["rel"]):
                    errs.append(f"{m}: non-finite values")
                elif one_d and v["rel"] > self.vtau_w:
                    errs.append(f"{m}: rms(W'-W) / max(rms(W), 1) = {v['rel']:.4f} > {self.vtau_w} (1-D tensor)")
                elif not one_d and v["rel"] > self.tau_w:
                    errs.append(f"{m}: ||W'-W||/||W|| = {v['rel']:.4f} > tau {self.tau_w}")
                mods[m] = {"rel_norm": v["rel"], "ops": n_ops.get(m, 0)}
            if self.max_modules_w is not None and len(mods) > self.max_modules_w:
                errs.append(f"{len(mods)} modules edited > max_modules {self.max_modules_w}")
            return not errs, {"errors": errs[:50], "n_errors": len(errs), "modules": mods}
        from .primitives.wedit import check_wedit
        return check_wedit(ed, self.store, self.base_store, tau=self.tau_w, max_modules=self.max_modules_w,
                           device=self._store_device(), vector_tau=self.vtau_w, rescale_tol=self.rescale_tol)

    def check(self, lo) -> tuple[bool, dict]:
        if self.is_edit_obj(lo):
            return self.check_edit(lo)
        if self.mock:
            errs, mods = [], {}
            for m, v in lo.modules_.items():
                if not self.store.has(m):
                    errs.append(f"{m}: not in model")
                if v["rank"] > self.rank_cap:
                    errs.append(f"{m}: rank {v['rank']} > cap {self.rank_cap}")
                if v["rel"] > self.tau:
                    errs.append(f"{m}: ||dW||/||W0|| = {v['rel']:.4f} > tau {self.tau}")
                mods[m] = {"rank": v["rank"], "rel_norm": v["rel"]}
            return not errs, {"errors": errs[:50], "n_errors": len(errs), "modules": mods}
        from .primitives.lora import check_lora
        return check_lora(lo, self.store, rank_cap=self.rank_cap, tau=self.tau, allowed_leaves=self.allowed)

    def report_edit(self, ed) -> dict:
        ok, rep = self.check_edit(ed)
        mods = rep.get("modules", {})
        top = sorted(mods.items(), key=lambda kv: -kv[1].get("rel_norm", 0))[:8]
        return {"kind": "weights", "modules": len(mods), "max_rank": 0, "check_ok": ok,
                "check_errors": rep.get("errors", [])[:5],
                "top_rel_norm": [{"module": m, **{k: round(v, 5) if isinstance(v, float) else v for k, v in d.items()}}
                                 for m, d in top],
                "total_rel_norm": round(math.sqrt(sum(d.get("rel_norm", 0) ** 2 for d in mods.values())), 5),
                "ops": ed.ops[-6:] if self.mock else _trunc(ed.summary())}

    def report(self, lo, max_intruder_modules: int = 16) -> dict:
        if self.is_edit_obj(lo):
            return self.report_edit(lo)
        ok, rep = self.check(lo)
        mods = rep.get("modules", {})
        top = sorted(mods.items(), key=lambda kv: -kv[1].get("rel_norm", 0))[:8]
        out = {"modules": len(mods), "max_rank": lo.max_rank(), "check_ok": ok, "check_errors": rep.get("errors", [])[:5],
               "top_rel_norm": [{"module": m, **{k: round(v, 5) if isinstance(v, float) else v for k, v in d.items()}}
                                for m, d in top],
               "total_rel_norm": round(math.sqrt(sum(d.get("rel_norm", 0) ** 2 for d in mods.values())), 5)}
        if self.mock:
            out["intruder_mean"] = round(0.35 + 0.1 * (lo.damage > lo.gain), 3)
            out["ops"] = lo.ops[-6:]
        elif mods:
            from .primitives.ops import intruder_scores
            al = []
            for m, _ in top[:max_intruder_modules]:
                U0, _, _ = self.store.svd(m, 32)
                _, a = intruder_scores(lo, m, U0)
                al += a
            out["intruder_mean"] = round(sum(al) / len(al), 4) if al else None
            if self.stack:
                out["stack_overlap"] = _overlap(lo, [x for _, x in self.stack])
        return out

    def t_inspect(self, lora):
        lo = self._get(lora, "lora")
        return self.report(lo)

    # ------------------------------------------------------------------ evaluation
    def lora_name(self, lo) -> str:
        return f"{self.name_prefix}-{lo.digest()}"

    def evaluate_lora(self, lo, level: str = "screen", charge: bool = True) -> tuple[float | None, dict]:
        """Register + score `lo` (empty LoRA = keep the model unchanged). Identical LoRAs are cached.
        charge=True at level 'screen' goes through the `evaluate` tool of ws.api (budget + dedupe there)."""
        if charge and level == "screen":
            self.build_api()
            key, prefix = self._handle_of(lo)
            h = self._new(prefix, lo)
            try:
                fb = self.api.call("evaluate", {key: h})
            finally:
                with self._lock:
                    self._h.pop(h, None)
                    self._owner.pop(h, None)
            return fb.get("score"), fb
        # Several threads may evaluate at once (run.py, mode=weights): the lock covers only the cache and the
        # report / check (weight store, applying an edit); evaluator.register / feedback run concurrently.
        with self._lock:
            empty = not lo.modules()
            name = None if empty else self.lora_name(lo)
            key = (name, level)
            if key in self._evaluated:
                return self._evaluated[key]
            rep = self.report(lo) if not empty else {"modules": 0, "keep": True}
            if not empty and not rep["check_ok"]:
                fb = {"error": ("edit" if self.is_edit_obj(lo) else "LoRA") + " failed the sandbox checks",
                      "lora_stats": rep}
                self._evaluated[key] = (None, fb)
                return None, fb
            pending = getattr(self._local, "eval_pending", 0)   # the evaluate tool charges after this returns
            cost = {"evaluate_left": self.left("evaluate") - pending,
                    "gpu_seconds_left": round(self.left("gpu_seconds"), 1),
                    "gpu_seconds_spent": round(self.spent.get("gpu_seconds", 0), 1)}
        if name is not None and not self.evaluator.known(name):
            self.evaluator.register(lo, name)
        score, fb = self.evaluator.feedback(name, level, lora_report=rep, cost=cost)
        with self._lock:
            self._evaluated[key] = (score, fb)
        if level == "screen":
            self._note_screened(name, fb)
        return score, fb

    def t_compare(self, cid, other="parent", level="screen"):
        """Item-level comparison of two already-evaluated models on the same items; ids and statistics only."""
        if level not in ("screen", "full", "gate"):
            raise ValueError("level must be 'screen', 'full' or 'gate'")

        def name_of(c):
            if c in (None, "parent"):
                return None
            lo = self.load_candidate(str(c))
            if lo is None:
                raise KeyError(f"unknown candidate {c!r}")
            return self.lora_name(lo) if lo.modules() else None
        A, B = self.evaluator.cached(name_of(cid), level), self.evaluator.cached(name_of(other), level)
        if A is None or B is None:
            raise ValueError(f"{[c for c, r in ((cid, A), (other, B)) if r is None]} not evaluated at level {level!r} "
                             "yet (compare only reads existing results; it never evaluates)")
        out = {"cid": cid, "other": other, "level": level, "datasets": {}}
        for ds in sorted(set(A) & set(B)):
            a = {r["id"]: r for r in A[ds]["records"] if r.get("score") is not None}
            b = {r["id"]: r for r in B[ds]["records"] if r.get("score") is not None}
            ids = sorted(set(a) & set(b))
            broken = [i for i in ids if b[i]["score"] > 0.5 >= a[i]["score"]]
            fixed = [i for i in ids if a[i]["score"] > 0.5 >= b[i]["score"]]
            lens = sorted(b[i].get("len", 0) for i in ids)
            c1, c2 = (lens[len(lens) // 3], lens[2 * len(lens) // 3]) if lens else (0, 0)

            def bucket(i):
                n = b[i].get("len", 0)
                return "short" if n <= c1 else ("medium" if n <= c2 else "long")
            by = {k: {"n": 0, "broken": 0, "fixed": 0} for k in ("short", "medium", "long")}
            for i in ids:
                by[bucket(i)]["n"] += 1
            for i in broken:
                by[bucket(i)]["broken"] += 1
            for i in fixed:
                by[bucket(i)]["fixed"] += 1
            n = max(len(ids), 1)
            out["datasets"][ds] = {
                "n": len(ids), "acc": round(sum(a[i]["score"] for i in ids) / n, 4),
                "acc_other": round(sum(b[i]["score"] for i in ids) / n, 4),
                "broken": len(broken), "fixed": len(fixed), "net": len(fixed) - len(broken),
                "broken_ids": broken[:50], "fixed_ids": fixed[:50], "by_other_length": by,
                "truncated": round(sum(_trunc_frac(a[i]) for i in ids) / n, 4),
                "truncated_other": round(sum(_trunc_frac(b[i]) for i in ids) / n, 4)}
        return out

    def _parent_lora_for_loss(self):
        """The LoRA that makes the current model in the HF model: mode=edit after generation 0 the current LoRA
        (RAW, source 'current') on M_0; merge.mode adapter (real) the cumulative LoRA C_g on M_0 (a candidate is then
        scored as C_g (+) delta, _loss_lora); otherwise none (the HF model is the current model itself)."""
        if self.base_lora is not None and not self.mock:
            return self.base_lora
        return self.raw if (self.raw is not None and self.raw_info.get("source") == "current") else None

    def _loss_lora(self, lo):
        """The adapter whose loss is the candidate's: merge.mode adapter = the composed C_g (+) lo that the evaluator
        serves (same composition, same rank budget); else lo (mode=edit: a replacement of the current LoRA)."""
        if self.base_lora is None or self.mock or self.is_edit_obj(lo) or not lo.modules():
            return lo
        if self.evaluator is not None:
            return self.evaluator.compose(lo)
        from .primitives.lora import compose_cumulative
        return compose_cumulative(self.base_lora, lo)

    def loss_backend(self, lo, ref) -> str:
        """'vllm' (tools.train_loss_backend vllm, 2026-11-22): the LoRA is hot-loaded on the session's own main
        replica and the answer NLL read from vLLM's prompt logprobs; 'hf': the HF model (weight edits, a parent LoRA
        in mode=edit, mock, or the backend set to hf)."""
        b = str((self.cfg.get("tools") or {}).get("train_loss_backend") or "hf")
        if (b != "vllm" or self.mock or (ref is not None and ref is not self.base_lora) or self.evaluator is None
                or self.is_edit_obj(lo)):
            return "hf"
        try:
            srv = self._loss_replica()
        except Exception:  # noqa: BLE001 - no replica to score on
            return "hf"
        # an adapter above the replica's --max-lora-rank (after vLLM's rank padding) cannot be loaded there: ranks
        # can pass lora.rank_cap through rank1 / from_direction / scale_singular / project_out (no recompress)
        from .primitives.lora import padded_rank
        cap = int(getattr(srv, "max_lora_rank", None) or self.rank_cap)
        lo = self._loss_lora(lo) if lo is not None else lo     # adapter mode: the composed rank counts
        if lo is not None and lo.modules() and padded_rank(max(1, int(lo.max_rank()))) > cap:
            return "hf"
        return "vllm"

    def _loss_replica(self):
        """The calling session's main replica: its tool device's server, else replica (thread mod n) of the
        evaluator, the same replica its editor runs on."""
        d = self._dev()
        if d is not None and d.get("server") is not None:
            return d["server"]
        reps = list(getattr(self.evaluator, "replicas", None) or [self.evaluator.server])
        m = re.search(r"_t(\d+)_", str(self._cid()))
        return reps[int(m.group(1)) % len(reps)] if m else reps[0]

    def _loss_seconds_left(self) -> float:
        """Seconds a vLLM train_loss may still use: the session's exploration budget (run_python) or the
        generation's gpu_seconds, minus what this tool call already used, and the program's / run_python call's
        wall clock."""
        used = float(getattr(self._local, "gpu", 0.0))
        if getattr(self._local, "explore", False):
            xb, xs = self._xbudget()
            lim = [xb.get("gpu_seconds", 0.0) - xs.get("gpu_seconds", 0.0) - used]
        else:
            lim = [self.left("gpu_seconds") - used]
        dl = getattr(self._local, "deadline", None)
        if dl:
            lim.append(dl - time.time())
        return max(1.0, min(lim))

    def _vllm_loss(self, ex, lo) -> float:
        """Mean answer-token NLL of ex under the current model (lo None / empty) or the current model + lo, on the
        session's own replica (lo saved as a PEFT adapter, loaded there under a unique name, unloaded and deleted
        after, also on errors). Charged: the adapter load and the requests' time (not the disk writes or retry
        waits), and the requests stop at the budget / wall-clock limit (_loss_seconds_left)."""
        import shutil
        srv = self._loss_replica()
        deadline = time.time() + self._loss_seconds_left()
        enc = [self.hf.encode(e) for e in ex]       # the HF tokenizer and chat template of HFModel.loss
        name, path = None, None
        try:
            if lo is not None and lo.modules():
                from .primitives.lora import save_peft
                with self._lock:
                    self._tl_n += 1
                    name = f"tl_{self.name_prefix}_{self._tl_n}"
                path = self.out_dir / "train_loss_adapters" / name
                save_peft(lo, path, self.store, base_model=str(getattr(self.store, "model_dir", "")),
                          dtype=self.lcfg.get("save_dtype", "bfloat16"))
                t0 = time.time()
                try:
                    srv.load_lora(name, path)
                finally:
                    self._local.gpu = getattr(self._local, "gpu", 0.0) + (time.time() - t0)
            parent = getattr(self.evaluator, "parent_lora", None) or self.evaluator.served   # M_g: C_g's adapter
            nll, secs = vllm_answer_nll(srv.endpoint, name or parent, enc, deadline=deadline)
            self._local.gpu = getattr(self._local, "gpu", 0.0) + secs
            return nll
        finally:
            if name is not None:
                try:
                    srv.unload_lora(name)       # tolerates an adapter that never got loaded
                except Exception:  # noqa: BLE001 - a stale adapter only takes a CPU slot
                    pass
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

    def t_train_loss(self, lora=None, ids=None, n=64, edit=None):
        """Mean answer NLL on train items with and without the LoRA / weight edit (forward only). Real mode with an
        edit: HFModel.loss(ex, lora=<WeightEdit>) applies the edit. tools.train_loss_backend vllm (2026-11-22): a LoRA
        is scored on the session's own vLLM replica (loss_backend)."""
        lo = self._eval_obj({"lora": lora, "edit": edit})
        ids = self._ids(ids, default_n=int(n))
        ex = self._examples(ids)
        ref = self._parent_lora_for_loss()
        backend = self.loss_backend(lo, ref)
        key = tuple(sorted(ids)) if backend == "hf" else ("vllm",) + tuple(sorted(ids))
        cache = self.__dict__.setdefault("_parent_loss", {})
        if self.mock:
            def mloss(x):
                return 1.2 if x is None else round(1.2 - 2.0 * x.gain + 1.0 * x.damage, 6)
            loss = self._gpu(lambda: mloss(lo if lo.modules() else ref), est_seconds=1 + 0.05 * len(ex))
            if key not in cache:
                cache[key] = mloss(ref)
        elif backend == "vllm":
            loss = self._gpu(lambda: self._vllm_loss(ex, self._loss_lora(lo)), est_seconds=0, self_timed=True)
            if key not in cache:
                cache[key] = self._gpu(lambda: self._vllm_loss(ex, None), est_seconds=0, self_timed=True)
        else:
            loss = self._gpu(lambda: self.hf.loss(ex, lora=self._loss_lora(lo) if lo.modules() else ref), est_seconds=0)
            if key not in cache:
                cache[key] = self._gpu(lambda: self.hf.loss(ex, lora=ref), est_seconds=0)
        p = cache[key]
        return {"loss": round(float(loss), 6), "parent_loss": round(float(p), 6),
                "delta": round(float(loss) - float(p), 6), "n": len(ex)}

    def t_evaluate(self, lora=None, edit=None):
        with self._lock:
            lo = self._eval_obj({"lora": lora, "edit": edit})
            cached = bool(lo.modules()) and (self.lora_name(lo), "screen") in self._evaluated
            cid, cap = getattr(self._local, "cid", None), program_evaluate_cap(self.cfg)
            if cid and cap is not None and lo.modules() and not cached:
                n = self._prog_evals.get(cid, 0)       # a program's own api.evaluate calls (2026-11-21)
                if n >= cap:
                    raise _budget_exc()(f"api.evaluate: at most {cap} new evaluations per program "
                                        "(search.program_evaluate_calls); return your best LoRA instead")
                self._prog_evals[cid] = n + 1
        # charge_screens false: already evaluated (e.g. a free automatic screen) = no charge (_eval_cost reads it
        # after this returns); charged screens keep the accounting of before (the ToolAPI dedupe only)
        free = cached and not self.charge_screens
        self._local.eval_cached = free
        self._local.eval_pending = 1 if (lo.modules() and not free) else 0
        try:
            _, fb = self.evaluate_lora(lo, "screen", charge=False)   # ws.api charges (cost_fn) and dedupes
        finally:
            self._local.eval_pending = 0
        return fb

    def screen_candidate(self, cid: str) -> tuple[float | None, dict]:
        """screen_fn helper: screen evaluation of a saved candidate, charged on ws.api (budget exhausted -> (None,
        error)); free when charge_screens is false (2026-11-21)."""
        try:
            return self.evaluate_lora(self.load_candidate(cid), "screen", charge=bool(self.charge_screens))
        except _budget_exc() as e:
            return None, {"error": f"not evaluated: {e}"}

    # ------------------------------------------------------------------ candidates
    def cand_path(self, cid: str, kind: str = "lora") -> Path:
        """loras/<cid>.json | .safetensors (LoRA), loras/<cid>.wedit.json | .wedit.safetensors (weight edit)."""
        ext = ".json" if self.mock else ".safetensors"
        return self.out_dir / "loras" / (f"{cid}.wedit{ext}" if kind == "weights" else f"{cid}{ext}")

    def save_candidate(self, cid: str, lo) -> Path:
        p = self.cand_path(cid, "weights" if self.is_edit_obj(lo) else "lora")
        if self.mock:
            write_json(p, lo.to_json())
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            lo.save_raw(p)
        return p

    def load_candidate(self, cid: str):
        """The saved LoRA or weight edit of a candidate (None if neither exists)."""
        p, pe = self.cand_path(cid), self.cand_path(cid, "weights")
        if p.exists():
            if self.mock:
                return M.MockLoRA.from_json(json.load(open(p)))
            from .primitives.lora import LoRA
            return LoRA.load_raw(p)
        if pe.exists():
            if self.mock:
                return M.MockWeightEdit.from_json(json.load(open(pe)))
            from .primitives.wedit import WeightEdit
            return WeightEdit.load_raw(pe)
        return None

    def resolve_result(self, result):
        """main(api) return value -> LoRA object (empty = keep), or raise ValueError. mode=weights: -> weight edit
        ({"edit": h}; {"lora": h} is converted to the edit that adds it; None = empty edit = keep)."""
        if self.mode == "weights":
            return self._resolve_edit(result)
        if result is None:
            raise ValueError("main(api) returned None; return {'lora': handle} (or {'lora': None} to keep the model)")
        h = result.get("lora", "__missing__") if isinstance(result, dict) else result
        if h == "__missing__":
            raise ValueError("main(api) must return {'lora': <handle>}")
        if h is None:
            return self._empty()
        return self._get(h, "lora")

    def _resolve_edit(self, result):
        if result is None:
            raise ValueError("main(api) returned None; return {'edit': handle} (or {'edit': None} to keep the model)")
        if isinstance(result, dict):
            if "edit" in result:
                h = result["edit"]
                return self._empty_edit() if h is None else self._get(h, "edit")
            if "lora" in result:
                h = result["lora"]
                return self._empty_edit() if h is None else self._as_edit(self._get(h, "lora"))
            raise ValueError("main(api) must return {'edit': <handle>}")
        if isinstance(result, str):                 # a bare handle
            obj = self._get(result)
            if self._is_edit(obj):
                return obj
            if self._is_lora(obj):
                return self._as_edit(obj)
            raise TypeError(f"handle {result!r} is neither a weight edit nor a LoRA")
        raise ValueError("main(api) must return {'edit': <handle>}")


def _trunc_frac(rec: dict) -> float:
    """Share of a compact evaluation record's samples cut at max_tokens (avg@n records carry "trunc")."""
    from .evaluator import _trunc
    return _trunc(rec)


def _explore_timeout(cfg: dict) -> float:
    """editor.explore.timeout_s (the run_python wall clock; default 300)."""
    try:
        from rsicore.editor import editor_settings
        return float(editor_settings(cfg)["explore"]["timeout_s"])
    except Exception:  # noqa: BLE001
        return float((((cfg or {}).get("editor") or {}).get("explore") or {}).get("timeout_s") or 300)


# The tensor_compute child program (2026-11-21): reads tc_spec.json from its workdir, loads the tensors (shard files
# via safetensors, in.safetensors for tensors an edit already changed, or seeded synthetic tensors in mock mode),
# runs the editor's code with T / torch / math, writes vectors / lora factors / writes to out.safetensors and returns
# a manifest {result: JSON text | None, vectors, lora, writes}.
TC_WRAPPER = r'''
import json
import math


def _js(x, depth=0):
    import torch
    if depth > 8:
        return str(x)[:200]
    if isinstance(x, torch.Tensor):
        if x.numel() > 10000:
            raise ValueError(f"result holds a tensor with {x.numel()} elements (max 10000): return statistics, or "
                             "use vectors / lora")
        return _js(x.detach().cpu().tolist(), depth + 1)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, dict):
        return {str(k): _js(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_js(v, depth + 1) for v in x]
    if hasattr(x, "item"):
        try:
            return _js(x.item(), depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return str(x)[:200]


def main(api):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    spec = json.load(open("tc_spec.json"))
    torch.set_num_threads(int(spec["threads"]))
    T = {}
    if spec.get("synth"):
        ns = {}
        exec(spec["synth_src"], ns)
        for n, sp in spec["synth"].items():
            T[n] = ns["synth_tensor"](torch, sp)
    for n, (path, key) in (spec.get("sources") or {}).items():
        with safe_open(path, "pt") as f:
            T[n] = f.get_tensor(key).to(torch.float32)
    if spec.get("inputs"):
        with safe_open("in.safetensors", "pt") as f:
            for n in spec["inputs"]:
                T[n] = f.get_tensor(n).to(torch.float32)
    T = {n: T[n] for n in spec["names"]}
    T.update({a: T[c] for a, c in (spec.get("aliases") or {}).items()})
    V = {}
    if spec.get("vin"):
        with safe_open("in.safetensors", "pt") as f:
            V = {r: f.get_tensor("V|" + r).to(torch.float32) for r in spec["vin"]}
    g = {"__name__": "__tensor_compute__", "T": T, "V": V, "torch": torch, "math": math, "result": None,
         "vectors": None, "lora": None, "writes": None}
    exec(compile(spec["code"], "<tensor_compute>", "exec"), g)

    def ten(x, what):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not (x.is_floating_point() or x.dtype in (torch.int32, torch.int64)):
            raise TypeError(f"{what}: not a numeric tensor ({x.dtype})")
        return x.detach().to("cpu", torch.float32).contiguous().clone()

    def dct(x, what):
        if x is None:
            return {}
        if not isinstance(x, dict):
            raise TypeError(f"{what} must be a dict")
        return x
    man, tens = {"vectors": [], "lora": [], "writes": []}, {}
    for k, v in dct(g.get("vectors"), "vectors").items():
        tens["v|" + str(k)] = ten(v, f"vectors[{k!r}]")
        man["vectors"].append(str(k))
    for m, ab in dct(g.get("lora"), "lora").items():
        if not isinstance(ab, (tuple, list)) or len(ab) != 2:
            raise TypeError(f"lora[{m!r}] must be a pair (A [r, in], B [out, r])")
        tens["A|" + str(m)], tens["B|" + str(m)] = ten(ab[0], f"lora[{m!r}] A"), ten(ab[1], f"lora[{m!r}] B")
        man["lora"].append(str(m))
    for n, t in dct(g.get("writes"), "writes").items():
        tens["w|" + str(n)] = ten(t, f"writes[{n!r}]")
        man["writes"].append(str(n))
    if tens:
        save_file(tens, "out.safetensors")
    r = g.get("result")
    man["result"] = None if r is None else json.dumps(_js(r), allow_nan=False)
    return man
'''


def _ops_per_module(ops) -> dict:
    """mock: number of recorded ops per module (op[1] = module list)."""
    out = {}
    for op in ops or []:
        if len(op) > 1 and isinstance(op[1], list):
            for m in op[1]:
                if isinstance(m, str):
                    out[m] = out.get(m, 0) + 1
    return out


def _trunc(obj, n: int = 12, depth: int = 0):
    """Shorten a JSON-like summary (dict items / list entries) for tool results."""
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        items = list(obj.items())
        out = {str(k): _trunc(v, n, depth + 1) for k, v in items[:n]}
        if len(items) > n:
            out["..."] = f"{len(items) - n} more"
        return out
    if isinstance(obj, (list, tuple)):
        out = [_trunc(v, n, depth + 1) for v in list(obj)[:6]]
        return out + [f"... {len(obj) - 6} more"] if len(obj) > 6 else out
    if isinstance(obj, float):
        return round(obj, 6)
    return obj if isinstance(obj, (int, str, bool, type(None))) else str(obj)[:200]


def _overlap(lo, others) -> float | None:
    """Mean normalised principal-angle overlap (cos² averaged) between ΔW column spaces and the stack."""
    import torch
    vals = []
    for m, (A, B) in lo.factors.items():
        for o in others:
            if m not in o.factors:
                continue
            q1, _ = torch.linalg.qr(B)
            q2, _ = torch.linalg.qr(o.factors[m][1])
            s = torch.linalg.svdvals(q1.T @ q2)
            vals.append(float((s ** 2).mean()))
    return round(sum(vals) / len(vals), 4) if vals else None


NOTE_CHARS = 600


def vllm_answer_nll(endpoint: str, model: str, encoded, *, batch: int = 16, timeout: float = 1800,
                    deadline: float | None = None, post=None) -> tuple[float, float]:
    """(mean answer-token NLL, seconds spent in requests) (2026-11-22, tools.train_loss_backend vllm) of encoded =
    [(input_ids, labels)] (HFModel.encode: labels -100 on the prompt) from vLLM's /v1/completions prompt_logprobs:
    position j of a token-id prompt holds log p(ids[j] | ids[:j]) (the prompt token itself is always included), so
    the answer NLL is the mean of -logprob over the positions with labels[j] != -100, over all items' answer tokens.
    deadline (time.time() value): no request runs past it (timeouts shortened, no retry after it)."""
    from rsicore import _net
    post = post or _net.post_json
    url = endpoint.rstrip("/") + "/completions"
    tot, n, secs = 0.0, 0, 0.0
    for i in range(0, len(encoded), max(1, int(batch))):
        chunk = encoded[i:i + max(1, int(batch))]
        payload = {"model": model, "prompt": [[int(t) for t in ids] for ids, _ in chunk], "max_tokens": 1,
                   "temperature": 0.0, "prompt_logprobs": 1}
        last = None
        for attempt in range(4):
            left = timeout if deadline is None else min(timeout, deadline - time.time())
            if left <= 0:
                raise RuntimeError("train_loss stopped: its GPU budget or wall-clock limit ran out"
                                   + (f" (last error: {last})" if last else ""))
            t0 = time.time()
            try:
                resp = post(url, payload, timeout=left)
                secs += time.time() - t0
                break
            except (_net.HTTPError, OSError) as e:
                secs += time.time() - t0
                last = e
                if isinstance(e, _net.HTTPError) and e.status < 500:
                    raise
                if deadline is not None and time.time() + 2 ** attempt >= deadline:
                    raise RuntimeError(f"train_loss stopped at its GPU budget / wall-clock limit: {e}") from None
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"train_loss prompt_logprobs request failed: {last}")
        chs = sorted(resp.get("choices") or [], key=lambda c: c.get("index", 0))
        if len(chs) != len(chunk):
            raise RuntimeError(f"train_loss: {len(chs)} choices for {len(chunk)} prompts")
        for (ids, lab), ch in zip(chunk, chs):
            plp = ch.get("prompt_logprobs")
            if plp is None:
                raise RuntimeError("server returned no prompt_logprobs (vLLM >= 0.6 needed)")
            for j in range(1, min(len(ids), len(lab))):
                if lab[j] == -100:
                    continue
                pos = plp[j] if j < len(plp) else None
                v = (pos or {}).get(str(int(ids[j])))
                if v is None:
                    raise RuntimeError(f"train_loss: no logprob for token {ids[j]} at position {j}")
                tot -= float(v["logprob"] if isinstance(v, dict) else v)
                n += 1
    return tot / max(n, 1), secs


def cut_note(note: str, n: int = NOTE_CHARS) -> str:
    """A program's returned "note" (2026-11-22), whitespace collapsed, at most n characters."""
    t = " ".join(str(note).split())
    return t if len(t) <= n else t[:n - 3].rstrip() + "..."


EDIT_GROUPS = 8


def _layer_ranges(xs) -> str:
    """[1, 2, 3, 5] -> '1-3,5'."""
    xs = sorted(set(xs))
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def _span(vals, fmt=lambda v: f"{v:.3g}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}-{fmt(hi)}"


def _op_sig(op: dict) -> str:
    """A weight-edit op's settings without tensors: 'scale s=1.05', 'rows n=3 f=1.1', 'lowrank r=4', 'delta'."""
    k = op.get("op")
    if k == "scale":
        return f"scale s={op.get('s'):.4g}"
    if k == "toward_base":
        return f"toward_base alpha={op.get('alpha'):.4g}"
    if k in ("rows", "cols"):
        f = op.get("f")
        idx = op.get("idx")
        return (f"{k} n={'all' if idx is None else len(idx)} "
                f"f={f'{f:.4g}' if isinstance(f, (int, float)) else 'vector'}")
    if k == "lowrank":
        return f"lowrank r={op.get('rank')}"
    return str(k)


def edit_summary(lo, rep: dict) -> dict:
    """What a candidate changed (2026-11-22), for the history table: the module types it touched and, per type (and
    per op setting for a weight edit), the layers and the parameters (LoRA rank, ||dW||/||W0||, weight-edit ops).
    rep = the sandbox check report (its 'modules' has rank / rel_norm per module)."""
    from .primitives.naming import tensor_layer
    mods = rep.get("modules") or {}
    ops = {}
    if getattr(lo, "kind", "lora") == "weights" and callable(getattr(lo, "summary", None)):
        try:
            s = lo.summary()
            ops = s if isinstance(s, dict) else {}
        except Exception:                    # noqa: BLE001 - the summary is for the history only
            ops = {}
    groups = {}
    for m, d in mods.items():
        layer = tensor_layer(m)
        typ = m.split(".", 2)[2] if layer is not None else m
        lst = ops.get(m)
        sig = "; ".join(_op_sig(op) for op in lst if isinstance(op, dict)) if isinstance(lst, list) else None
        g = groups.setdefault((typ, sig), {"layers": [], "rank": [], "rel": []})
        if layer is not None:
            g["layers"].append(layer)
        g["rank"].append(d.get("rank"))
        g["rel"].append(d.get("rel_norm"))
    out = []
    for (typ, sig), g in sorted(groups.items(), key=lambda kv: (-len(kv[1]["rel"]), kv[0][0], kv[0][1] or "")):
        row = {"type": typ, "n": len(g["rel"])}
        if g["layers"]:
            row["layers"] = _layer_ranges(g["layers"])
        r = _span(g["rank"], fmt=lambda v: str(int(v)))
        if r and r != "0":
            row["rank"] = r
        rel = _span(g["rel"])
        if rel:
            row["rel_norm"] = rel
        if sig:
            row["ops"] = sig[:160]
        out.append(row)
    res = {"n": len(mods), "types": sorted({t for t, _ in groups}), "groups": out[:EDIT_GROUPS]}
    if len(out) > EDIT_GROUPS:
        res["more_groups"] = len(out) - EDIT_GROUPS
    return res


def program_build(ws: Workspace, program: str, cid: str, *, timeout_s: int, mem_gb: int, workdir: Path):
    """build_fn for rsicore.search.inner_search: run the program in the sandbox, resolve and check the LoRA,
    save it. Returns (artifact | None, error | None)."""
    from rsicore.sandbox import run_program

    if not program or "def main" not in program:
        return None, "no program (the reply must contain a ```python block defining main(api))"
    ws.begin_program(cid, timeout_s=timeout_s)
    t0 = time.time()
    try:
        res = run_program(program, ws.api, timeout_s=timeout_s, mem_gb=mem_gb, workdir=Path(workdir))
        if not res.ok:
            return None, f"program failed: {res.error}\n{(res.stderr or '')[-1500:]}"
        with ws._lock:              # builds may run concurrently (2026-11-21): the checks read the weight store
            try:
                lo = ws.resolve_result(res.result)
            except (KeyError, TypeError, ValueError) as e:
                return None, f"bad return value: {e}"
            ok, rep = ws.check(lo)
            if not ok:
                return None, "sandbox checks failed: " + "; ".join(rep["errors"][:8])
            path = ws.save_candidate(cid, lo)
        tools = ws.tool_log.get(cid, [])
        art = {"cid": cid, "path": str(path), "keep": not lo.modules(), "modules": len(lo.modules()),
               "max_rank": lo.max_rank(), "digest": lo.digest(), "kind": getattr(lo, "kind", "lora"),
               "tools_used": _count(tools),
               "strategy": classify(tools), "stdout": (res.stdout or "")[-1000:], "seconds": round(time.time() - t0, 1)}
        try:                        # (2026-11-22) modules, module types, parameters, for the history
            art["edit"] = edit_summary(lo, rep)
        except Exception as e:      # noqa: BLE001 - never fail a built candidate over its description
            art["edit"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        note = res.result.get("note") if isinstance(res.result, dict) else None
        if isinstance(note, str) and note.strip():   # (2026-11-22) the candidate's own account, for the history
            art["note"] = cut_note(note)
        return art, None
    finally:
        ws.end_program(cid)


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def classify(tools) -> str:
    """Primitive-usage category for behaviour analysis (RSI_LOOP_DESIGN §3.6). mode=weights (any w_* op used):
    "weights_only", "weights+construct", "weights+ops", "weights+construct+ops". tensor_compute (2026-11-21) logs
    "tensor_compute.lora" when it built a LoRA (a construction) and "tensor_compute.writes" when it wrote weights
    (a weights op). Paper tools (2026-11-21): project_out on a LoRA is a construction, on a weight edit
    ("project_out.edit") a weights op; load_edit adds the tools that built the saved object."""
    s = set(tools)
    construct = s & {"rank1", "from_direction", "scale_singular", "tensor_compute.lora", "project_out"}
    if "project_out.edit" in s:                 # project_out on a weight edit is a weights op, not a construction
        construct.discard("project_out")
    stack_ops = s & {"interpolate", "combine", "truncate_rank", "intruder_scale", "relchange_prune",
                     "orthogonalize", "ties", "scale", "drop"}
    if s & {"w_scale", "w_scale_rows", "w_scale_cols", "w_add_lora", "w_toward_original", "tensor_compute.writes",
            "project_out.edit"}:
        return "weights" + ("+construct" if construct else "") + ("+ops" if stack_ops else "") \
            + ("" if construct or stack_ops else "_only")
    if "raw_lora" in s:   # mode=edit: what was done to the raw LoRA
        if construct and stack_ops:
            return "raw+construct+ops"
        if construct:
            return "raw+construct"
        return "raw+ops" if stack_ops else "raw_as_is"
    if not (construct or stack_ops):
        return "keep"
    if construct:
        return "pure_construct" if not stack_ops else "construct+ops"
    return "ops_only"


# ============================================================================ editor wiring (2026-11-21)
EXPLORE_MOCK_CODE = "print(api.budget())"   # the mock editor's run_python call (editor.mock_tool_rounds times)


def explore_doc(ws: Workspace, gpu_tools: bool = False, gpu_seconds: float = 300.0) -> str:
    """Description of the run_python chat tool for Paper B: the rsicore text plus what is missing in exploration."""
    from rsicore.chat_tools import RUN_PYTHON_DOC
    have = set(tools_for_mode(ws.mode, cfg=ws.cfg))
    missing = [n for n in Workspace.EXPLORE_NEVER + (() if gpu_tools else Workspace.GPU_TOOLS) if n in have]
    if not ws.tc.get("explore", True):         # tensor_compute.explore false (2026-11-21)
        missing += [n for n in ("tensors", "tensor_compute") if n in have]
    out = RUN_PYTHON_DOC
    if missing:
        out += " Not available here: " + ", ".join(f"api.{n}" for n in missing) + "."
    if "raw_lora" in missing:
        out += " The raw LoRA is still usable as the handle 'RAW'."
    if gpu_tools:
        out += (f" The GPU tools here use a separate exploration budget ({gpu_seconds:.0f} GPU-seconds for this "
                "session), not your program budget.")
    if explore_persistent(ws.cfg):            # one Python session per proposal (2026-11-21)
        out += (" Variables, imports and handles persist across your run_python calls for this candidate (one Python "
                "session; a timeout or crash restarts it and loses them).")
        if "save_edit" in have:
            return out + (" Your final program runs separately: build what it returns there, or save an object here "
                          "with api.save_edit(handle, name) and return {'lora': api.load_edit(name)['lora']} (or 'edit').")
        return out + " Your final program runs separately and must build what it returns."
    if "save_edit" in have:
        return out + (" Handles made here are freed after the call; save an object with api.save_edit(handle, name) "
                      "to load it later (api.load_edit) here or in your final program.")
    return out + " Handles made here are freed after the call; the final program must build what it returns."


def search_history(past: dict | None = None):
    """rsicore.chat_tools.SearchHistory for Paper B (2026-11-21): past = {gen: gen_dir} of earlier generations
    (none for the E task); rows summarized by prompts.history_row, same_as by the LoRA / edit digest, one line per
    earlier generation from its accepted.json."""
    from rsicore.chat_tools import SearchHistory
    from rsicore.records import read_json

    from .prompts import history_row

    def gen_fn(g, d):
        acc = read_json(Path(d) / "accepted.json", {}) or {}
        out = {"accepted": acc.get("cid") or acc.get("source")}
        if acc.get("source") == "keep" and acc.get("reason"):
            out["reason"] = str(acc["reason"])[:200]
        rs = (read_json(Path(d) / "reference_screen.json", {}) or {}).get("summary") or {}
        if rs.get("n_items"):        # the then-current model's own screen answers (2026-11-21)
            out["reference_screen"] = {k: rs.get(k) for k in ("n_items", "samples_per_item", "accuracy",
                                                              "correct_counts", "truncation_rate")}
        return out
    return SearchHistory(past or {}, row_fn=history_row, gen_fn=gen_fn,
                         key_fn=lambda r: (r.get("artifact") or {}).get("digest"))


def editor_chat_tools(ws: Workspace, cfg: dict, past: dict | None = None):
    """rsicore.chat_tools.ChatTools whose run_python runs on ws.explore_api (editor.explore.gpu_tools /
    gpu_seconds, timeout_s / mem_gb, editor.tool_result_chars), read_record and (editor.history_tool, default on)
    search_history over this search's candidates and past = {gen: gen_dir} of earlier generations; None when
    editor.tool_rounds is 0."""
    from rsicore.chat_tools import ChatTools, chat_tools_kwargs
    from rsicore.editor import editor_settings
    s = editor_settings(cfg)
    if int(s.get("tool_rounds") or 0) <= 0:
        return None
    history = search_history(past) if s.get("history_tool", True) else None
    gpu, secs = bool(s["explore"].get("gpu_tools")), float(s["explore"].get("gpu_seconds") or 0)

    persistent = explore_persistent(cfg)

    def factory(tag, ctx):
        # persistent (2026-11-21): one api per proposal -> (api, cleanup, before_call); else one per run_python call
        return ws.explore_api(tag, gpu_tools=gpu, gpu_seconds=secs, persistent=persistent)
    return ChatTools(factory, run_python_doc=explore_doc(ws, gpu, secs), history=history, **chat_tools_kwargs(cfg))


def explore_persistent(cfg: dict) -> bool:
    """editor.explore.persistent (2026-11-21; default false): one Python session per proposal (rsicore.chat_tools)."""
    return bool((((cfg or {}).get("editor") or {}).get("explore") or {}).get("persistent", False))


def editor_mock_policy(policy, cfg: dict):
    """Mock editor policy with editor.mock_tool_rounds run_python calls (EXPLORE_MOCK_CODE) before the program,
    when the tool loop is on (editor.tool_rounds > 0); the program itself is the one of a run without tools."""
    from rsicore.chat_tools import with_mock_exploration
    from rsicore.editor import editor_settings
    s = editor_settings(cfg)
    if policy is None or int(s.get("tool_rounds") or 0) <= 0:
        return policy
    return with_mock_exploration(policy, int(s.get("mock_tool_rounds") or 0), EXPLORE_MOCK_CODE)
