"""ToolAPI the editor's program can call (registered on rsicore.sandbox.ToolAPI).

    model_info()          architecture summary, budget, action space, current model's config, tools
    profile_table()       rows {key, layer, group, bits, group_size, quantizer, dkl_mean, dkl_p99, top1_drop, params}
                          (whole-model KL when ONLY that group is RTN-quantized; forward pass on 64 texts)
    hessian_table()       rows {key, layer, group, quantizer, bits, group_size, herr_per_token, herr_rel, params}
                          (Hessian-weighted error of that group alone, rtn / gptq / awq; every calibration token)
    activation_table()    rows {layer, group, module, input, rms_mean, rms_max_over_median, absmax_over_median,
                                outlier_frac, red_flag}
    group_error(config)   per-group Hessian-weighted error of a config + the groups with the largest share
    bytes_table()         rows {key, layer, group, params, modules, mib: {"b4_g128": ..., "bf16": ...}}
    bytes(config)         packed size of a config {gib, total_bytes, linear_avg_bits, by_group_gib, fits_budget}
    solve_allocation(...) knapsack over the profile table -> config (removable: tools.solve_allocation=false)
Invalid configs raise a tool error whose message lists every problem (schema.ConfigError).
No tool evaluates a config: every returned config gets ONE KL evaluation on the fixed KL set outside the program
(run.kl_candidate, 2026-11-25).

explore=True (2026-11-21): the API of the editor's run_python chat tool (rsicore.chat_tools), used while it
writes a candidate: the same tools, except that group_error refuses settings whose Hessian-weighted error is not
cached yet (lazy GPTQ / AWQ on the GPU) unless editor.explore.gpu_tools is true.
"""
from __future__ import annotations

from .allocation import solve_allocation as _solve
from .diagnostics import summarize_profile
from .problem import QuantProblem
from .quant import schema
from .quant.spec import TIED_GROUPS

PACKED_SOLVER_DOC = (" Packed format: a tied pair (qk + v, gdn_qkv + gdn_z) gets one setting (fixing one group fixes "
                     "both), only packable settings are used.")


class ToolError(RuntimeError):
    pass


EXPLORE_DEADLINE_MARGIN_S = 30.0   # (2026-11-26) an exploration GPU tool stops this long before the call's time limit


def make_tool_api(problem: QuantProblem, cfg: dict, *, cid: str | None = None, explore: bool = False,
                  deadline: dict | None = None):
    """The program's API. No tool spends a budget. cid is ignored (accepted for older callers). explore (the
    editor's run_python): group_error only on cached settings unless editor.explore.gpu_tools. deadline (explore,
    2026-11-26): {"t": time.monotonic() deadline of the current run_python call} set before every call
    (run.Runner.chat_tools); an uncached group_error then stops in time with a ToolError instead of being killed."""
    from rsicore.sandbox import ToolAPI
    tcfg = cfg.get("tools", {})
    gpu_tools = bool(((cfg.get("editor") or {}).get("explore") or {}).get("gpu_tools", False))
    api = ToolAPI({})
    spec = problem.spec

    def model_info():
        """Architecture, budget, action space, the incumbent config (best so far) and tool list."""
        s = spec.summary()
        bf16 = problem.size({"default": {"bits": 16, "group_size": 128, "quantizer": "rtn"}})["gib"]
        return {"budget_gib": problem.budget_gib, "bf16_gib": round(bf16, 3), "num_layers": spec.num_layers,
                "full_attention_layers": [i for i, t in enumerate(spec.layer_types) if t == "full_attention"],
                "groups_full_attention_layer": ["qk", "v", "o", "gate_up", "down"],
                "groups_linear_attention_layer": ["gdn_qkv", "gdn_z", "gdn_ba", "gdn_out", "gate_up", "down"],
                "num_group_keys": len(spec.group_keys()), "params_by_group": s["params_by_group"],
                "embed_params": s["embed_params"], "lm_head_params": s["lm_head_params"],
                "lm_head_tied": s["lm_head_tied"],
                "allowed_bits": problem.space["bits"], "group_sizes": problem.space["group_sizes"],
                "quantizers": problem.space["quantizers"], "embed_options": problem.space["embed"],
                "lm_head_options": problem.space["lm_head"],
                "incumbent_config": (problem.incumbent or {}).get("config"),
                "incumbent_kl": problem.incumbent_kl, "tools": api.names()} | (
            {"packed_format": True,
             "tied_groups": [list(t) for t in TIED_GROUPS],
             "packed_rules": "qk and v (and gdn_qkv and gdn_z) of a layer need the same bits and group_size; a "
                             "quantized module needs rows*bits and columns*bits divisible by 32 and columns divisible "
                             "by group_size; quantizers rtn / gptq; embed and lm_head bf16"}
            if problem.space.get("packed") else {})

    def profile_table():
        """Whole-model DeltaKL when ONLY this module group is RTN-quantized (original model = reference)."""
        return problem.tables["profile"]

    def profile_summary():
        """Most / least sensitive groups per bit width, and per-group-type sums of the profile table."""
        return summarize_profile(problem.tables["profile"])

    def activation_table():
        """Input activation statistics per linear module (red_flag = heavy outliers)."""
        return problem.tables["activation"]

    def hessian_table():
        """Hessian-weighted output error tr(dW H dW^T) per token (herr_per_token) and relative to tr(W H W^T)
        (herr_rel) when ONLY this group is quantized with (quantizer, bits) at group size 128; rtn / gptq / awq.
        Uses every calibration token (no forward pass): a local, low-noise complement of profile_table."""
        return problem.tables.get("hessian") or []

    def group_error(config, top: int = 10):
        """Per-group Hessian-weighted error of a config (no evaluation, no forward pass; settings not cached yet
        are quantized on the GPU first, seconds per module): {total_per_token,
        top: [(key, share)], groups: {key: {setting, herr_per_token, herr_rel, share}}}. Shows where a config's
        quantization error concentrates."""
        try:
            if explore and not gpu_tools:
                missing = problem.herr_missing(config)
                if missing:
                    raise ToolError(f"uncached setting: not available while exploring ({len(missing)} module "
                                    f"settings not computed yet, e.g. {missing[0]}); the hessian_table settings "
                                    "are cached, and your final program may call group_error on any config")
            if explore and gpu_tools and hasattr(problem, "tool_device"):
                missing = problem.herr_missing(config)
                if missing:
                    # (2026-11-26) uncached settings while exploring: computed on a GPU of
                    # gpu_layout.explore_tool_gpus, stopping before the run_python call's time limit
                    from .problem import HerrDeadline
                    t = (deadline or {}).get("t")
                    dl = None if t is None else t - EXPLORE_DEADLINE_MARGIN_S
                    try:
                        with problem.tool_device(deadline=dl) as (cache, lock):
                            return problem.group_error(config, top=int(top), cache=cache, lock=lock, deadline=dl)
                    except HerrDeadline:
                        left = len(problem.herr_missing(config))
                        raise ToolError(f"ran out of time in this run_python call: {len(missing) - left} of "
                                        f"{len(missing)} uncached module settings were computed and cached, {left} "
                                        "remain; call group_error again to continue (cached settings return at "
                                        "once)") from None
            return problem.group_error(config, top=int(top))
        except schema.ConfigError as e:
            raise ToolError(str(e)) from None

    def bytes_table():
        """Packed MiB of every group key for each (bits, group size)."""
        return problem.tables["bytes"]

    def bytes(config):  # noqa: A001 - tool name fixed by the design
        """Packed size of a config (validates it)."""
        try:
            b = problem.size(config)
        except schema.ConfigError as e:
            raise ToolError(str(e)) from None
        return {"gib": round(b["gib"], 4), "total_bytes": b["total_bytes"], "linear_avg_bits": round(b["linear_avg_bits"], 3),
                "by_group_gib": {k: round(v, 4) for k, v in b["by_group_gib"].items()},
                "budget_gib": problem.budget_gib, "fits_budget": b["gib"] <= problem.budget_gib + 1e-9}

    def solve_allocation(budget_gib: float = None, weights: dict = None, bits: list = None, group_sizes: list = None,
                         quantizer: str = "gptq", fixed_rules: list = None, fixed_groups: dict = None,
                         embed="bf16", lm_head=None):
        """Multiple-choice knapsack over the profile table: minimise sum(weight(key) * dkl(key, bits, gs))
        s.t. size <= budget_gib (default: this generation's budget). weights: {selector: multiplier}
        (selectors like '*.down', '60-63.gate_up', '7.qk', 'v'); fixed_rules / fixed_groups use the
        config's rules / groups syntax and are kept as given. lm_head None = 8 if the action space allows it, else
        "bf16". Returns a config."""
        budget = float(budget_gib if budget_gib is not None else problem.budget_gib)
        fixed = {}
        if fixed_rules or fixed_groups:
            dflt = {"bits": 16, "group_size": 128, "quantizer": "rtn"}
            # the packed rules are checked on the solver's result (it gives a fixed group's tied partner the same
            # setting), not on this stand-in config whose unfixed groups are BF16 fill-ins
            sp_fixed = dict(problem.space, packed=False)
            try:
                nf = schema.normalize({"default": dflt, "rules": fixed_rules or [], "groups": fixed_groups or {}},
                                      spec, sp_fixed)
                n0 = schema.normalize({"default": dflt}, spec, sp_fixed)
            except schema.ConfigError as e:
                raise ToolError(str(e)) from None
            fixed = {k: s for k, s in nf["groups"].items() if s != n0["groups"][k]}
            fixed = {k: ({"bits": 16} if s["bits"] >= 16 else s) for k, s in fixed.items()}
        try:
            return _solve(spec, problem.tables["profile"], budget, weights=weights, bits=bits,
                          group_sizes=group_sizes, quantizer=quantizer, fixed=fixed, embed=embed, lm_head=lm_head,
                          gs_factor=cfg.get("solver", {}).get("group_size_factor"), action_space=problem.space)
        except (ValueError, schema.ConfigError) as e:
            raise ToolError(str(e)) from None

    api.register("model_info", model_info)
    if tcfg.get("profile_table", True):
        api.register("profile_table", profile_table)
        api.register("profile_summary", profile_summary)
    if tcfg.get("activation_table", True):
        api.register("activation_table", activation_table)
    if tcfg.get("hessian_table", True):
        api.register("hessian_table", hessian_table)
    if tcfg.get("group_error", True):
        api.register("group_error", group_error)
    api.register("bytes_table", bytes_table)
    api.register("bytes", bytes)
    if tcfg.get("solve_allocation", True):
        # the packed format adds its rules to the description (the dequant text stays as it was)
        api.register("solve_allocation", solve_allocation,
                     doc=(solve_allocation.__doc__.strip() + PACKED_SOLVER_DOC)
                     if (getattr(problem, "space", None) or {}).get("packed") else "")
    return api
