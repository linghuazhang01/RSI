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
    screen_results(ids, filter, limit, offset)  the incumbent's (latest model's) answers to this round's screen items:
                          prompt, gold, per-sample scores / extracted answers / finish / lengths (rsicore.screen_view)
    screen_output(id, sample, offset)            one sample's full output text (thinking + answer) in pages
Invalid configs raise a tool error whose message lists every problem (schema.ConfigError).
No tool evaluates a config (2026-11-23, the hard-wired flow): candidates are never screened, every returned config gets
ONE gate evaluation outside the program (run.gate_candidate); there is no api.evaluate, no evaluate budget and no
compare (candidates have no screen records).

explore=True (2026-11-21): the API of the editor's run_python chat tool (rsicore.chat_tools), used while it
writes a candidate: the same tools, except that group_error refuses settings whose Hessian-weighted error is not
cached yet (lazy GPTQ / AWQ on the GPU) unless editor.explore.gpu_tools is true.
"""
from __future__ import annotations

from .allocation import solve_allocation as _solve
from .diagnostics import summarize_profile
from .problem import QuantProblem
from .quant import schema


class ToolError(RuntimeError):
    pass


def make_tool_api(problem: QuantProblem, cfg: dict, *, cid: str, explore: bool = False):
    """The program's API (cid: the candidate id, or the exploration tag). No tool spends a budget. explore (the
    editor's run_python): group_error only on cached settings unless editor.explore.gpu_tools."""
    from rsicore.sandbox import ToolAPI
    tcfg = cfg.get("tools", {})
    gpu_tools = bool(((cfg.get("editor") or {}).get("explore") or {}).get("gpu_tools", False))
    api = ToolAPI({})
    out_chars = int(tcfg.get("screen_output_chars", 4000))
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
                "incumbent_config": (problem.incumbent or {}).get("config"),
                "incumbent_scores": problem.incumbent_scores, "tools": api.names()}

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
        """Per-group Hessian-weighted error of a config (no evaluation, no forward pass; free): {total_per_token,
        top: [(key, share)], groups: {key: {setting, herr_per_token, herr_rel, share}}}. Shows where a config's
        quantization error concentrates."""
        try:
            if explore and not gpu_tools:
                missing = problem.herr_missing(config)
                if missing:
                    raise ToolError(f"uncached setting: not available while exploring ({len(missing)} module "
                                    f"settings not computed yet, e.g. {missing[0]}); the hessian_table settings "
                                    "are cached, and your final program may call group_error on any config")
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

    def screen_results(ids: list = None, filter: str = "all", limit: int = 50, offset: int = 0, gen: int = None):
        """The incumbent's (latest model's) answers to this round's screen items (held-back TRAINING items, k sampled
        answers each, thinking mode): [{id, dataset, prompt (cut), gold, scores [k], answers (extracted final
        answers), finish (stop | length), lengths (chars incl. thinking), truncated_thinking}]. filter: all | wrong
        (no sample right) | unstable (some right) | right (every sample right); sorted by id; page with limit /
        offset. gen = an earlier round's screen (default: this round)."""
        from rsicore import screen_view
        recs = problem.screen_records(gen)
        if not recs:
            raise ToolError("no screen records of the incumbent for this round" if gen is None
                            else f"no screen records of the incumbent for round {gen}")
        try:
            return screen_view.rows(recs, problem.screen_items(recs), ids=ids, filter=str(filter),
                                    limit=min(int(limit), 200), offset=int(offset))
        except ValueError as e:
            raise ToolError(str(e)) from None

    def screen_output(id: str, sample: int = 0, offset: int = 0, gen: int = None):  # noqa: A002 - tool argument name
        """One sample's full output text (its thinking + the answer) of the incumbent on a screen item, in pages:
        {id, sample, n_samples, score, answer, finish, has_thinking, offset, total_chars, text, next_offset}."""
        from rsicore import screen_view
        for recs in problem.screen_records(gen).values():
            rec = next((r for r in recs if str(r.get("id")) == str(id)), None)
            if rec is not None:
                try:
                    return screen_view.output(rec, sample=int(sample), offset=int(offset), chars=out_chars)
                except ValueError as e:
                    raise ToolError(str(e)) from None
        raise ToolError(f"{id!r} is not a screen item of this round (see screen_results)")

    def solve_allocation(budget_gib: float = None, weights: dict = None, bits: list = None, group_sizes: list = None,
                         quantizer: str = "gptq", fixed_rules: list = None, fixed_groups: dict = None,
                         embed="bf16", lm_head=8):
        """Multiple-choice knapsack over the profile table: minimise sum(weight(key) * dkl(key, bits, gs))
        s.t. size <= budget_gib (default: this generation's budget). weights: {selector: multiplier}
        (selectors like '*.down', '60-63.gate_up', '7.qk', 'v'); fixed_rules / fixed_groups use the
        config's rules / groups syntax and are kept as given. Returns a config."""
        budget = float(budget_gib if budget_gib is not None else problem.budget_gib)
        fixed = {}
        if fixed_rules or fixed_groups:
            dflt = {"bits": 16, "group_size": 128, "quantizer": "rtn"}
            try:
                nf = schema.normalize({"default": dflt, "rules": fixed_rules or [], "groups": fixed_groups or {}},
                                      spec, problem.space)
                n0 = schema.normalize({"default": dflt}, spec, problem.space)
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
    api.register("screen_results", screen_results)
    api.register("screen_output", screen_output)
    api.register("bytes_table", bytes_table)
    api.register("bytes", bytes)
    if tcfg.get("solve_allocation", True):
        api.register("solve_allocation", solve_allocation)
    return api
