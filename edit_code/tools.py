"""ToolAPI the editor's program can call (registered on rsicore.sandbox.ToolAPI).

    model_info()          architecture summary, budget, action space, current model's config, tools
    profile_table()       rows {key, layer, group, bits, group_size, quantizer, dkl_mean, dkl_p99, top1_drop, params}
                          (whole-model KL when ONLY that group is RTN-quantized; forward pass on 64 texts)
    hessian_table()       rows {key, layer, group, quantizer, bits, group_size, herr_per_token, herr_rel, params}
                          (Hessian-weighted error of that group alone, rtn / gptq / awq; every calibration token)
    activation_table()    rows {layer, group, module, input, rms_mean, rms_max_over_median, absmax_over_median,
                                outlier_frac, red_flag}
    group_error(config)   per-group Hessian-weighted error of a config + the groups with the largest share
    compare(cid, other)   item-level comparison of two evaluated candidates (default other = the incumbent):
                          flips with ids, by output length, truncation; no item content
    bytes_table()         rows {key, layer, group, params, modules, mib: {"b4_g128": ..., "bf16": ...}}
    bytes(config)         packed size of a config {gib, total_bytes, linear_avg_bits, by_group_gib, fits_budget}
    evaluate(config)      build the candidate and run the screen evaluation -> L1 feedback   [cost: evaluate=1]
    solve_allocation(...) knapsack over the profile table -> config (removable: tools.solve_allocation=false)
Invalid configs raise a tool error whose message lists every problem (schema.ConfigError).
"""
from __future__ import annotations

from .allocation import solve_allocation as _solve
from .diagnostics import summarize_profile
from .problem import EvalBudgetExhausted, QuantProblem
from .quant import schema


class ToolError(RuntimeError):
    pass


def make_tool_api(problem: QuantProblem, cfg: dict, *, cid: str, per_program_evaluate: int):
    from rsicore.sandbox import ToolAPI
    tcfg = cfg.get("tools", {})
    api = ToolAPI({"evaluate": max(0, min(per_program_evaluate, problem.evals_left))})
    spec = problem.spec
    calls = {"evaluate": 0}

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
                "incumbent_scores": problem.incumbent_scores, "evaluate_left": problem.evals_left,
                "tools": api.names()}

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
            return problem.group_error(config, top=int(top))
        except schema.ConfigError as e:
            raise ToolError(str(e)) from None

    def compare(cid: str, other: str = "incumbent", stage: str = "screen"):
        """Compare two evaluated candidates item by item on the same items (default other = the incumbent;
        stage screen | full): per dataset accuracy of both, broken (other right, cid wrong) / fixed counts and
        ids, flips by the other's output length (short / medium / long), truncation rate, mean output length.
        Ids and statistics only (no item content). cid = a candidate id from the table, or '<cid>_e<k>' for
        api.evaluate calls."""
        if stage not in ("screen", "full"):
            raise ToolError("stage must be 'screen' or 'full'")
        try:
            return problem.compare(str(cid), str(other), stage)
        except ValueError as e:
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

    def evaluate(config):
        """Build this config as a candidate and run the screen evaluation (KL + screen items); returns
        the same structured feedback you get for submitted programs. Uses one evaluate."""
        calls["evaluate"] += 1
        try:
            r = problem.evaluate(config, "screen", f"{cid}_e{calls['evaluate']}", origin=f"tool:{cid}")
        except schema.ConfigError as e:
            raise ToolError(str(e)) from None
        except (ValueError, EvalBudgetExhausted) as e:
            raise ToolError(str(e)) from None
        return {"score": r["score"], "cached": bool(r.get("cached")), **r["feedback"]}

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
    if tcfg.get("compare", True):
        api.register("compare", compare)
    api.register("bytes_table", bytes_table)
    api.register("bytes", bytes)
    api.register("evaluate", evaluate, cost={"evaluate": 1})
    if tcfg.get("solve_allocation", True):
        api.register("solve_allocation", solve_allocation)
    return api
