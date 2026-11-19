"""Quantization library for Paper A (torch + safetensors + transformers; no llm-compressor).

spec        module groups / shapes from safetensors headers (Qwen3.5 dense incl. GDN projections)
schema      editor config schema, validation, resolution
sizes       analytic packed size bytes(config)
quantizers  RTN / GPTQ / AWQ-lite, packing, dequantization        (torch)
hessian     calibration pass on the current model: Hessians + activation stats (transformers)
cache       per-generation layer quantization cache (packed codes on disk, lazy fill)
export      fake-quant BF16 checkpoint export (vLLM-loadable)
kl          exact token KL with a local transformers model
Import torch-dependent submodules explicitly (spec/schema/sizes work without torch).
"""
