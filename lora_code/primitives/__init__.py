"""Weight-space primitives for Paper B (torch / safetensors / transformers / peft are imported lazily).

naming.py     Qwen3.5 module names, vLLM packed-module mapping, PEFT key format
weights.py    WeightStore: lazy safetensors access, top-k SVD (cached)
lora.py       LoRA (A, B factors), sandbox checks, PEFT save/load (vLLM-loadable), merge into weights
wedit.py      WeightEdit (dense ops: scale, rows, cols, lowrank, toward_base), check_wedit, edit_into_model
construct.py  rank1, from_direction, scale_singular
ops.py        scale, interpolate, combine, drop, truncate_rank, intruder_scale, relchange_prune,
              orthogonalize, ties
train.py      TrainData (train-id restriction), HFModel (loss, sgd with peft for the raw LoRA, acts, act_ratio)
"""
