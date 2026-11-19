"""Gradient tools and activation statistics with transformers + peft (gpu1 in the real run, CPU in tests).

TrainData holds the ONLY items gradient / activation tools may touch: this generation's `train` role of
the target domain. Every call that takes ids goes through TrainData.check(), which refuses any other id
(signal, screen, gate, kl or held-out items) with PermissionError.

HFModel lazily loads the current model (BF16, gradient checkpointing) and provides:
  loss(examples, lora)            mean token NLL of the target spans
  grad(examples, modules, lora)   ∂L/∂W for the given modules at W + ΔW, returned as top-k SVD
  sgd(examples, lora, ...)        short LoRA training with peft (AdamW), warm-started from `lora`
  acts(texts, modules, k)         input/output activation statistics (mean, norms, top principal directions)
  grams(texts, modules)           input Gram matrices Σ x xᵀ (RegMean)
  act_ratio(texts, lora)          ||ΔW h|| / ||W h|| per module
"""
from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager


class TrainData:
    def __init__(self, examples: dict[str, dict]):
        """examples: id -> {"messages": [...], "target": str, "source": "self"|"gold"}"""
        self.examples = dict(examples)
        self.allowed = set(self.examples)

    def ids(self) -> list[str]:
        return sorted(self.allowed)

    def check(self, ids) -> list[str]:
        ids = list(ids)
        bad = [i for i in ids if i not in self.allowed]
        if bad:
            raise PermissionError(f"{len(bad)} id(s) are not in this generation's train set (e.g. {bad[0]!r}); "
                                  "only train-role items may be used by gradient / activation tools")
        return ids

    def get(self, ids) -> list[dict]:
        return [self.examples[i] for i in self.check(ids)]

    def texts(self, ids) -> list[str]:
        return ["\n\n".join(m["content"] for m in e["messages"]) for e in self.get(ids)]


class HFModel:
    def __init__(self, model_dir, store, *, device: str = "cuda:0", dtype: str = "bfloat16", max_len: int = 4096,
                 grad_checkpointing: bool = True, seed: int = 42, devices: list[str] | None = None,
                 micro_batch: int | None = None):
        """devices: several GPUs -> the model is split across them layer by layer (device_map); inputs go to
        `device` (= the first one). One GPU (default): the whole model on `device`."""
        self.model_dir = str(model_dir)
        self.store = store
        self.device = device
        self.devices = [d for d in (devices or []) if d]
        # micro_batch: forward at most this many sequences at once (gradient accumulation in training);
        # 80 GB cards need 1 (fp32 logits over a ~248k vocabulary are ~4 GB per 4k-token sequence)
        self.micro_batch = int(micro_batch) if micro_batch else None
        self.dtype_name = dtype
        self.max_len = max_len
        self.grad_checkpointing = grad_checkpointing
        self.seed = seed
        self._model = None
        self._tok = None
        self.gpu_seconds = 0.0

    # ------------------------------------------------------------------ loading
    @property
    def model(self):
        if self._model is None:
            import torch
            import transformers

            cfg = transformers.AutoConfig.from_pretrained(self.model_dir)
            arch = (getattr(cfg, "architectures", None) or ["AutoModelForCausalLM"])[0]
            cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
            if len(self.devices) > 1 and all(d.startswith("cuda:") for d in self.devices):
                ids = [int(d.split(":")[1]) for d in self.devices]
                # first GPU also holds the inputs, logits (vocab ~248k) and activations: give it less weight
                per = {i: torch.cuda.get_device_properties(i).total_memory for i in ids}
                mm = {i: int(per[i] * (0.45 if k == 0 else 0.85)) for k, i in enumerate(ids)}
                self._model = cls.from_pretrained(self.model_dir, dtype=getattr(torch, self.dtype_name),
                                                  device_map="auto", max_memory=mm)
                self.device = f"cuda:{ids[0]}"
            else:
                self._model = cls.from_pretrained(self.model_dir, dtype=getattr(torch, self.dtype_name))
                self._model.to(self.device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            if self.grad_checkpointing and hasattr(self._model, "gradient_checkpointing_enable"):
                self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self._model.config, "use_cache"):
                self._model.config.use_cache = False
        return self._model

    @property
    def tokenizer(self):
        if self._tok is None:
            import transformers
            cfg = transformers.AutoConfig.from_pretrained(self.model_dir)
            vocab = int(getattr(getattr(cfg, "text_config", cfg), "vocab_size"))
            try:
                tok = transformers.AutoTokenizer.from_pretrained(self.model_dir)
                ids = tok("hello world", add_special_tokens=False)["input_ids"]
                if not ids or max(ids) >= vocab:
                    raise ValueError("unusable tokenizer")
                self._tok = tok
            except Exception:  # noqa: BLE001  (tests: tiny model without tokenizer files)
                self._tok = _ByteTokenizer(cfg)
        return self._tok

    def _enc(self, text: str) -> list[int]:
        tok = self.tokenizer
        if isinstance(tok, _ByteTokenizer):
            return tok.encode_text(text)
        return tok(text, add_special_tokens=False)["input_ids"]

    def release(self):
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def module(self, short: str):
        return self.model.get_submodule(self.store.full_name(short))

    # ------------------------------------------------------------------ tokenisation
    def encode(self, ex: dict):
        """(input_ids, labels) with the prompt masked (-100); Qwen chat template, thinking off
        (matches eval/configs/tasks.yaml `thinking: false`)."""
        tok = self.tokenizer
        msgs = ex["messages"]
        if getattr(tok, "chat_template", None):
            kw = {"enable_thinking": False}
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": ex["target"]}],
                                           tokenize=False, **kw)
            if not full.startswith(prompt):  # template rewrote the prompt part: fall back to concatenation
                full = prompt + ex["target"] + (tok.eos_token or "")
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
        else:
            p_ids = self._enc("".join(m["content"] + "\n" for m in msgs))
            f_ids = p_ids + self._enc(ex["target"])
        f_ids = f_ids[: self.max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        return f_ids, labels[: len(f_ids)]

    def _bs(self, batch_size: int) -> int:
        return max(1, min(int(batch_size), self.micro_batch)) if self.micro_batch else max(1, int(batch_size))

    def batches(self, examples, batch_size: int):
        import torch
        batch_size = self._bs(batch_size)
        enc = [self.encode(e) for e in examples]
        for i in range(0, len(enc), batch_size):
            chunk = enc[i:i + batch_size]
            L = max(len(x) for x, _ in chunk)
            pad = getattr(self.tokenizer, "pad_token_id", None) or 0
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            lab = torch.full((len(chunk), L), -100, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, (x, y) in enumerate(chunk):
                ids[j, :len(x)] = torch.tensor(x)
                lab[j, :len(y)] = torch.tensor(y)
                att[j, :len(x)] = 1
            yield ids.to(self.device), lab.to(self.device), att.to(self.device)

    def _loss(self, ids, lab, att):
        import torch.nn.functional as F
        out = self.model(input_ids=ids, attention_mask=att)
        logits = out.logits[:, :-1].float()
        tgt = lab[:, 1:].to(logits.device)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)

    # ------------------------------------------------------------------ LoRA application via hooks
    @contextmanager
    def applied(self, lora, trainable: bool = False):
        """Temporarily add x Aᵀ Bᵀ to the outputs of the LoRA's modules (forward hooks)."""
        handles = []
        if lora is not None:
            for m in lora.modules():
                A, B = lora.factors[m]
                mod = self.module(m)
                dt = mod.weight.dtype
                A_ = A.to(mod.weight.device, dt)      # the module's own GPU (model may be split)
                B_ = B.to(mod.weight.device, dt)

                def hook(_mod, inp, out, A_=A_, B_=B_):
                    return out + (inp[0] @ A_.T) @ B_.T

                handles.append(mod.register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ tools
    def _timed(self, fn):
        import torch
        t0 = time.time()
        try:
            return fn()
        finally:
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            self.gpu_seconds += time.time() - t0

    def loss(self, examples, lora=None, batch_size: int = 4) -> float:
        import torch

        def run():
            tot, n = 0.0, 0
            with torch.no_grad(), self.applied(lora):
                for ids, lab, att in self.batches(examples, batch_size):
                    tot += float(self._loss(ids, lab, att)) * ids.shape[0]
                    n += ids.shape[0]
            return tot / max(n, 1)
        return self._timed(run)

    def grad(self, examples, modules, lora=None, k: int = 8, batch_size: int = 4) -> dict:
        """Top-k SVD of the mean-loss gradient ∂L/∂W for each module, evaluated at W + ΔW(lora)."""
        import torch

        def run():
            mods = {m: self.module(m) for m in modules}
            for mod in mods.values():
                mod.weight.requires_grad_(True)
            acc = {m: None for m in modules}
            nb = 0
            try:
                with self.applied(lora):
                    for ids, lab, att in self.batches(examples, batch_size):
                        loss = self._loss(ids, lab, att)
                        loss.backward()
                        nb += 1
                        for m, mod in mods.items():
                            g = mod.weight.grad.float()
                            acc[m] = g.clone() if acc[m] is None else acc[m] + g
                            mod.weight.grad = None
            finally:
                for mod in mods.values():
                    mod.weight.requires_grad_(False)
                    mod.weight.grad = None
            out = {}
            for m, G in acc.items():
                G = G / max(nb, 1)
                if min(G.shape) <= 2048:
                    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
                    U, S, V = U[:, :k], S[:k], Vh[:k].T
                else:
                    torch.manual_seed(self.seed)
                    U, S, V = torch.svd_lowrank(G, q=k + 8, niter=4)
                    U, S, V = U[:, :k], S[:k], V[:, :k]
                out[m] = {"U": U.cpu(), "S": S.cpu(), "V": V.cpu(), "fro": float(G.norm())}
            return out
        return self._timed(run)

    def sgd(self, examples, lora=None, *, modules=None, rank: int = 16, steps: int = 50, lr: float = 1e-4,
            batch_size: int = 4, weight_decay: float = 0.0, warmup: int = 0, time_budget_s: float | None = None,
            checkpoints: list[float] | None = None, init_seed: int | None = None):
        """Short LoRA training with peft on `examples` (cycled), warm-started from `lora`.

        Trained modules = lora's modules ∪ `modules`; adapter rank = max(rank, lora ranks). Returns
        (final LoRA, info) where info has the loss curve, steps done, seconds and optional checkpoints
        (list of (fraction, LoRA)). Stops early when time_budget_s is exhausted."""
        import torch
        from peft import LoraConfig, get_peft_model

        from .lora import LoRA

        def run():
            seed = self.seed if init_seed is None else init_seed
            torch.manual_seed(seed)
            rng = random.Random(seed)
            mods = sorted(set(modules or []) | set(lora.factors if lora is not None else []))
            if not mods:
                raise ValueError("sgd needs `modules` or a non-empty `lora`")
            r = max([int(rank)] + ([lora.rank(m) for m in lora.factors] if lora is not None else []))
            full = {m: self.store.full_name(m) for m in mods}
            cfg = LoraConfig(r=r, lora_alpha=r, lora_dropout=0.0, target_modules=sorted(full.values()),
                             bias="none", init_lora_weights=True)
            pm = get_peft_model(self.model, cfg)
            try:
                named = dict(pm.named_modules())
                layers = {}
                for m, fn in full.items():
                    lm = named.get("base_model.model." + fn)
                    if lm is None:
                        raise KeyError(f"peft did not wrap {fn}")
                    la, lb = lm.lora_A["default"].weight, lm.lora_B["default"].weight
                    with torch.no_grad():
                        la.data = la.data.float()
                        lb.data = lb.data.float()
                        if lora is not None and m in lora.factors:
                            A, B = lora.factors[m]
                            k = A.shape[0]
                            la[:k] = A.to(la.device)
                            lb.zero_()
                            lb[:, :k] = B.to(lb.device)
                    layers[m] = (la, lb)
                params = [p for pair in layers.values() for p in pair]
                for p in params:
                    p.requires_grad_(True)
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
                order = list(range(len(examples)))
                rng.shuffle(order)
                losses, ckpts = [], []
                marks = sorted({max(1, int(round(f * steps))) for f in (checkpoints or [])})
                t0 = time.time()
                pm.train()
                done = 0
                for step in range(int(steps)):
                    sel = [examples[order[(step * batch_size + j) % len(order)]] for j in range(batch_size)]
                    cur = lr * min(1.0, (step + 1) / warmup) if warmup else lr
                    for g in opt.param_groups:
                        g["lr"] = cur
                    opt.zero_grad(set_to_none=True)
                    # micro-batches (gradient accumulation), weighted by label tokens = one batch's mean loss
                    chunks = list(self.batches(sel, batch_size))
                    ntok = [int((lab[:, 1:] != -100).sum()) for _, lab, _ in chunks]
                    tot_tok, loss_val = max(1, sum(ntok)), 0.0
                    for (ids, lab, att), n in zip(chunks, ntok):
                        if n == 0:
                            continue
                        loss = self._loss(ids, lab, att) * (n / tot_tok)
                        loss.backward()
                        loss_val += float(loss)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    losses.append(loss_val)
                    done = step + 1
                    if done in marks:
                        ckpts.append((done / steps, _extract(layers)))
                    if time_budget_s is not None and time.time() - t0 > time_budget_s:
                        break
                final = _extract(layers)
                if marks and (not ckpts or ckpts[-1][0] < done / steps):
                    ckpts.append((done / steps, final.copy()))
                return final, {"losses": losses, "steps_done": done, "rank": r, "modules": len(mods),
                               "seconds": time.time() - t0, "checkpoints": ckpts}
            finally:
                pm.eval()
                base = pm.unload()  # remove LoRA layers, restore plain modules
                self._model = base
                for p in self._model.parameters():
                    p.requires_grad_(False)

        def _extract(layers):
            out = LoRA()
            for m, (la, lb) in layers.items():
                out.factors[m] = (la.detach().float().cpu().clone(), lb.detach().float().cpu().clone())
            return out

        return self._timed(run)

    def acts(self, texts, modules, k: int = 4, batch_size: int = 4) -> dict:
        """Per module: mean input / output vectors, mean norms and the top-k principal directions of the
        input second moment (uncentred), from forward passes over `texts` (all tokens)."""
        import torch

        def run():
            stats = {m: {"n": 0, "sx": None, "sy": None, "nx": 0.0, "ny": 0.0, "G": None} for m in modules}
            hs = []
            for m in modules:
                mod = self.module(m)

                def hook(_mod, inp, out, m=m):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    y = out.detach().float().reshape(-1, out.shape[-1])
                    st = stats[m]
                    st["n"] += x.shape[0]
                    st["sx"] = x.sum(0) if st["sx"] is None else st["sx"] + x.sum(0)
                    st["sy"] = y.sum(0) if st["sy"] is None else st["sy"] + y.sum(0)
                    st["nx"] += float(x.norm(dim=1).sum())
                    st["ny"] += float(y.norm(dim=1).sum())
                    g = x.T @ x
                    st["G"] = g if st["G"] is None else st["G"] + g

                hs.append(mod.register_forward_hook(hook))
            try:
                with torch.no_grad():
                    for ids, att in self._text_batches(texts, batch_size):
                        self.model(input_ids=ids, attention_mask=att)
            finally:
                for h in hs:
                    h.remove()
            out = {}
            for m, st in stats.items():
                n = max(st["n"], 1)
                G = st["G"] / n
                ev, evec = torch.linalg.eigh(G)
                top = torch.flip(evec[:, -k:], dims=[1]).T.cpu()
                out[m] = {"in_mean": (st["sx"] / n).cpu(), "out_mean": (st["sy"] / n).cpu(),
                          "in_norm": st["nx"] / n, "out_norm": st["ny"] / n,
                          "pcs": top, "pc_var": torch.flip(ev[-k:], dims=[0]).cpu().tolist(), "tokens": n}
            return out
        return self._timed(run)

    def grams(self, texts, modules, batch_size: int = 4) -> dict:
        import torch

        def run():
            G = {m: None for m in modules}
            hs = []
            for m in modules:
                def hook(_mod, inp, out, m=m):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    g = x.T @ x
                    G[m] = g if G[m] is None else G[m] + g
                hs.append(self.module(m).register_forward_hook(hook))
            try:
                with torch.no_grad():
                    for ids, att in self._text_batches(texts, batch_size):
                        self.model(input_ids=ids, attention_mask=att)
            finally:
                for h in hs:
                    h.remove()
            return {m: g.cpu() for m, g in G.items() if g is not None}
        return self._timed(run)

    def act_ratio(self, texts, lora, batch_size: int = 4) -> dict:
        """Mean over tokens of ||ΔW h|| / ||W h|| for every module of `lora` (TaDA activation ratio)."""
        import torch

        def run():
            acc = {m: [0.0, 0] for m in lora.modules()}
            hs = []
            for m in lora.modules():
                A, B = lora.factors[m]
                dev = self.module(m).weight.device
                A_, B_ = A.to(dev).float(), B.to(dev).float()

                def hook(_mod, inp, out, m=m, A_=A_, B_=B_):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    y = out.detach().float().reshape(-1, out.shape[-1])
                    d = (x @ A_.T) @ B_.T
                    r = d.norm(dim=1) / y.norm(dim=1).clamp_min(1e-6)
                    acc[m][0] += float(r.sum())
                    acc[m][1] += r.numel()
                hs.append(self.module(m).register_forward_hook(hook))
            try:
                with torch.no_grad():
                    for ids, att in self._text_batches(texts, batch_size):
                        self.model(input_ids=ids, attention_mask=att)
            finally:
                for h in hs:
                    h.remove()
            return {m: s / max(n, 1) for m, (s, n) in acc.items()}
        return self._timed(run)

    def _text_batches(self, texts, batch_size):
        import torch
        batch_size = self._bs(batch_size)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = [self._enc(t)[: min(self.max_len, 1024)] or [0] for t in chunk]
            L = max(len(e) for e in enc)
            ids = torch.zeros((len(enc), L), dtype=torch.long)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e)
                att[j, :len(e)] = 1
            yield ids.to(self.device), att.to(self.device)


class _ByteTokenizer:
    """Fallback for tiny test models without tokenizer files: bytes mod vocab."""
    chat_template = None
    eos_token = None
    pad_token_id = 0

    def __init__(self, config):
        tc = getattr(config, "text_config", config)
        self.vocab = int(tc.vocab_size)

    def encode_text(self, s: str) -> list[int]:
        return [1 + (b % (self.vocab - 1)) for b in s.encode()]

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": self.encode_text(s)}


def estimate_seconds(n_steps: int, batch_size: int, sec_per_example: float) -> float:
    return math.ceil(n_steps * batch_size * sec_per_example)
