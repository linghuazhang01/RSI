"""KL fitness for the EvoPress-style baseline (edit_code.baselines.evopress_search), one worker PROCESS per GPU
(2026-11-23; threads share one GIL, see the profile table).

Each worker loads M_0 (HF transformers) on its GPU, computes the reference log-probs of the calibration contexts
(the profile table's: diagnostics.profile_contexts texts cut at profile_max_len tokens, packed into batches of
profile_batch_tokens), quantizes embed / lm_head once (RTN, like the fake-quant export), then serves jobs: a config's
module settings are written into the model in place (only the modules whose setting changed since the previous job;
dequantized from the shared layer cache on the GPU, bits 16 = the original weight) and KL(M_0 || config) is measured
on the first quarter of the batches ("small") or all of them ("full"). The weights are the same tensors the fake-quant
export writes, so the fitness is the KL of the model the gate would score.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
import traceback
from pathlib import Path

STAGES = ("small", "full")


def _worker(model_dir: str, device: str, spec_path: str, prep_dir: str, texts: list[str], dcfg: dict, embed, lm_head,
            damp: float, jobs, results, tag: str = "") -> None:  # pragma: no cover - exercised on CPU by the tests
    import torch
    from .common import read_json
    from .quant import quantizers as Q
    from .quant.cache import QuantCache
    from .quant.hessian import load_hf_model, tokenize_texts
    from .quant.kl import kl_to_reference_packed, pack, reference_packed
    from .quant.sizes import EMBED_GROUP
    from .quant.spec import ModelSpec

    parent = os.getppid()

    def watchdog():                    # the run was killed: do not keep holding the GPU
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                os._exit(1)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        spec = ModelSpec.from_json(read_json(spec_path))
        model = load_hf_model(Path(model_dir), device)
        if texts and not isinstance(texts[0], str):        # already token ids (tests)
            batches = [torch.tensor([list(t)[: int(dcfg["profile_max_len"])]], dtype=torch.long) for t in texts]
        else:
            batches = tokenize_texts(Path(model_dir), texts, int(dcfg["profile_max_len"]))
        packed = pack(batches, int(dcfg.get("profile_batch_tokens") or 8192))
        refs = [r.cpu() for r in reference_packed(model, packed)]
        # chunks of each packed batch (reference_packed yields ceil((n - 1) / 512) chunks per context, in order)
        per_batch = [sum(math.ceil(max(n - 1, 0) / 512) for n in lens) for _, lens in packed]
        n_small = max(1, len(packed) // 4)
        cache = QuantCache(Path(prep_dir) / "cache", spec, Path(model_dir), hessian_dir=Path(prep_dir) / "hessians",
                           device=device, damp=damp, log=lambda *a: None)
        with torch.no_grad():         # embed / lm_head as the export writes them (RTN at EMBED_GROUP)
            for t, prec in ((spec.embed, embed), (spec.lm_head, lm_head)):
                if t is not None and prec != "bf16":
                    p = model.get_parameter(t["name"])
                    p.data.copy_(Q.rtn(p.data.float(), int(prec), EMBED_GROUP).dequant(p.dtype, device=p.device))
        current_s = {m.name: None for m in spec.modules}     # setting key now in the model (None = original)
        results.put(("ready", device, len(packed)))
        while True:
            job = jobs.get()
            if job is None:
                return
            jid, assign, stage = job
            with torch.no_grad():
                for name, s in assign.items():
                    key = (int(s["bits"]), s.get("quantizer"), int(s.get("group_size") or 0))
                    if current_s.get(name) == key or (key[0] >= 16 and current_s.get(name) is None):
                        continue
                    p = model.get_parameter(name)
                    if key[0] >= 16:
                        w = cache.weight(name)
                    else:
                        q = cache.get(name, key[1], key[0], key[2])
                        w = Q.dequant(q, dtype=p.dtype, device=p.device)
                    p.data.copy_(w.to(p.device, p.dtype))
                    current_s[name] = key
            nb = n_small if stage == "small" else len(packed)
            r = kl_to_reference_packed(model, packed[:nb], refs[:sum(per_batch[:nb])])
            results.put(("done", jid, float(r["mean"])))
    except BaseException:  # noqa: BLE001 - reported to the parent
        results.put(("error", device, traceback.format_exc()))


class KLPool:
    """fitness(norms, stage) over worker processes on `devices` (jobs pulled from one queue)."""

    def __init__(self, devices: list[str], *, model_dir, spec_path, prep_dir, texts, dcfg: dict, embed, lm_head,
                 damp: float = 0.01, log=print, timeout_s: float = 7200):
        import multiprocessing as mp
        from .quant.schema import module_assignment  # noqa: F401 - import check before spawning
        self.ctx = mp.get_context("spawn")
        self.jobs, self.results = self.ctx.Queue(), self.ctx.Queue()
        self.log, self.timeout_s = log, timeout_s
        self.procs = [self.ctx.Process(target=_worker, name=f"evopress-w{i}", daemon=True,
                                       args=(str(model_dir), d, str(spec_path), str(prep_dir), list(texts), dict(dcfg),
                                             embed, lm_head, damp, self.jobs, self.results))
                      for i, d in enumerate(devices)]
        try:
            for p in self.procs:
                p.start()
            ready = 0
            while ready < len(self.procs):
                msg = self._get()
                if msg[0] == "ready":
                    ready += 1
                    log(f"[baselines] evopress worker on {msg[1]} ready ({msg[2]} calibration batches)")
        except BaseException:          # a failed start must not leave the other workers holding their GPUs
            self.terminate()
            raise
        self._next = 0

    def terminate(self):
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            if p.pid is not None:
                p.join(timeout=30)

    def _get(self):
        t0 = time.time()
        while True:
            try:
                msg = self.results.get(timeout=10)
            except queue.Empty:
                dead = [p.name for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"evopress workers died: {dead}")
                if time.time() - t0 > self.timeout_s:
                    raise TimeoutError("evopress workers did not answer")
                continue
            if msg[0] == "error":
                raise RuntimeError(f"evopress worker on {msg[1]} failed:\n{msg[2]}")
            return msg

    def __call__(self, spec, norms: list[dict], stage: str) -> list[float]:
        from .quant.schema import module_assignment
        assert stage in STAGES
        ids = []
        for n in norms:
            self._next += 1
            ids.append(self._next)
            self.jobs.put((self._next, module_assignment(n, spec), stage))
        got = {}
        while len(got) < len(ids):
            _, jid, kl = self._get()
            got[jid] = kl
        return [got[i] for i in ids]

    def close(self):
        for _ in self.procs:
            self.jobs.put(None)
        for p in self.procs:
            p.join(timeout=60)
        self.terminate()
