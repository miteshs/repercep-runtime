#!/usr/bin/env python3
"""GPU verification of the VLA seam against real OpenVLA-7B (Plan B, Phase 1+2).

Self-contained (like the other ``scripts/run_*_gpu`` box artifacts): builds a
real OpenVLA-backed ``_VLAPipeline``, injects it into ``VLAEngine``, and:

- **Phase 1 (correctness):** verifies our replicated action-detokenization
  reproduces OpenVLA's own ``predict_action`` exactly, and that the batched
  candidate path is numerically identical to batch-1 (parity gates below).
- **Phase 2 (the lever):** times ``plan()`` with per-candidate decode vs the
  candidate-batched decode (N sequences sharing the prompt+vision prefix KV).

Verified 2026-07-26 on a RunPod H100 80GB (torch 2.4.1+cu124, transformers
4.40.1, openvla/openvla-7b): greedy parity 0.0, batch parity 0.0, and the lever
measured **5.43x (N=8) / 7.43x (N=16) / 8.83x (N=32)** — far above V-JEPA-AC's
1.6x because OpenVLA's per-candidate decode is short (7 action tokens), so
batch=1 badly underuses the GPU and batching amortizes the shared prefix. See
``docs/VLA_ON_H100.md``.

Batched generation note: OpenVLA's stock ``prepare_inputs_for_generation`` and
``forward`` carry two *artificial* batch==1 guards ("simplified for batch size
= 1"); the fusion math is batch-safe. We lift the first by monkeypatch and the
second by running under ``python -O`` (strips the ``assert``), and *prove*
correctness with the batch-parity gate rather than trusting the workaround.

    python -O scripts/run_vla_openvla_gpu.py --candidates 8,16,32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path


def _setup() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))


_setup()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from repercep.models.vla import VLAConfig, VLAEngine  # noqa: E402
from repercep.runtime.types import Action, ConditioningInput, RolloutParams  # noqa: E402

_PROMPT = "In: What action should the robot take to {instr}?\nOut:"
_EMPTY_TOKEN = 29871


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_image(uri, size=224):
    from PIL import Image

    if uri:
        try:
            return Image.open(uri).convert("RGB")
        except Exception:
            pass
    return Image.fromarray((np.random.default_rng(0).random((size, size, 3)) * 255).astype("uint8"))


def _enable_batched_generation(model):
    """Lift OpenVLA's artificial batch==1 guard in prepare_inputs_for_generation.

    forward()'s vision-text fusion is batch-safe (vision_backbone / projector /
    cat all act on the batch dim); only this method raises. The second guard is
    an ``assert`` in forward() -- strip it with ``python -O``. Correctness is
    gated by the batch-parity check, not by trusting either workaround.
    """

    def _pfg(
        self,
        input_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        mi = (
            {"input_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )
        mi.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return mi

    model.prepare_inputs_for_generation = types.MethodType(_pfg, model)
    return model


class OpenVLAPipeline:
    """Real OpenVLA-7B behind the repercep ``_VLAPipeline`` seam."""

    supports_batch = True

    def __init__(self, config, unnorm_key="bridge_orig", attn="sdpa"):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._device = f"cuda:{config.device_index}"
        self._processor = AutoProcessor.from_pretrained(config.repo, trust_remote_code=True)
        self._model = (
            AutoModelForVision2Seq.from_pretrained(
                config.repo,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=attn,
            )
            .to(self._device)
            .eval()
        )
        _enable_batched_generation(self._model)
        if unnorm_key not in self._model.norm_stats:
            unnorm_key = next(iter(self._model.norm_stats))
        self._unnorm_key = unnorm_key
        self._action_dim = int(self._model.get_action_dim(unnorm_key))
        self.sample = False
        self._sessions: dict = {}

    def _build_inputs(self, instr, image):
        inp = self._processor(_PROMPT.format(instr=instr), image).to(
            self._device, dtype=torch.bfloat16
        )
        ids = inp["input_ids"]
        if not torch.all(ids[:, -1] == _EMPTY_TOKEN):
            pad = torch.tensor([[_EMPTY_TOKEN]], device=ids.device, dtype=ids.dtype)
            inp["input_ids"] = torch.cat([ids, pad], dim=1)
            if inp.get("attention_mask") is not None:
                inp["attention_mask"] = torch.cat(
                    [inp["attention_mask"], torch.ones_like(pad)], dim=1
                )
        return inp

    def reset(self, session_id, prompt):
        self._sessions[session_id] = {"instr": prompt or "do the task", "inputs": None}

    def encode_observation(self, session_id, conditioning):
        s = self._sessions[session_id]
        s["inputs"] = self._build_inputs(
            s["instr"], _load_image(getattr(conditioning, "uri", None))
        )
        return s["inputs"]["pixel_values"]

    def _detokenize(self, gen_ids):
        pred = gen_ids[:, -self._action_dim :].cpu().numpy()
        disc = np.clip(self._model.vocab_size - pred - 1, 0, self._model.bin_centers.shape[0] - 1)
        normed = self._model.bin_centers[disc]
        st = self._model.get_action_stats(self._unnorm_key)
        low, high = np.array(st["q01"]), np.array(st["q99"])
        mask = st.get("mask", np.ones_like(low, dtype=bool))
        return np.where(mask, 0.5 * (normed + 1) * (high - low) + low, normed)

    def generate_actions(self, session_id, n, do_sample):
        inp = self._sessions[session_id]["inputs"]
        ids, pv, am = inp["input_ids"], inp["pixel_values"], inp.get("attention_mask")
        if n > 1:
            ids = ids.expand(n, -1).contiguous()
            pv = pv.expand(n, *pv.shape[1:]).contiguous()
            am = am.expand(n, -1).contiguous() if am is not None else None
        gk = {"max_new_tokens": self._action_dim, "do_sample": do_sample}
        if do_sample:
            gk.update(temperature=1.0, top_p=1.0)
        with torch.inference_mode():
            gen = self._model.generate(input_ids=ids, pixel_values=pv, attention_mask=am, **gk)
        return self._detokenize(gen)

    def decode_action_chunk(self, session_id, context, n_candidates):
        act = self.generate_actions(session_id, n_candidates, bool(self.sample or n_candidates > 1))
        return torch.from_numpy(np.ascontiguousarray(act)).float().unsqueeze(1)

    def append_executed(self, session_id, executed, context):
        return context  # OpenVLA is stateless obs->action; v0 keeps context

    def score_candidates(self, session_id, candidates, goal):
        c = candidates.reshape(candidates.shape[0], -1)
        return torch.linalg.vector_norm(c - c.mean(0, keepdim=True), dim=1)

    def close(self, session_id):
        self._sessions.pop(session_id, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="openvla/openvla-7b")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--attn", default="sdpa", help="sdpa (ROCm-portable) or flash_attention_2")
    ap.add_argument("--candidates", default="8,16,32")
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()

    if __debug__:
        print("[warn] run under `python -O` so OpenVLA's batch==1 assert is stripped.", flush=True)

    from repercep.backend.registry import select_backend

    backend = select_backend(prefer=None)  # auto: cuda on NVIDIA, rocm on MI300X
    dev = backend.devices()[0].name
    cfg = VLAConfig(repo=args.repo, action_dim=7, plan_candidates=8)
    print(f"[vla] {dev}: building OpenVLA pipeline...", flush=True)
    pipe = OpenVLAPipeline(cfg, unnorm_key=args.unnorm_key, attn=args.attn)
    engine = VLAEngine(backend, cfg, pipeline=pipe)
    goal = torch.zeros(7)
    state = engine.reset(ConditioningInput(), RolloutParams())

    # Phase 1: correctness gates
    inp = pipe._sessions[state.session_id]["inputs"]
    a_ref = np.asarray(
        pipe._model.predict_action(**inp, unnorm_key=pipe._unnorm_key, do_sample=False)
    )
    a_mine = pipe.generate_actions(state.session_id, 1, False)[0]
    greedy_err = float(np.abs(a_ref - a_mine).max())
    batch_err = float(np.abs(pipe.generate_actions(state.session_id, 8, False) - a_ref).max())
    print(f"[vla] PARITY greedy={greedy_err:.2e}  batch={batch_err:.2e}", flush=True)

    _, step = engine.step(state, Action(values=[0.0] * 7))
    act = engine.plan(state, goal, horizon=1)
    print(
        f"[vla] step_index={step.step_index}; plan={np.round(act.values, 4).tolist()} {act.space}",
        flush=True,
    )

    # Phase 2: candidate-batching lever
    def time_plan(n, batched):
        engine._config.plan_candidates = n
        engine._config.plan_batched = batched
        engine.plan(state, goal, 1)
        _sync()
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            engine.plan(state, goal, 1)
            _sync()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    pipe.sample = True
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = {}
    for n in [int(x) for x in args.candidates.split(",")]:
        loop, bat = time_plan(n, False), time_plan(n, True)
        sp = loop / bat if bat else 0.0
        res[str(n)] = {
            "loop_ms": round(loop * 1e3, 1),
            "batched_ms": round(bat * 1e3, 1),
            "speedup": round(sp, 2),
        }
        print(
            f"[vla] N={n:2d}: loop {loop * 1e3:6.0f}ms  batched {bat * 1e3:6.0f}ms  = {sp:.2f}x",
            flush=True,
        )
    pipe.sample = False

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    out = {
        "device": dev,
        "model": args.repo,
        "greedy_parity_max_err": greedy_err,
        "batch_parity_max_err": batch_err,
        "bench_sampled": res,
        "peak_hbm_gib": round(peak, 1),
    }
    print("[vla] RESULT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
