"""MI300X-vs-H100 LLM throughput gate. Pre-registered: docs/LLM_SILICON_GATE_PLAN.md

Deviation from the plan's "client co-located against 127.0.0.1", recorded
deliberately: this uses vLLM's in-process LLM API rather than the OpenAI server.
That is a TIGHTENING of the same concern (the plan wanted the network out of the
measurement; in-process removes it entirely) and it makes the two GPUs far more
comparable by deleting HTTP client/server variance. Continuous batching is still
exercised -- N prompts are submitted in one generate() call and the scheduler
batches them exactly as it would online.
"""
import argparse
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Plan §2.1. Reported separately, never averaged.
SHAPES = [
    {"tag": "S1-interactive",      "prompt_toks": 512,  "decode": 512, "conc": 1},
    {"tag": "S2-interactive-load", "prompt_toks": 512,  "decode": 512, "conc": 32},
    {"tag": "S3-batch",            "prompt_toks": 4096, "decode": 128, "conc": 32},
]
REPEATS = 5

ACC_PROMPTS = [
    "What is the capital of France?", "Explain gravity in one sentence.",
    "Write a haiku about winter.", "What is 17 * 23?",
    "Name three primary colors.", "Who wrote Pride and Prejudice?",
    "What does HTTP stand for?", "Define entropy briefly.",
    "What is the boiling point of water in Celsius?", "Translate 'good morning' to Spanish.",
    "What is the largest planet?", "Summarize photosynthesis in one line.",
    "What year did the Apollo 11 landing occur?", "What is a prime number?",
    "Name the longest river in Africa.", "What is the speed of light?",
    "Define recursion in programming.", "What is the chemical symbol for gold?",
    "How many continents are there?", "What is machine learning?",
]


def build_prompt(tok, n_tokens):
    """A prompt of exactly n_tokens, built by truncation so both GPUs see the
    identical token sequence."""
    filler = ("The quick brown fox jumps over the lazy dog. " * 2000)
    ids = tok(filler, add_special_tokens=False)["input_ids"][:n_tokens]
    assert len(ids) == n_tokens, f"wanted {n_tokens}, got {len(ids)}"
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default="/workspace/gate_result.json")
    # Added 2026-08-16 for the MoE gate (docs/LLM_MOE_GATE_PLAN.md). The dense
    # gate ran TP=1 on both vendors, which is why this never existed. Mixtral
    # 8x7B in bf16 (~87 GB) fits MI300X's 192 GB and does NOT fit one 80 GB
    # H100, so the H100 leg cannot be expressed at all without this flag.
    #
    # It is recorded in the result JSON because a TP=1-vs-TP=2 comparison is a
    # per-NODE result, not a per-GPU one. Quoting it as a silicon ratio would
    # be exactly the class of fake number the Cosmos 3 batching gate caught.
    # The plan's §4 requires any leg where this differs to say so in its first
    # paragraph; storing it means the JSON cannot silently lose that fact.
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # Default off so every dense-gate invocation is unchanged. Needed by some
    # MoE checkpoints whose config carries an auto_map; a no-op where vLLM has
    # a native implementation, and it does not affect what is measured.
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import torch
    gpu = torch.cuda.get_device_name(0)
    is_rocm = getattr(torch.version, "hip", None) is not None

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              enforce_eager=False, gpu_memory_utilization=0.90,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=args.trust_remote_code)

    result = {
        "gpu": gpu, "vendor": "AMD" if is_rocm else "NVIDIA",
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "<default>"),
        "model": args.model, "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpus_used": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "shapes": [],
    }
    try:
        import vllm

        result["vllm"] = vllm.__version__
    except Exception:
        result["vllm"] = "?"

    for shape in SHAPES:
        prompt = build_prompt(tok, shape["prompt_toks"])
        prompts = [prompt] * shape["conc"]
        # min==max with ignore_eos guarantees EXACTLY decode tokens per request:
        # the plan's equal-work check, enforced rather than measured after.
        sp = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            seed=0,
            max_tokens=shape["decode"],
            min_tokens=shape["decode"],
            ignore_eos=True,
        )

        llm.generate(prompts, sp)  # warm THIS shape (per-shape autotune lesson)

        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            times.append(time.perf_counter() - t0)

        gen_toks = sum(len(o.token_ids) for out in outs for o in out.outputs)
        expected = shape["conc"] * shape["decode"]
        med = sorted(times)[len(times) // 2]
        row = {
            **shape,
            "elapsed_s_median": round(med, 4),
            "elapsed_s_all": [round(t, 4) for t in times],
            "generated_tokens": gen_toks,
            "expected_tokens": expected,
            "equal_work_ok": gen_toks == expected,
            "output_tok_per_s": round(gen_toks / med, 2),
            "per_request_latency_s": round(med, 4),
        }
        result["shapes"].append(row)
        print(json.dumps(row), flush=True)

    # Accuracy smoke test (plan §6) -- greedy, saved for cross-GPU diff.
    acc_sp = SamplingParams(temperature=0.0, max_tokens=64, seed=0)
    acc_outs = llm.generate(ACC_PROMPTS, acc_sp)
    result["accuracy_smoke"] = [
        {"prompt": p, "output": o.outputs[0].text}
        for p, o in zip(ACC_PROMPTS, acc_outs, strict=True)
    ]

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("WROTE", args.out, flush=True)
    summary = {r["tag"]: r["output_tok_per_s"] for r in result["shapes"]}
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
