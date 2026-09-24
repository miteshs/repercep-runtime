# VLA (OpenVLA-7B) on MI300X — the vendor-neutral row (2026-07-26)

**One line:** the exact same `VLAEngine` + OpenVLA path that ran on H100 runs on
**AMD Instinct MI300X** with **identical exact parity (0.0)** and the same
5–10× candidate-batching lever — **no code changes** (one backend-selector line).
This is the "runs on AMD, which OpenVLA/NIM don't optimize" point, measured.

## Setup

- **Box:** AMD Developer Cloud (`devcloud.amd.com`, DigitalOcean-backed), 1×
  **MI300X 192GB VF**, region `atl1`, $1.99/hr. Created via the web console
  (API can't order the `-devcloud` size); managed/deleted via the DO API.
- **Image/env:** Ubuntu 24.04, **ROCm 7.0.2**, Python 3.12. Ships **no torch** —
  installed **torch 2.10.0+rocm7.0** (+ torchvision) from
  `--index-url https://download.pytorch.org/whl/rocm7.0`, then a
  `venv --system-site-packages` for `transformers==4.40.1` etc. (see gotchas).
  `attn_implementation="sdpa"` (no flash-attn build). Same `openvla/openvla-7b`.
- **Run:** `python -O scripts/run_vla_openvla_gpu.py --candidates 8,16,32 --attn sdpa`
  — the *same committed driver* as H100; `select_backend` auto-picks `rocm`.

## Result — identical correctness, same lever

| Gate / metric | H100 | **MI300X** |
|---|---:|---:|
| greedy parity (vs `predict_action`) | 0.0 | **0.0** |
| batch parity (8 identical → batch-1) | 0.0 | **0.0** |
| lever **N=8** (loop → batched) | 985→182 = 5.43× | 1041→198 = **5.26×** |
| lever **N=16** | 1963→264 = 7.43× | 2088→278 = **7.52×** |
| lever **N=32** | 3905→443 = 8.83× | 4204→427 = **9.85×** |
| peak HBM | 23.9 GiB | 23.9 GiB |

MI300X per-decode latency tracks H100 closely (the 7-token decode is
launch/overhead-bound, not FLOP-bound, on both), and the candidate-batching
speedup is **silicon-agnostic** — it even edges ahead at N=32 (9.85× vs 8.83×).
That is the whole thesis: the control-regime serving lever is not a
CUDA-specific trick, and Repercep serves it on AMD with the same engine.

## ROCm gotchas hit (for the next run)

- **No preinstalled torch** on the ROCm 7.0.2 image (RunPod's H100 image has it).
  Install from the `rocm7.0` wheel index; it bundles its own ROCm runtime and
  sees the MI300X (`torch.version.hip=7.0.51831`, `cuda.is_available()=True`).
- **`--break-system-packages` fought a Debian-managed `typing_extensions`**
  (RECORD-less, can't uninstall). Fix: a **`venv --system-site-packages`** —
  inherits the ROCm torch, installs the rest cleanly, and dodges the memory's
  "PyPI torch clobbers ROCm torch" trap (never let `--ignore-installed` pull
  `torch` via `accelerate`'s dep). Needs `apt install python3.12-venv` first.
- **`select_backend(prefer="cuda")` raises on ROCm** (`torch.version.cuda` is
  None). Use `prefer=None` (auto) — it picks `rocm`. The device string stays
  `cuda:0` (ROCm torch masquerades HIP as cuda), so no pipeline change.
- **Access:** the account SSH key is injected, but the key must be **loaded in
  the agent** locally (`ssh-add`); `fail2ban` bans after ~3 bad attempts, so
  don't guess usernames — it's `root`. See [[amd-developer-cloud-access]].

## Caveats (unchanged from H100)

The candidate **scorer** (value / rollout / reward) is still the open modeling
question, not the batched decode; OpenVLA is stateless obs→action; verification
used a synthetic observation. The lever result stands regardless. The two
artificial OpenVLA batch>1 guards are lifted the same way (monkeypatch +
`python -O`) and proven correct by batch-parity 0.0 on AMD too.
