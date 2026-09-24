"""The model-specific DreamZero pipeline -- Phase 1 of the port.

Implements the :class:`~repercep.models.dreamzero._DreamZeroPipeline` Protocol
against the real components, treating the research repo
(``dreamzero0/dreamzero``, Apache-2.0) as a dependency the same way
``lingbot_va_pipeline.py`` treats ``wan_va`` -- ``sys.path``-inserted, never
``pip install``-ed (the pyproject hard-requires ``tensorrt``/``ray``/
``mujoco``/``deepspeed``/a ``gear`` package that plain imports never touch).
Confirmed working on an H100 2026-07-13 -- see ``docs/DREAMZERO_PORT_PLAN.md``
§2b for the full verification trail (exact numbers, load-bearing values, and
why each one needed checking rather than guessing).

What this module owns, session-keyed, that the reference doesn't need
(single global-state ``ARDroidRoboarenaPolicy`` process per rank):

- **the session-swap mechanism**: unlike LingBot-VA's ``wan_va`` transformer
  (named multi-session cache via a ``cache_name`` kwarg),
  ``WANPolicyHead``'s KV/cross-attn caches, ``current_start_frame``,
  ``language``, ``ys``, and ``clip_feas`` are plain mutable instance
  attributes on one ``nn.Module`` -- there is no per-call cache-name
  parameter. :func:`_swap_in`/:func:`_swap_out` save/restore these around
  every model call, so one resident model serves N sessions without
  cross-talk;
- the 2x2 multi-camera grid layout (``dreamzero_cotrain.py``'s
  ``_prepare_video``, DROID branch -- wrist doubled across the top row,
  left/right exterior on the bottom row);
- ``embodiment_id`` resolution from the checkpoint's own
  ``experiment_cfg/conf.yaml`` ``embodiment_tag_mapping`` registry (not
  hardcoded -- a wrong value would silently misroute through another
  embodiment's trained weights rather than crash, so it's read per
  checkpoint, not assumed);
- action (and state) denormalization read directly from
  ``experiment_cfg/metadata.json``'s per-channel q01/q99 stats -- the same
  formula LingBot-VA's ``_denormalize_actions`` already implements,
  reimplemented here rather than depending on ``groot.vla.data.transform``.

The denoise loop itself is untouched reference code
(``WANPolicyHead.lazy_joint_video_action``) -- this module's job is
constructing its inputs and managing what state persists across calls.

GPU-only: nothing here runs without the checkpoint bundle and a CUDA device.
The engine's unit tests inject a fake pipeline instead
(``tests/test_dreamzero.py``).
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend
    from repercep.models.dreamzero import DreamZeroConfig
    from repercep.runtime.types import ConditioningInput

_IMPORT_HELP = (
    "DreamZero pipeline needs the research repo importable as 'groot' "
    "(git clone https://github.com/dreamzero0/dreamzero and add its root to "
    "sys.path, e.g. REPERCEP_DREAMZERO_SRC=/path/to/dreamzero) -- do NOT "
    "'pip install' it (pulls in tensorrt/ray/mujoco/deepspeed/gear, none of "
    "which plain imports need, and tensorrt won't build on ROCm) -- see "
    "docs/DREAMZERO_PORT_PLAN.md §4."
)

#: Camera resolution DreamZero-DROID was trained at (experiment_cfg/conf.yaml
#: VideoResize), before the 2x2 grid concat.
_CAM_H, _CAM_W = 176, 320

#: DROID's used channel order within the model's padded action_dim=32 /
#: max_state_dim=64 registers -- confirmed against conf.yaml's
#: action_concat_order / state_concat_order (port plan §2b).
_DROID_ACTION_KEYS = ("joint_position", "gripper_position")

#: Instance attributes on WANPolicyHead that make up its (non-multi-session)
#: native state -- swapped per session_id around every model call.
_SWAP_ATTRS = (
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "clip_feas",
    "ys",
    "current_start_frame",
    "language",
)


def _ensure_groot_importable() -> None:
    src = os.environ.get("REPERCEP_DREAMZERO_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import groot  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(_IMPORT_HELP) from exc


#: The reference's static DiT-call-skip mask only has hand-tuned presets for
#: these four values (``WANPolicyHead.__init__``, GPU-verified 2026-07-14 by
#: reading ``wan_flow_matching_action_tf.py`` directly) -- ``NUM_DIT_STEPS``
#: set to anything else (including unset, or our "full compute" choice of 16)
#: falls through to the ``else`` branch, an all-``True`` 16-of-16 mask. There
#: is no smooth 1-16 range; a DiT-step-scan lever must sweep exactly these
#: four values plus "full" (any other value), not arbitrary integers.
DIT_STEP_MASK_PRESETS = (5, 6, 7, 8)


def build_pipeline(backend: Backend, config: DreamZeroConfig) -> DreamZeroPipeline:
    """Import the research package and construct the real pipeline.

    Sets two env vars ``WANPolicyHead.__init__`` reads **once, at
    construction time, before any pipeline call runs** (GPU-verified
    2026-07-14 by reading the source directly -- both assumptions this
    function's env vars replace were wrong until then):

    - ``TORCHDYNAMO_DISABLE``: forces eager mode unless ``config.compile``
      opts in. The reference wraps several submodules (TextEncoder,
      ImageEncoder, VAE, the flow-matching scheduler's
      ``multistep_uni_p_bh_update``) in ``torch.compile`` unconditionally at
      construction time, and the scheduler's history tensors change rank step
      to step, which hits Dynamo's ``FailOnRecompileLimitHit`` on the very
      first chunk (port plan §2b).
    - ``NUM_DIT_STEPS``: baked into ``self.dit_step_mask`` at
      ``WANPolicyHead.__init__`` (see :data:`DIT_STEP_MASK_PRESETS`) --
      **not** re-read per call, despite being named like a per-call
      parameter. A first attempt at wiring this from inside ``_infer()``
      (post-construction) silently had zero effect: every chunk kept logging
      the reference's own default 8-step mask regardless of
      ``config.num_dit_steps``, because the env var wasn't set until after
      the model — and its mask — already existed. A fresh pipeline is
      required to actually change this lever; mutating ``config.num_dit_steps``
      on an already-built engine does nothing (same for
      ``DYNAMIC_CACHE_SCHEDULE`` below, read at the same line).
    - ``DYNAMIC_CACHE_SCHEDULE``: same construction-time-only story as
      ``NUM_DIT_STEPS`` (same ``__init__`` block) -- ``"True"``/``"False"``
      confirmed against the source's own
      ``os.getenv(..., "False").lower() == "true"`` check.
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "0" if config.compile else "1"
    os.environ["NUM_DIT_STEPS"] = str(config.num_dit_steps)
    os.environ["DYNAMIC_CACHE_SCHEDULE"] = "True" if config.enable_dit_cache else "False"
    _ensure_groot_importable()
    return DreamZeroPipeline(backend, config)


class DreamZeroPipeline:
    """Session-keyed DreamZero inference on one resident ``WANPolicyHead``."""

    def __init__(self, backend: Backend, config: DreamZeroConfig) -> None:
        import torch
        from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

        self._config = config
        self._device = backend.torch_device(config.device_index)
        self._dtype = getattr(torch, config.dtype)

        root = config.repo if os.path.isdir(config.repo) else _download_checkpoint(config.repo)
        self._root = root
        with open(os.path.join(root, "config.json")) as f:
            config_dict = json.load(f)
        ah_cfg = config_dict["action_head_cfg"]["config"]
        # Own the loading path: a full finetune (train_architecture="full")
        # already covers the whole model in its own shards -- the reference
        # default would otherwise download the ~65.6 GB Wan2.1 base DiT for
        # nothing, since it's immediately overwritten strict=False below.
        ah_cfg["skip_component_loading"] = True
        if "defer_lora_injection" in ah_cfg:
            ah_cfg["defer_lora_injection"] = False
        vla_config = VLAConfig(**config_dict)

        model = VLA(vla_config)
        _load_shards(model, root)
        model.eval()
        model.requires_grad_(False)
        model.to(dtype=self._dtype)
        model.to(device=self._device)
        model.post_initialize()

        self._model = model
        self._action_head = model.action_head
        self._embodiment_id = _resolve_embodiment_id(root)
        self._action_stats = _load_action_stats(root, _DROID_ACTION_KEYS)
        self._tokenizer = _load_tokenizer()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._cfg_batched_warned = False

    # --- _DreamZeroPipeline protocol ---

    def reset(self, session_id: str, prompt: str | None) -> None:
        self._sessions[session_id] = {
            "prompt": prompt or "",
            "swap": dict.fromkeys(_SWAP_ATTRS),
            "pending_latent_video": None,
            "last_video_pred": None,
        }
        # A fresh session's swap has current_start_frame absent -> treat as 0
        # (WANPolicyHead's own default) so _swap_in doesn't need a special case.
        self._sessions[session_id]["swap"]["current_start_frame"] = 0

    def encode_observation(self, session_id: str, conditioning: ConditioningInput) -> torch.Tensor:
        """The seed forward: DreamZero has no lighter-weight "just encode"
        primitive separate from "run one block" (unlike LingBot-VA's
        streaming-VAE encode) -- ``lazy_joint_video_action`` bundles cache
        creation + the seed frame's clean K/V write + the first block's
        denoise loop into one call whenever ``current_start_frame == 0``. So
        this runs the real model once, same cost as any other chunk.
        """
        import torch

        if not conditioning.uri:
            raise ValueError("DreamZero reset needs conditioning.uri = camera image directory")
        canvas = _load_grid_frame(conditioning.uri)  # (2*_CAM_H, 2*_CAM_W, 3) uint8
        images = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).to(self._device)  # (1,1,H,W,3)
        session = self._sessions[session_id]
        latents, actions = self._infer(session, images, latent_video=None)
        session["last_video_pred"] = latents
        session["pending_actions"] = actions
        return _flatten_latent(latents)

    def infer_chunk(
        self, session_id: str, current_start_frame: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        session = self._sessions[session_id]
        latent_video = session.pop("pending_latent_video", None)
        if latent_video is None:
            # Imagination mode (no recondition() call pushed real reality):
            # the model's own last prediction stands in.
            latent_video = session.get("last_video_pred")
        cam_h2, cam_w2 = 2 * _CAM_H, 2 * _CAM_W
        # data["images"] shape must match even when its content is bypassed
        # by latent_video -- see port plan §2b; zeros are fine, unused.
        dummy = torch.zeros(
            1,
            self._config.num_frame_per_block,
            cam_h2,
            cam_w2,
            3,
            dtype=torch.uint8,
            device=self._device,
        )
        latents, actions = self._infer(session, dummy, latent_video=latent_video)
        session["last_video_pred"] = latents
        return _flatten_latent(latents), actions

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        current_start_frame: int,
    ) -> int:
        session = self._sessions[session_id]
        if obs_latent is not None:
            session["pending_latent_video"] = _unflatten_latent(obs_latent)
        else:
            session["pending_latent_video"] = None  # infer_chunk falls back to last_video_pred
        return self._config.num_frame_per_block

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # --- internals ---

    def _infer(
        self, session: dict[str, Any], images: torch.Tensor, *, latent_video: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch
        from transformers.feature_extraction_utils import BatchFeature

        cfg = self._config
        self._apply_levers(cfg)
        text_ids, text_mask = self._tokenize(session["prompt"])
        neg_ids, neg_mask = self._tokenize("")
        state = torch.zeros(
            1, cfg.num_state_per_block, cfg.max_state_dim, dtype=self._dtype, device=self._device
        )
        embodiment_id = torch.tensor([self._embodiment_id], device=self._device)

        action_input = BatchFeature(
            data={
                "images": images,
                "text": text_ids,
                "text_attention_mask": text_mask,
                "text_negative": neg_ids,
                "text_attention_mask_negative": neg_mask,
                "state": state,
                "embodiment_id": embodiment_id,
            }
        )
        backbone_output = BatchFeature(
            data={"backbone_features": torch.empty(1, 1, 0, device=self._device)}
        )

        _swap_in(self._action_head, session["swap"])
        with torch.inference_mode():
            out = self._action_head.lazy_joint_video_action(
                backbone_output, action_input, latent_video=latent_video
            )
        _swap_out(self._action_head, session["swap"], _SWAP_ATTRS)

        video_pred = out["video_pred"]
        actions_norm = out["action_pred"]  # (1, action_horizon, action_dim) normalized
        used_dim = cfg.used_action_dim or cfg.action_dim
        actions = _denormalize_actions(actions_norm, self._action_stats, used_dim)
        return video_pred, actions

    def _apply_levers(self, cfg: DreamZeroConfig) -> None:
        """Wire the Phase-2 latency levers (port plan §4) that ARE plain
        per-call-read instance attributes -- confirmed by reading the
        source directly, 2026-07-14. ``num_dit_steps``/``enable_dit_cache``
        are NOT here: they're baked into ``self.dit_step_mask`` /
        ``self.dynamic_cache_schedule`` once at ``WANPolicyHead.__init__``
        (see :func:`build_pipeline`'s docstring) and mutating them here had
        silently zero effect until that was caught on the first real H100
        run -- a fresh pipeline is required to change either.

        - ``cfg_scale``: ``WANPolicyHead.cfg_scale`` is a plain attribute
          (``__init__`` hardcodes ``self.cfg_scale = 5.0``, read fresh at the
          CFG-combine line every diffusion step) that was previously never
          wired from ``DreamZeroConfig`` at all -- invisible until now only
          because our config default (5.0) happened to match theirs.
        - ``local_attn_size``: lives on ``self._action_head.model`` (the
          actual DiT / ``CausalWanModel``), **not** ``self._action_head``
          itself (``WANPolicyHead`` reads ``self.model.local_attn_size`` at
          its reset-condition check) -- read fresh per call by the DiT's own
          forward-path attribute accesses, so this one genuinely is
          per-call-mutable, unlike the two env-var levers above.
        """
        self._action_head.cfg_scale = cfg.cfg_scale
        if cfg.local_attn_size is not None:
            self._action_head.model.local_attn_size = cfg.local_attn_size
        if cfg.cfg_batched and not self._cfg_batched_warned:
            self._cfg_batched_warned = True
            import warnings

            warnings.warn(
                "DreamZeroConfig.cfg_batched=True requested, but batching "
                "cond+uncond into one forward (port plan §4 lever 1) is not "
                "yet implemented in DreamZeroPipeline -- falling back to the "
                "reference's sequential cond/uncond forwards. Implementing "
                "this needs WANPolicyHead._run_diffusion_steps read from the "
                "research clone on a GPU pod (not available in this "
                "environment) -- see docs/DREAMZERO_PORT_PLAN.md §4.",
                stacklevel=2,
            )

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self._tokenizer(
            [text], padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        return enc.input_ids.to(self._device), enc.attention_mask.to(self._device)


# --- internals (module-level helpers) ---


def _download_checkpoint(repo: str) -> str:
    from huggingface_hub import snapshot_download

    path: str = snapshot_download(
        repo, ignore_patterns=["tensorrt/*", "trainer_state.json", "wandb_config.json"]
    )
    return path


def _load_shards(model: Any, root: str) -> None:
    from safetensors.torch import load_file

    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        index = json.load(f)
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(os.path.join(root, shard_file)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DreamZero checkpoint at {root!r} doesn't match the model: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys "
            f"(sample: {missing[:5] or unexpected[:5]})"
        )


def _resolve_embodiment_id(root: str) -> int:
    """Read the checkpoint's own ``embodiment_tag_mapping`` registry.

    Not hardcoded: this indexes a per-embodiment weight matrix
    (``CategorySpecificLinear``) inside the DiT -- a wrong value runs without
    error and silently misroutes through another embodiment's trained
    weights, so it's resolved per checkpoint rather than assumed (port plan
    §2b). Needs PyYAML; conf.yaml is small (~350 KB) and already fetched
    alongside the checkpoint.
    """
    import yaml

    with open(os.path.join(root, "experiment_cfg", "metadata.json")) as f:
        metadata = json.load(f)
    (embodiment_tag,) = metadata.keys()
    with open(os.path.join(root, "experiment_cfg", "conf.yaml")) as f:
        conf = yaml.safe_load(f)
    mapping = _find_embodiment_tag_mapping(conf)
    return int(mapping[embodiment_tag])


def _find_embodiment_tag_mapping(node: Any) -> dict[str, int]:
    """DFS for the first ``embodiment_tag_mapping`` dict in the nested config.

    ``conf.yaml`` repeats an identical copy of this registry under every
    transform stanza -- any one of them is authoritative, so the first found
    wins.
    """
    if isinstance(node, dict):
        if "embodiment_tag_mapping" in node and isinstance(node["embodiment_tag_mapping"], dict):
            return dict(node["embodiment_tag_mapping"])
        for value in node.values():
            found = _find_embodiment_tag_mapping(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_embodiment_tag_mapping(item)
            if found is not None:
                return found
    return None  # type: ignore[return-value]


def _load_action_stats(root: str, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """q01/q99 per channel, concatenated in ``keys`` order -- see port plan §2b."""
    import torch

    with open(os.path.join(root, "experiment_cfg", "metadata.json")) as f:
        metadata = json.load(f)
    (embodiment_tag,) = metadata.keys()
    action_stats = metadata[embodiment_tag]["statistics"]["action"]
    q01 = torch.cat([torch.tensor(action_stats[k]["q01"], dtype=torch.float32) for k in keys])
    q99 = torch.cat([torch.tensor(action_stats[k]["q99"], dtype=torch.float32) for k in keys])
    return {"q01": q01, "q99": q99}


def _load_tokenizer() -> Any:
    """umT5 tokenizer from the Wan base repo -- DROID only ships DiT/action-head
    deltas. The dir has no model config.json (tokenizer-only), so
    AutoTokenizer/AutoConfig can't resolve it; load the fast-tokenizer file
    directly instead (port plan §2b).
    """
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    wan_repo = "Wan-AI/Wan2.1-I2V-14B-480P"
    tokenizer_json = hf_hub_download(wan_repo, "google/umt5-xxl/tokenizer.json")
    special_tokens_map = hf_hub_download(wan_repo, "google/umt5-xxl/special_tokens_map.json")
    with open(special_tokens_map) as f:
        specials = json.load(f)
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        pad_token=specials.get("pad_token", "<pad>"),
        eos_token=specials.get("eos_token", "</s>"),
        unk_token=specials.get("unk_token", "<unk>"),
    )


def _load_grid_frame(obs_dir: str) -> Any:
    """Per-camera PNGs -> the 2x2 grid canvas DreamZero-DROID was trained on.

    Layout confirmed against ``dreamzero_cotrain.py``'s ``_prepare_video``
    (DROID branch) and independently cross-checked via exact arithmetic
    against ``frame_seqlen=880`` (port plan §2b): wrist view doubled in
    width fills the top row, left/right exterior views fill the bottom row.
    """
    import numpy as np
    from PIL import Image

    def _load(name: str) -> Any:
        img = Image.open(os.path.join(obs_dir, f"{name}.png")).convert("RGB")
        img = img.resize((_CAM_W, _CAM_H), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    left = _load("exterior_image_1_left")
    right = _load("exterior_image_2_left")
    wrist = _load("wrist_image_left")

    canvas = np.zeros((2 * _CAM_H, 2 * _CAM_W, 3), dtype=np.uint8)
    canvas[:_CAM_H, :] = np.repeat(wrist, 2, axis=1)
    canvas[_CAM_H:, :_CAM_W] = left
    canvas[_CAM_H:, _CAM_W:] = right
    return canvas


def _swap_in(action_head: Any, saved: dict[str, Any]) -> None:
    for attr, value in saved.items():
        setattr(action_head, attr, value)


def _swap_out(action_head: Any, saved: dict[str, Any], attrs: tuple[str, ...]) -> None:
    for attr in attrs:
        saved[attr] = getattr(action_head, attr)


def _flatten_latent(latent5d: torch.Tensor) -> torch.Tensor:
    """``(1, C, T, H, W)`` -> ``(T, C*H*W)`` -- the seam's ``WorldState.context``
    layout, same convention as ``lingbot_va_pipeline.flatten_latent5d``."""
    _b, c, t, h, w = latent5d.shape
    return latent5d[0].permute(1, 0, 2, 3).reshape(int(t), c * h * w)


def _unflatten_latent(flat: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_flatten_latent`. Geometry is fixed for DreamZero-DROID
    (out_dim=16, 44x80 latent spatial -- port plan §2b), unlike LingBot-VA
    where it varies per task config."""
    c, h, w = 16, 44, 80
    return flat.reshape(-1, c, h, w).permute(1, 0, 2, 3).unsqueeze(0)


def _denormalize_actions(
    actions_norm: torch.Tensor, stats: dict[str, torch.Tensor], used_dim: int
) -> torch.Tensor:
    """``(x+1)/2*(q99-q01)+q01`` per channel -- StateActionTransform's mode="q99"
    (port plan §2b), applied to the used [0:used_dim] slice of the model's
    padded action register."""
    q01, q99 = stats["q01"], stats["q99"]
    used = actions_norm[0, :, :used_dim].float().cpu()  # (action_horizon, used_dim)
    return (used + 1) / 2 * (q99 - q01) + q01
