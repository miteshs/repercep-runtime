"""World-model engines for Repercep.

Generation camp: Cosmos-Predict-7B and Wan-2.2 (T2V-A14B). Control camp on the
interactive seam: V-JEPA 2-AC (energy-MPC planning), LingBot-VA 2.0, DreamZero
(video-action policy), Cosmos 3 Nano (omnimodal policy, Phase-0 scaffold), and
a token-decoding VLA (OpenVLA / π0-FAST class, Phase-0 scaffold) — three
planning regimes (search / generate / decode+score), one Protocol.
"""

from __future__ import annotations

from repercep.models.cosmos import (
    DEFAULT_REPO,
    CosmosConfig,
    CosmosEngine,
    GuardrailError,
)
from repercep.models.cosmos3 import (
    DEFAULT_REPO as COSMOS3_DEFAULT_REPO,
)
from repercep.models.cosmos3 import (
    Cosmos3Config,
    Cosmos3Engine,
)
from repercep.models.dreamzero import (
    DEFAULT_REPO as DREAMZERO_DEFAULT_REPO,
)
from repercep.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroEngine,
)
from repercep.models.lingbot_va import (
    DEFAULT_REPO as LINGBOT_VA_DEFAULT_REPO,
)
from repercep.models.lingbot_va import (
    LingBotVAConfig,
    LingBotVAEngine,
)
from repercep.models.vjepa2_ac import (
    DEFAULT_ENCODER_REPO,
    VJepa2ACConfig,
    VJepa2ACEngine,
)
from repercep.models.vla import (
    DEFAULT_REPO as VLA_DEFAULT_REPO,
)
from repercep.models.vla import (
    VLAConfig,
    VLAEngine,
)
from repercep.models.wan import DEFAULT_REPO as WAN_DEFAULT_REPO
from repercep.models.wan import NATIVE_FPS as WAN_NATIVE_FPS
from repercep.models.wan import SMALL_REPO as WAN_SMALL_REPO
from repercep.models.wan import WanConfig, WanEngine

__all__ = [
    "COSMOS3_DEFAULT_REPO",
    "DEFAULT_ENCODER_REPO",
    "DEFAULT_REPO",
    "DREAMZERO_DEFAULT_REPO",
    "LINGBOT_VA_DEFAULT_REPO",
    "VLA_DEFAULT_REPO",
    "WAN_DEFAULT_REPO",
    "WAN_NATIVE_FPS",
    "WAN_SMALL_REPO",
    "Cosmos3Config",
    "Cosmos3Engine",
    "CosmosConfig",
    "CosmosEngine",
    "DreamZeroConfig",
    "DreamZeroEngine",
    "GuardrailError",
    "LingBotVAConfig",
    "LingBotVAEngine",
    "VJepa2ACConfig",
    "VJepa2ACEngine",
    "VLAConfig",
    "VLAEngine",
    "WanConfig",
    "WanEngine",
]
