"""Values that are deliberately NOT configurable.

Everything here has a correct answer. Exposing these as environment variables
would create support burden and let a deployment break itself, so they live as
constants and are changed by editing code, not by editing a deployment.

Before moving anything out of this module into an environment variable, both
must hold:

1. it cannot be determined automatically or defaulted sensibly, and
2. the default path still works without it.

Inheriting a setting from ``worker-comfyui`` is not a justification: that
project is a general-purpose ComfyUI host and must be configurable. This one is
a specialized FLUX.2 worker and must not be.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# ComfyUI process
# --------------------------------------------------------------------------
# Internal to the container. Nothing outside the pod can reach these, so there
# is nothing for an operator to tune.
COMFY_HOST: Final = "127.0.0.1"
COMFY_PORT: Final = 8188
COMFY_BASE_URL: Final = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_ROOT: Final = "/comfyui"
COMFY_MODELS_ROOT: Final = f"{COMFY_ROOT}/models"

COMFY_START_TIMEOUT_S: Final = 300
COMFY_RESTART_TIMEOUT_S: Final = 180
COMFY_HEALTH_POLL_INTERVAL_S: Final = 1.0

# worker-comfyui exposes websocket reconnect settings. This worker has no
# websocket to reconnect: it polls /history, which cannot silently lose
# progress, so there is nothing here to tune.

# --------------------------------------------------------------------------
# Job execution
# --------------------------------------------------------------------------
# Bounded per job, below any sane endpoint execution timeout. Without this one
# pathological request can occupy a GPU indefinitely.
INFERENCE_TIMEOUT_S: Final = 600

# Failure-recovery policy, not deployment policy. On serverless a poisoned
# worker keeps pulling jobs off the queue and failing them, so a worker that
# cannot recover must exit and let the platform replace it.
OOM_STRIKES_BEFORE_EXIT: Final = 2
OOM_STRIKE_WINDOW_S: Final = 600

# Fragmentation is a real OOM cause with highly variable context lengths
# (reference images change the sequence length between requests).
TORCH_ALLOC_CONF: Final = "expandable_segments:True"

# --------------------------------------------------------------------------
# Reference image download
# --------------------------------------------------------------------------
# This layer runs in front of other people's infrastructure. The limits are
# security-relevant, so they are not operator-tunable.
MAX_IMAGE_BYTES: Final = 20 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES: Final = 60 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT_S: Final = 20
IMAGE_TOTAL_DOWNLOAD_TIMEOUT_S: Final = 60
MAX_REDIRECTS: Final = 3
ALLOWED_IMAGE_MIME: Final = frozenset({"image/jpeg", "image/png", "image/webp"})

# Blocked network ranges. Cloud metadata (169.254.169.254) lives inside
# 169.254.0.0/16; it is called out here because it is the single most valuable
# SSRF target on a cloud host.
BLOCKED_IPV4_NETWORKS: Final = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "224.0.0.0/4",
    "240.0.0.0/4",
)
BLOCKED_IPV6_NETWORKS: Final = (
    "::1/128",
    "::/128",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)

# --------------------------------------------------------------------------
# Request limits
# --------------------------------------------------------------------------
MAX_STEPS: Final = 60
MAX_PROMPT_CHARS: Final = 8000  # bounded by the Qwen3 encoder context
MIN_DIMENSION: Final = 256
MAX_DIMENSION: Final = 4096

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
# base64 is the primary path, not a degraded fallback. This ceiling exists so
# an oversized response fails with a clear error instead of being rejected by
# the platform.
MAX_INLINE_RESPONSE_BYTES: Final = 5 * 1024 * 1024

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
API_VERSION: Final = "1"
