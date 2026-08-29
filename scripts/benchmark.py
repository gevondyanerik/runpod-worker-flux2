#!/usr/bin/env python3
"""Measure real VRAM and latency, on real hardware.

    python scripts/benchmark.py --refs 0 1 2 --sizes 1024x1024 1536x1024

Prints a table and, with --emit-probes, the ``VramProbe`` literals to paste
into ``app/variants.py``. Those literals must only ever come from this script:
a hand-written probe is worse than none, because it will be believed.

The numbers come from ComfyUI's ``/system_stats``, not from this process.
Inference happens in the ComfyUI subprocess, so this process's CUDA allocator
has touched nothing and would report a confident zero.

What is reported is device memory in use, read immediately after the job — not
a true high-water mark, because ComfyUI exposes no peak counter. Since the
worker runs one job at a time and nothing else shares the GPU, it tracks the
peak closely, but it is a proxy and is named as one.

Device totals, not torch's own counters: ComfyUI may run the CUDA allocator in
``cudaMallocAsync`` mode, where ``torch_vram_total`` reports a few megabytes
regardless of what the model is actually holding. Measured on an RTX PRO 4500
it read 33 MB while the device had 12.8 GB in use.

Must run inside the worker container.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config as config_module
from app import constants
from app import handler as handler_module
from app.comfy_client import ComfyClient


def reference_uri(width: int, height: int) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (180, 120, 90)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def device_stats() -> dict:
    """The first CUDA device as ComfyUI sees it."""
    url = f"{constants.COMFY_BASE_URL}/system_stats"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    devices = payload.get("devices") or [{}]
    return devices[0]


def vram_gb() -> tuple[float, float]:
    """(in use on the device, device total), in GB."""
    device = device_stats()
    total = float(device.get("vram_total", 0)) / 1e9
    free = float(device.get("vram_free", 0)) / 1e9
    return (max(0.0, total - free), total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--sizes", nargs="+", default=["1024x1024", "1536x1024"])
    parser.add_argument("--emit-probes", action="store_true")
    args = parser.parse_args()

    config = config_module.load()
    encoder = "fp4" if "fp4" in config.variant.text_encoder.filename else "bf16"

    comfy = ComfyClient()
    comfy.start()
    handler_module.init(comfy)

    gpu_name = str(device_stats().get("name", "unknown"))
    if "cuda" not in gpu_name.lower() and "cpu" in gpu_name.lower():
        print("ComfyUI is not on a CUDA device", file=sys.stderr)
        return 2

    print(f"{'size':<12}{'refs':>5}{'seconds':>10}{'used GB':>10}{'total GB':>10}")
    probes: list[str] = []

    for size in args.sizes:
        width, height = (int(part) for part in size.split("x"))
        for refs in args.refs:
            payload = {
                "prompt": "a product photograph on a plain background",
                "width": width,
                "height": height,
                "seed": 42,
                "images": [reference_uri(1024, 1024) for _ in range(refs)],
            }
            started = time.monotonic()
            response = handler_module.handler(
                {"id": f"bench-{size}-{refs}", "input": payload}
            )
            elapsed = time.monotonic() - started
            in_use, total = vram_gb()

            if "error" in response:
                print(f"{size:<12}{refs:>5}   {response['error']['code']}")
                continue

            print(f"{size:<12}{refs:>5}{elapsed:>10.1f}{in_use:>10.2f}{total:>10.2f}")
            probes.append(
                "    VramProbe(\n"
                f"        width={width},\n"
                f"        height={height},\n"
                f"        refs={refs},\n"
                f'        text_encoder="{encoder}",\n'
                f"        peak_allocated_gb={in_use:.2f},\n"
                f"        peak_reserved_gb={in_use:.2f},\n"
                f'        gpu_name="{gpu_name}",\n'
                f'        measured_at="{time.strftime("%Y-%m-%d")}",\n'
                "    ),"
            )

    if args.emit_probes:
        print("\nvram_probes=(")
        for probe in probes:
            print(probe)
        print("),")

    comfy.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
