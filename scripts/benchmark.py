#!/usr/bin/env python3
"""Measure real peak VRAM and latency, on real hardware.

    python scripts/benchmark.py --refs 0 1 2 --sizes 1024x1024 1536x1024

Prints a table and, with --emit-probes, the ``VramProbe`` literals to paste
into ``app/variants.py``. Those literals must only ever come from this script:
a hand-written probe is worse than none, because it will be believed.

Must run inside the worker container, with ComfyUI already started.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config as config_module
from app import handler as handler_module
from app.comfy_client import ComfyClient


def reference_uri(width: int, height: int) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (180, 120, 90)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peaks() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return (0.0, 0.0)
    return (
        torch.cuda.max_memory_allocated() / 1e9,
        torch.cuda.max_memory_reserved() / 1e9,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--sizes", nargs="+", default=["1024x1024", "1536x1024"])
    parser.add_argument("--emit-probes", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device: this script measures real hardware", file=sys.stderr)
        return 2

    gpu_name = torch.cuda.get_device_name(0)
    config = config_module.load()
    encoder = "fp4" if "fp4" in config.variant.text_encoder.filename else "bf16"

    comfy = ComfyClient()
    comfy.start()
    handler_module.init(comfy)

    print(f"{'size':<12}{'refs':>5}{'seconds':>10}{'alloc GB':>10}{'resvd GB':>10}")
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
            reset_peak()
            started = time.monotonic()
            response = handler_module.handler(
                {"id": f"bench-{size}-{refs}", "input": payload}
            )
            elapsed = time.monotonic() - started
            allocated, reserved = peaks()

            if "error" in response:
                print(f"{size:<12}{refs:>5}   {response['error']['code']}")
                continue

            print(
                f"{size:<12}{refs:>5}{elapsed:>10.1f}{allocated:>10.2f}{reserved:>10.2f}"
            )
            probes.append(
                "    VramProbe(\n"
                f"        width={width},\n"
                f"        height={height},\n"
                f"        refs={refs},\n"
                f'        text_encoder="{encoder}",\n'
                f"        peak_allocated_gb={allocated:.2f},\n"
                f"        peak_reserved_gb={reserved:.2f},\n"
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
