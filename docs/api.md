# API reference

The Runpod endpoint takes `{"input": {...}}` and returns either a result object
or an `error` object. This page is the complete contract; the README has the
short version.

`api_version` is `"1"`. A breaking change to the shapes below increments it.

---

## Generate

### Input

| Field | Type | Default | Constraints |
|---|---|---|---|
| `prompt` | string | — | **required**, 1–8000 characters after trimming |
| `images` | string[] | `[]` | `http(s)://` URLs or `data:image/*;base64,` URIs |
| `width` | integer | `DEFAULT_WIDTH` (1024) | 256–4096 |
| `height` | integer | `DEFAULT_HEIGHT` (1024) | 256–4096 |
| `n` | integer | 1 | 1 – `MAX_IMAGES_PER_REQUEST` |
| `steps` | integer | profile default | 1–60 |
| `guidance` | number | profile default | 0.0–20.0 |
| `seed` | integer | random | 0 – 2⁶³−1 |
| `output_format` | string | `DEFAULT_OUTPUT_FORMAT` | `webp`, `png`, `jpeg` |
| `quality` | integer | `DEFAULT_OUTPUT_QUALITY` | 1–100 |

Unknown fields are **rejected**, not ignored. `width × height` must not exceed
`MAX_PIXELS`, and both are rounded *down* to a multiple of 16 — never stretched,
because the aspect ratio you asked for is not the server's to change.

### Output

```json
{
  "images": [
    {
      "b64": "UklGRt4...",
      "mime_type": "image/webp",
      "seed": 42,
      "index": 0,
      "width": 1024,
      "height": 1024
    }
  ],
  "variant": "klein-4b",
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "guidance": 1.0,
  "reference_count": 0,
  "timings_ms": {
    "download": 0,
    "inference": 2140,
    "upload": 31,
    "total": 2180
  },
  "api_version": "1"
}
```

| Field | Notes |
|---|---|
| `images[].b64` | Present unless S3 output is configured |
| `images[].url` | Present **instead of** `b64` when S3 is configured |
| `width` / `height` | The dimensions of the image returned, not the request |
| `adjusted` | Present and `true` when those differ from what you asked for |
| `references` | Present when the request had reference images (see below) |
| `timings_ms` | Wall-clock milliseconds per phase |

### Reference reporting

```json
"references": [
  {"index": 0, "source_px": [2400, 1600], "used_px": [1254, 836], "downscaled": true}
]
```

Each reference is capped at `REF_MAX_PIXELS` before encoding, because
references enter the transformer context and their resolution drives both
memory and latency. When one is resized the response says so — a silent resize
is as dishonest as a silent drop.

### Output geometry with references

- **No `width`/`height` given.** The output takes its size from the first
  reference, after that reference has been scaled to fit `REF_MAX_PIXELS`. An
  edit of a 3:4 photo comes back 3:4. `adjusted` will be `true`.
- **`width` or `height` given.** Those win, aligned down to a multiple of 16.

### Seeds and batches

One seed covers the whole batch, and every returned image carries it. ComfyUI
draws a batch of noise from a single generator, so image *i* of a batch of *n*
is **not** the image you get from that seed with `n=1`. The reproducible unit
is `(seed, n, index)`.

---

## Capabilities

```json
{"input": {"op": "capabilities"}}
```

Runs without touching the GPU. Because an operator configures almost nothing,
this is how an integrator discovers an endpoint's limits without a failed
request.

```json
{
  "api_version": "1",
  "variant": "klein-4b",
  "description": "Distilled 4B, fp8 — the default. Best speed/size balance.",
  "license": "apache-2.0",
  "distilled": true,
  "precision": "fp8",
  "text_encoder": "bf16",
  "default_steps": 4,
  "default_guidance": 1.0,
  "default_width": 1024,
  "default_height": 1024,
  "max_reference_images": 6,
  "ref_max_pixels": 1048576,
  "max_output_pixels": 4194304,
  "max_images_per_request": 4,
  "max_prompt_chars": 8000,
  "output": "base64",
  "gpu": "NVIDIA GeForce RTX 4090",
  "measured_limits": []
}
```

`measured_limits` carries VRAM probes measured on real hardware, when any have
been recorded for the active profile. It is empty rather than estimated: a
fabricated number would be trusted.

---

## Errors

```json
{
  "error": {
    "code": "INVALID_RESOLUTION",
    "message": "4096x4096 is 16777216 pixels, above the 4194304 limit.",
    "retryable": false,
    "details": {"max_pixels": 4194304}
  }
}
```

`details` is present only when there is something machine-readable to say.
`retryable` is on the wire because it is not inferable from the code alone.

### Request errors — do not retry

| Code | Meaning |
|---|---|
| `MISSING_PROMPT` | `prompt` absent, blank, or not a string |
| `INVALID_INPUT` | A field is malformed, out of range, or unknown |
| `INVALID_RESOLUTION` | Outside 256–4096, or over `MAX_PIXELS` |
| `TOO_MANY_IMAGES` | More references than this endpoint accepts |

### Reference errors

| Code | Retryable | Meaning |
|---|---|---|
| `INVALID_IMAGE_URL` | no | Bad scheme, unresolvable host, non-public address, too many redirects |
| `IMAGE_DOWNLOAD_FAILED` | **yes** | The remote server failed or timed out |
| `IMAGE_TOO_LARGE` | no | One image over 20 MB |
| `TOTAL_INPUT_TOO_LARGE` | no | References over 60 MB in total |
| `INVALID_IMAGE` | no | Not a decodable JPEG, PNG or WebP |

### Deployment errors

These mean the endpoint is misconfigured; they will not fix themselves.

| Code | Meaning |
|---|---|
| `UNSUPPORTED_VARIANT` | `FLUX2_VARIANT` is not a known profile |
| `UNSUPPORTED_GPU_ARCH` | The profile needs a newer GPU (NVFP4 needs Blackwell) |
| `MODEL_ASSET_MISSING` | A weight file is absent, or ComfyUI cannot see it |
| `MODEL_CHECKSUM_MISMATCH` | A download did not match its pinned digest |
| `MODEL_AUTH_REQUIRED` | An override names a gated repository and `HF_TOKEN` is unset |
| `PROFILE_NOT_READY` | The profile's sampling defaults are unconfirmed |
| `COMFYUI_START_FAILED` | ComfyUI did not become healthy |
| `WORKFLOW_INVALID` | The generated graph was rejected — please file a bug |

### Inference and output errors

| Code | Retryable | Meaning |
|---|---|---|
| `CUDA_OUT_OF_MEMORY` | **yes** | Try fewer references or a smaller size |
| `INFERENCE_FAILED` | **yes** | Execution failed inside ComfyUI |
| `INFERENCE_TIMEOUT` | **yes** | Over the 600 s per-job limit |
| `OUTPUT_TOO_LARGE` | no | Over the 5 MB inline limit; use S3 or ask for less |
| `OUTPUT_UPLOAD_FAILED` | **yes** | The S3 upload failed |

After `CUDA_OUT_OF_MEMORY` or `INFERENCE_TIMEOUT` the worker restarts ComfyUI,
and exits if that does not help. A retry may therefore land on a fresh worker
and pay a cold start.

---

## Examples

Text to image:

```json
{"input": {"prompt": "a red bicycle leaning against a white wall"}}
```

Edit one reference, keeping its aspect ratio:

```json
{
  "input": {
    "prompt": "change the bag colour to deep blue, keep everything else",
    "images": ["https://example.com/bag.jpg"]
  }
}
```

Combine two references — order is semantic:

```json
{
  "input": {
    "prompt": "put the logo from image 2 onto the bag in image 1",
    "images": ["https://example.com/bag.jpg", "https://example.com/logo.png"],
    "width": 1024,
    "height": 1024
  }
}
```

A reproducible batch:

```json
{"input": {"prompt": "a still life with lemons", "n": 4, "seed": 12345}}
```
