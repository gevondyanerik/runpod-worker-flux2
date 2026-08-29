# Contributing

Thanks for looking. This worker is deliberately opinionated, so the most useful
thing to know before opening a pull request is which way each argument has
already been settled.

## The three rules

**1. `app/config.py` is the only module that reads the environment.**
CI enforces it (`scripts/audit_config.py`). Everything else takes a `Config`.

**2. A new environment variable needs both of these to be true:**

- it cannot be determined automatically or defaulted sensibly, **and**
- the default path still works without it.

"worker-comfyui exposes it" is not a reason. That project is a general-purpose
ComfyUI host and must be configurable; this one is a specialised FLUX.2 worker
and must not be. `app/constants.py` records why each non-configurable value is
non-configurable — read it before proposing to move something out.

**3. Requests expose generation parameters, never implementation.**
`steps` and `guidance` are things a caller reasonably varies per image.
`sampler`, `scheduler`, model filenames, node ids and workflow JSON are
properties of the deployment. Adding one of those to the schema turns every
future change to the graph into a breaking API change.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Everything except the Docker build runs without a GPU or network access.

## Before opening a pull request

```bash
ruff check . && ruff format .
mypy app bootstrap
pytest tests -q
python scripts/audit_config.py
python scripts/audit_env.py
```

If you changed the graph, regenerate the golden files deliberately and include
the diff in your pull request:

```bash
python scripts/update_goldens.py
```

CI fails on a stale golden file, because a graph change nobody looked at is
exactly what they exist to catch.

If you added or removed a variable, update **both** `.env.example` and
`.runpod/hub.json`; `scripts/audit_env.py` checks that they still agree with
`app/config.py`.

## Adding a model profile

Profiles live in `app/variants.py` and nowhere else.

1. The weights must be **Apache-2.0 and public**. FLUX.2-dev and the klein 9B
   family are non-commercial and are out of scope on purpose — see the
   Licensing section of the README.
2. Pin the revision and the SHA-256 from the Hugging Face API. Do not use
   `main`.
3. Sampling defaults must come from the official ComfyUI template for that
   model, not from a guess. A profile with unconfirmed defaults fails
   `is_ready` and will not be served — that check exists because guessed
   numbers produce a worker that runs and is quietly wrong.
4. Add it to the `FLUX2_VARIANT` options in `.runpod/hub.json` and to the table
   in the README.

## VRAM probes

`VramProbe` entries may only be produced by `scripts/benchmark.py` running on
real hardware, and must be committed with the GPU name and the date. A
hand-written probe is worse than no probe, because it will be believed.

## Style

Match the surrounding code. Comments explain *why*, not *what* — if a line
needs a comment to say what it does, rewrite the line instead.
