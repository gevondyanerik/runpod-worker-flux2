# Design notes

Why this worker is shaped the way it is. The README says what it does; this
says what was decided against, and why, so that a future change knows what it
is undoing.

---

## 1. Only Apache-2.0 models

FLUX.2-dev and the whole klein 9B family are released under the FLUX
Non-Commercial Licence. Its §1.c excludes "revenue-generating activity" and any
use with "direct interactions with or that has impact on end users" — which is
most of what anyone deploys an image endpoint for. §3 imposes conditions on
distribution, and baking those weights into a public image *is* distribution.

The alternative would be a licence gate: ship the non-commercial models, make
the operator accept terms, hope the acceptance means something. That gate is
one bug away from distributing weights nobody agreed to, and it would put this
project in the position of making promises on a user's behalf.

Keeping the registry Apache-only removes the question entirely. There is no
gate to get wrong, and anything deployable from here is commercially usable.
The cost is that the strongest FLUX.2 models are unavailable — accepted
deliberately.

## 2. Zero configuration is a supported production setup

The first test in `tests/unit/test_config.py` creates a `Config` from an empty
environment and asserts it works. That is not a convenience feature; it is the
main one.

An endpoint that needs seven variables before it produces a pixel has seven
ways to be misconfigured, and every one of them becomes a support request. So:
the default profile's weights are in the image, every setting has a working
default, and no credential is required on the default path.

`FLUX2_VARIANT` unset falls back. `FLUX2_VARIANT` *wrong* fails startup — a
typo must never silently deploy a different model, and those two cases deserve
opposite treatment.

## 3. One module reads the environment

`app/config.py`, enforced by `scripts/audit_config.py` in CI.

Configuration surfaces grow by accident: someone needs one value in one place,
reads it inline, and two years later nobody can say what a deployment does
without grepping. A test makes the claim checkable rather than aspirational.

The bar for a new variable is in CONTRIBUTING.md: underivable *and*
non-essential to the default path. "worker-comfyui has it" is not a reason —
that project is a general-purpose ComfyUI host and must be configurable.

## 4. The graph is built in code, not loaded from JSON

`app/workflow.py` constructs the ComfyUI API graph programmatically. The
obvious alternative is a static template per profile, exported from the ComfyUI
UI.

It does not work here. The reference chain is variable-length — zero to six
images, each adding five nodes — and ComfyUI's API format has no notion of a
bypassed node: a node is either in the graph or it is not. A static template
would need N pre-made slots and deletion logic, and its node ids churn every
time someone re-saves it from the UI.

Building from named constants gives one source of truth and readable diffs.
The regression risk that a template would have covered is handled by golden
files in `tests/workflows/golden`: the graph is pinned, and a change that
nobody looked at fails CI.

The shape itself is not invented. It is the official
`image_flux2_klein_text_to_image` and `image_flux2_klein_image_edit_4b_*`
templates, flattened out of their subgraphs — including the detail that each
reference conditions *both* the positive and the negative branch, and that an
edit takes its output geometry from `GetImageSize` on the first scaled
reference.

## 5. Requests expose parameters, not implementation

`prompt`, `width`, `height`, `n`, `steps`, `guidance`, `seed`, `output_format`,
`quality`. Not `sampler`, not `scheduler`, not model filenames, not node ids,
not workflow JSON. Unknown fields are rejected rather than ignored.

The line is: things a caller reasonably varies per image are exposed; things
that are properties of the deployment are not. Accepting a sampler name would
make every future change to the graph a breaking API change, for a parameter
whose correct value is a property of the model.

If you want the other trade, `worker-comfyui` exists and is good at it.

## 6. Sampling defaults are confirmed, not guessed

`SamplingDefaults.confirmed` gates `VariantConfig.is_ready`, and the startup
path refuses to serve an unready profile.

The distilled values come from the official ComfyUI templates: 4 steps at cfg
1.0. That is the count the model is trained to converge in, so it is the
model's number rather than a choice.

The base profiles ship 28 steps at cfg 4.0, where the template says 20 at 5.0.
That came out of a grid search — five step counts against four guidance values,
three seeds per cell, scored on whether a text prompt rendered legibly — not
out of taste. 28 was the peak; 50 was tied for worst. This is a deliberate
departure and the only one, which is why `app/variants.py` explains it at the
constant rather than leaving it to be found.

Either way the point stands: a worker with plausible-looking guessed defaults
runs perfectly and produces quietly worse images, which is a much more
expensive failure than not starting.

The same reasoning governs `VramProbe`: only `scripts/benchmark.py` on real
hardware may write one. A fabricated number is worse than a missing one because
it will be trusted.

## 7. Weights are pinned twice

Every built-in asset carries a repository revision *and* a SHA-256, both read
from the Hugging Face API and committed.

Pinning the revision means a profile resolves to the weights it was tested
against, whatever upstream does later. The digest is verified after a fresh
download, and a mismatch deletes the file rather than serving it. Already-
present files are checked by size only — hashing 8 GB on every cold start would
cost more than it protects against, and the revision pin already prevents the
content from changing underneath.

`scripts/check_assets.py` runs weekly and reports a rename, a replaced file, a
changed licence or a newly gated repository, so upstream drift surfaces on a
schedule rather than in front of a paying request.

## 8. The reference loader assumes the URL is hostile

An operator deploys this inside their own network and then lets arbitrary
callers hand it URLs. That is a server-side request forgery machine unless it
is built not to be.

The defence that matters most is resolve-then-pin. Validating a hostname and
then handing the URL to an HTTP client is a DNS-rebinding hole: the client's own
lookup can return a different, private address. So the hostname is resolved
once, *every* returned address is validated, and the connection is made to a
validated IP with the original `Host` header and TLS SNI.

Redirects are followed manually because each hop needs the same treatment; a
public URL that 302s to `169.254.169.254` is the standard cloud-metadata
exfiltration path. The size ceiling is enforced while streaming rather than
from `Content-Length`, which the server controls and can lie about.

These limits live in `app/constants.py` rather than in configuration. They are
security boundaries, and a deployment should not be able to widen them.

## 9. A sick worker exits

On serverless, a worker whose GPU is in a bad state keeps pulling jobs off the
queue and failing them. That is worse than a dead worker, because the platform
replaces a dead one and happily keeps feeding a sick one.

So an out-of-memory failure or a lost ComfyUI first triggers a restart, and if
that does not help — or it happens twice inside ten minutes — the process
exits. Two strikes rather than one: a single OOM can be one oversized request
rather than a broken worker.

## 10. Memory warnings do not block startup

The architecture check for NVFP4 is fatal: on pre-Blackwell hardware the format
may load without delivering the speed or memory win, and running slowly and
silently is worse than refusing.

VRAM and RAM checks only warn. A table of estimates that refused to start would
make this worker impossible to run on hardware it may well handle, and the
operator is better placed to judge than the table is.

## 11. base64 is the primary output path

Not a fallback for people who have not set up S3 yet. Most callers want the
bytes back in the response, and an endpoint that needs object storage
configured before it returns an image has a credential requirement on its
default path.

S3 is opt-in, all three variables together or none, and when it is not
configured the worker never mentions it. A feature you have not enabled should
be invisible, not a reproach.

The 5 MB inline ceiling exists so an oversized response fails with a message
that says what to do, instead of being silently rejected by the platform.

## 12. Everything testable runs on a laptop

The full suite is about 195 tests in roughly a second, with no GPU and no
network. `tests/fake_comfy.py` implements the ComfyUI client's surface in
memory with failure injection, so the recovery paths — OOM, a dead process, a
rejected graph — are tested rather than hoped for.

A test suite that needs a 24 GB card is a test suite nobody runs.

## 13. The worker registers with Runpod before it boots

`main()` calls `runpod.serverless.start()` almost immediately and provisions
models and starts ComfyUI on a background thread. Jobs then block until that
thread is done, so the first request absorbs the cold start rather than the
startup absorbing it.

This is backwards from how it reads, and it is not a preference. Runpod gives a
worker a short window to attach to its queue — measured at about three minutes
on a Hub test pod — and a 12.5 GB profile plus a ComfyUI start does not reliably
fit inside it. Booting first meant the platform killed the runtime with
`prepare AI API: context deadline exceeded` before a single job was served, with
nothing in the log to explain it, because the worker never got far enough to
log. Runpod's own `worker-comfyui` has the same shape: ComfyUI goes to the
background in `start.sh` and the handler waits on it per job.

The cost is that a startup failure can no longer crash the process, so it is
recorded and returned as a coded error on every job instead. That is the better
failure anyway: `UNSUPPORTED_GPU_ARCH` in a job response says what is wrong,
where an unresponsive runtime says nothing.

`capabilities` is the exception — it is answered from configuration alone, so it
works while the boot is still running.
