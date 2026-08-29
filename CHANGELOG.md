# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

The Runpod Hub indexes releases, not commits, so every release here corresponds
to a published Hub build.

## [1.0.0] - 2026-08-29

First release.

### Added

- Text-to-image and multi-reference image editing on FLUX.2 klein 4B, through a
  request schema that exposes generation parameters and no implementation.
- Five Apache-2.0 profiles: `klein-4b` (default, baked into the image),
  `klein-4b-bf16`, `klein-4b-nvfp4`, `klein-4b-base`, `klein-4b-base-bf16`.
- Zero-configuration deployment: an endpoint created with an empty environment
  boots and serves.
- Every model file pinned by repository revision and SHA-256, verified after
  download.
- Reference-image loader with resolve-then-pin DNS handling, per-hop redirect
  validation, blocked private ranges and a streamed byte ceiling.
- Output size follows the first reference when a request has references and no
  explicit dimensions, matching the official ComfyUI edit template.
- Optional S3 output; base64 otherwise, as the primary path.
- Startup hardware checks, including a hard architecture requirement for NVFP4.
- Out-of-memory recovery: restart, then exit so the platform replaces the
  worker.
- `capabilities` operation, so an integrator can discover the endpoint's limits
  without a failed request.
- Runpod Hub manifests, CI covering lint, types, tests, the Docker build, and
  weekly upstream asset drift.
- The worker attaches to Runpod's queue before provisioning models and starting
  ComfyUI, so the first job absorbs the cold start instead of the platform
  killing a runtime that has not finished booting. `capabilities` answers during
  that window; a startup failure is returned as a coded error on every job.
