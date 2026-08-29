# Security

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/gevondyanerik/runpod-worker-flux2/security/advisories/new).
Please do not open a public issue for anything exploitable.

Include what you can reproduce: the request, the deployment shape, and what you
observed. You will get an acknowledgement within a few days.

## What this worker treats as hostile

A deployed endpoint accepts prompts and image URLs from callers, and runs inside
the operator's own network. The reference-image loader is therefore written on
the assumption that every URL is an attack:

- **Scheme allowlist.** Only `http`, `https` and `data:` URIs are accepted.
- **Resolve-then-pin.** A hostname is resolved once, *every* returned address is
  validated, and the connection is made to a validated IP with the original
  `Host` header and TLS SNI. Validating a hostname and then letting the HTTP
  client resolve it again is a DNS-rebinding hole.
- **Per-hop redirect validation.** Redirects are followed manually and each hop
  is fully re-validated, DNS included. A public URL that redirects to
  `169.254.169.254` is the classic cloud-metadata exfiltration path.
- **Blocked ranges.** Private, loopback, link-local, carrier-grade NAT,
  multicast and reserved ranges, in both IPv4 and IPv6, with IPv4-mapped IPv6
  unwrapped so `::ffff:127.0.0.1` is judged as loopback.
- **Streamed byte ceiling.** Enforced while reading, not from `Content-Length`,
  which the server controls and can lie about.
- **Total budget.** 20 MB per image, 60 MB per request, so many
  just-under-the-limit references cannot add up to a memory problem.

These limits are constants, not settings: they are security boundaries, and a
deployment should not be able to widen them by accident.

## Other properties worth knowing

- **No credentials on the default path.** Every model is public and
  Apache-2.0. `HF_TOKEN` is read only when an expert override names a gated
  repository.
- **Prompts are never logged.** They are user content, and there is no good
  default other than "never", so there is no switch to get wrong.
- **Errors carry no stack traces.** Callers get a stable code and a readable
  message; tracebacks stay in the worker's own logs.
- **Weights are pinned and verified.** Every built-in asset is pinned by
  repository revision and SHA-256, and a fresh download that does not match is
  deleted rather than served.
- **ComfyUI is not reachable.** It listens on loopback inside the container and
  is never exposed.

## Scope

Expert overrides (`DIFFUSION_MODEL_REPO` and friends) load whatever you point
them at. That is the operator's decision and their trust boundary, not a
vulnerability in this worker.
