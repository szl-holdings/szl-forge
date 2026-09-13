# Security boundary and negative-space contract

This first slice is an authenticated loopback research application, not a hardened
multi-tenant cloud service. Its operator owns the filesystem and process account.
No hostile-same-user sandbox, remote execution agent, signature authority or
production readiness is claimed.

## Controls implemented

The app requires a local random password of at least 32 characters. UI and data
APIs use HTTP Basic only over a loopback bind; /healthz exposes only static service
liveness. The CLI offers no public-bind switch. Hostnames are allowlisted, proxy
headers are disabled, responses have no-store/no-referrer/CSP headers, and
cross-origin POSTs are refused. POST/PUT/PATCH bodies are bounded to 16KiB before
application parsing. HTML is escaped by Jinja; there is no client script and no
raw provider response rendering.

Only two fixed upstream GET paths exist: /api/tags and /api/ps. An operator-configured
literal loopback or tailnet IPv4 plus port 11434 is mandatory. Configuration is
limited to eight named endpoints. No DNS resolution, redirects, public IPs, proxy
inheritance or outbound authentication is allowed. Responses are bounded in time,
size and inventory count; compressed bodies are rejected. Errors are sanitized.
These are inventory observations, not cryptographic node qualification.

Artifact paths come only from operator configuration, not browser requests.
The loader refuses symlink/reparse paths, bounds regular-file reads, hashes the
exact bytes read, accepts a fixed file set, validates tensor names/shapes/dtype and
finiteness, and uses safetensors bytes only. The manifest is not signed. A malicious
actor who can replace both artifact and manifest is outside its authenticity
boundary. Do not import or load quarantined joblib, pickle or remote Python.

Train/evaluation commands are local and explicitly acknowledged. There are no
web train, dispatch, shell, upload, publish or merge routes. No OpenRouter API calls,
GPU scheduling, Docker socket mounts, Tailscale policy writes, Cloudflare changes,
key rotation, paid jobs, HF writes, or GitHub writes occur in the workbench or
training CLI. The separate blueprints module can publish code only through an
explicit protected-main workflow using the existing Forge credential selector.
It permits two exact new model identities, refuses non-blueprint contents, uses
a conditional parent commit, and verifies immutable bytes after each publication.
Request-start journals preserve mutation uncertainty; no automatic retry occurs.
This is not the trained-weight publisher and cannot authorize a model release.

## Remaining controls before an operational service

Integrate the existing signed source/release evidence; never convert an unsigned
local checksum into a verified pool receipt. Enforce current source/policy/model
identity at the canonical router, not through this app's labels. Add calibration,
OOD/abstention, resource limits and owner lifecycle controls under their existing
authorities. A client-supplied `verified=true` is never sufficient evidence.

A reverse proxy or Tailscale Serve integration must receive separate TLS, Host,
Origin and identity-header testing; do not allow '*' hosts/origins as a workaround.
MagicDNS, IPv6 and TLS private endpoint adapters are not shipped in this version.
Do not solve that limitation by widening an arbitrary-URL input.

Publishable build artifacts need platform-specific dependency hashes, trusted
base-image digests if containerized, SBOM/scan results, native GitHub status checks
and exact deployed-byte readback. The development dependency ranges and local
CPU test environment are not that release closure. Docker/Windows/CUDA checks
were not run for this package. Existing owner environments remain unchanged.

Keep raw prompts, secrets, local identities, keys, private tailnet addresses and
training examples out of public evidence. The app does not log prompt bodies;
operator/server tooling outside this process must follow the same policy.
