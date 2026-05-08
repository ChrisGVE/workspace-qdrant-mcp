# Providing the embedding-provider API key to memexd

`memexd` reads the API key for any OpenAI-compatible provider from an
environment variable named by `embedding.api_key_env_var` in the daemon
config (default `OPENAI_API_KEY`). The key is wrapped in
`secrecy::SecretString` inside the process so it never appears in
`Debug` output, tracing spans, or metric labels.

What differs across deployments is *how* that environment variable
gets set. This page documents four opt-in approaches; pick whichever
matches the platform constraints. The daemon needs no changes — it
only reads the env var.

## Option 1 — Env file (recommended for local installs)

Store the key in a 0600-protected env file and let the launcher source
it before exec'ing the daemon. Works on macOS and Linux.

**Setup**

```sh
mkdir -p ~/.config/memexd
cp assets/wrappers/memexd.env.example ~/.config/memexd/env
chmod 600 ~/.config/memexd/env
$EDITOR ~/.config/memexd/env   # set OPENAI_API_KEY=...
```

**macOS (LaunchAgent)** — install the wrapper alongside `memexd` and
point the LaunchAgent at it:

```sh
cp assets/wrappers/memexd-launch.sh ~/.local/bin/memexd-launch
chmod +x ~/.local/bin/memexd-launch
sed "s|__HOME__|$HOME|g" assets/launchd/com.workspace-qdrant.memexd.plist \
    > ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist 2>/dev/null || true
launchctl load   ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
```

The wrapper checks file permissions before sourcing and aborts with
exit 78 (`EX_CONFIG`) if the file is anything other than 0600 / 0400.

**Linux (systemd user service)** — the shipped unit file already
declares `EnvironmentFile=-%h/.config/memexd/env` (the leading dash
makes it optional, so the unit still starts when the file is absent):

```sh
systemctl --user daemon-reload
systemctl --user restart memexd
```

For a system service, place the file at `/etc/memexd/env` (root-owned,
mode 0600); the system unit references that path.

## Option 2 — `launchctl setenv` / `systemctl set-environment`

Inject the key into the service manager's environment without a file.
Convenient for ephemeral testing or rotation; does **not** survive a
reboot on macOS.

**macOS**

```sh
read -s OPENAI_API_KEY && launchctl setenv OPENAI_API_KEY "$OPENAI_API_KEY" && unset OPENAI_API_KEY
launchctl kickstart -k gui/$(id -u)/com.workspace-qdrant.memexd
```

`launchctl getenv OPENAI_API_KEY` will then print the value, so anyone
with shell access on the same user session can read it. Use option 1
when that matters.

**Linux**

```sh
systemctl --user set-environment OPENAI_API_KEY=...
systemctl --user restart memexd
```

To clear it: `systemctl --user unset-environment OPENAI_API_KEY`.

## Option 3 — Container env (Docker / Compose)

The shipped `docker/docker-compose.dev.yml` and
`docker/docker-compose.prod.yml` already declare:

```yaml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY:-}
```

so the value is interpolated from the host shell or a `docker/.env`
file at compose-up time. The daemon container receives it as an env
var; no wrapper script is involved inside the container.

```sh
# preferred — keep the value out of shell history
echo "OPENAI_API_KEY=sk-..." >> docker/.env
chmod 600 docker/.env
docker-compose -f docker/docker-compose.dev.yml up -d
```

Or pass it inline (visible in `ps`):

```sh
OPENAI_API_KEY=sk-... docker-compose -f docker/docker-compose.dev.yml up -d
```

For Kubernetes deployments, mount the value via a `Secret` and
reference it in the pod spec — the daemon does not care which
mechanism delivers the env var.

## Option 4 — Direct shell launch

When `memexd` is started manually from a shell (development, ad-hoc
testing, or one-off reindex runs) the binary inherits the launching
shell's environment via `std::env::var`. No wrapper, plist, unit, or
config file is involved — just export the key in the shell that runs
the daemon:

```sh
export OPENAI_API_KEY=sk-...
~/.local/bin/memexd                # or `cargo run -p memexd -- ...`
```

For a one-off invocation that does not pollute the shell history:

```sh
OPENAI_API_KEY=sk-... ~/.local/bin/memexd
```

This path is convenient during development but does not persist across
reboots and leaves the value visible to anything that can read the
process environment of the launching user.

## Choosing between the four

| Concern | Option 1 (env file) | Option 2 (set-environment) | Option 3 (container env) | Option 4 (shell launch) |
|---|---|---|---|---|
| Survives reboot | yes | macOS: no, Linux: no | yes (compose / k8s state) | no |
| Visible to other shell sessions of same user | no | yes (`launchctl getenv`) | container-scoped | only inside that shell |
| Plaintext at rest | yes (single file, 0600) | no | yes (`docker/.env` or k8s `Secret`) | no (shell only) |
| Suitable for CI / production | yes | no (manual rotation) | yes | no (interactive only) |

The shipped wrapper, plist template, and compose files all default to
"no key set" so users on FastEmbed do not need to configure anything —
the secret-handling path only kicks in when
`embedding.provider = "openai_compatible"`. The same is true for
direct shell launches: with `provider = "fastembed"` the daemon never
reads the API-key env var.
