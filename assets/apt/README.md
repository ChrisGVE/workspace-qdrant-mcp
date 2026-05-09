# APT repository — workspace-qdrant-mcp

This directory documents how to publish the `.deb` packages produced by
the release workflow as a signed APT repository hosted on GitHub Pages,
and how end users install from it.

The packaging itself is wired up — `cargo deb -p wqm-cli` builds a
`.deb` from `[package.metadata.deb]` in `src/rust/cli/Cargo.toml`, and
the release workflow attaches one `amd64` and one `arm64` package to
each GitHub release. The repository signing/publishing pieces below
need to be set up once by the maintainer.

## What ships

Each release builds two packages:

```
workspace-qdrant-mcp_<version>_amd64.deb
workspace-qdrant-mcp_<version>_arm64.deb
```

Each contains:

```
/usr/bin/wqm
/usr/bin/memexd
/lib/systemd/system/memexd.service
/usr/share/doc/wqm-cli/{copyright, changelog.gz, README}
```

Maintainer scripts (`postinst`, `prerm`, `postrm`) only call
`systemctl daemon-reload` and stop the system memexd unit on
upgrade/remove. They never enable, start, or remove user data.

## One-time maintainer setup

### 1. Create the GPG signing key

```bash
gpg --full-generate-key
# - Key type: RSA and RSA
# - Key size: 4096
# - Expiry: 2y (or longer)
# - Real name: ChrisGVE APT Repository
# - Email: christian.c.berclaz@gmail.com
```

Export the public key to commit into the repo:

```bash
gpg --armor --export <key-id> > assets/apt/pubkey.asc
```

Export the private key for use as a GitHub Actions secret:

```bash
gpg --armor --export-secret-keys <key-id> > /tmp/apt-signing.key
# Add the contents of /tmp/apt-signing.key as repo secret APT_GPG_KEY.
# Add the passphrase as APT_GPG_PASSPHRASE (leave empty if no passphrase).
```

### 2. Set up the GitHub Pages APT repository

Create `ChrisGVE/workspace-qdrant-apt` (or any other name). Enable
GitHub Pages from `main` branch / root.

The hosting layout follows the standard flat layout:

```
.
├── pubkey.asc                  # public GPG key
├── dists/
│   └── stable/
│       ├── Release             # signed metadata
│       ├── Release.gpg         # detached signature
│       ├── InRelease           # clear-signed metadata
│       └── main/
│           ├── binary-amd64/
│           │   ├── Packages
│           │   ├── Packages.gz
│           │   └── *.deb
│           └── binary-arm64/
│               └── ...
└── pool/
    └── main/
        └── workspace-qdrant-mcp/
            └── *.deb
```

### 3. Wire the publish job (future task)

The release workflow uploads the `.deb` files to the GitHub Release.
A separate workflow (not yet wired — out of scope for the current
pending-cleanup round) should:

1. Download the `.deb` artefacts from the release.
2. Move them into `pool/main/workspace-qdrant-mcp/`.
3. Regenerate `dists/stable/main/binary-{amd64,arch64}/Packages{,.gz}`
   via `apt-ftparchive packages`.
4. Sign `Release`/`InRelease` via the GPG key from secrets.
5. Push to the GH-Pages-backed repo.

`reprepro` is the conventional helper, but `apt-ftparchive` is fine
for a single-distribution single-component repo.

## End-user install

```bash
# Install the public key.
curl -fsSL https://chrisgve.github.io/workspace-qdrant-apt/pubkey.asc \
  | sudo gpg --dearmor -o /etc/apt/keyrings/workspace-qdrant.gpg

# Add the repository.
echo "deb [signed-by=/etc/apt/keyrings/workspace-qdrant.gpg] \
https://chrisgve.github.io/workspace-qdrant-apt stable main" \
  | sudo tee /etc/apt/sources.list.d/workspace-qdrant.list

sudo apt update
sudo apt install workspace-qdrant-mcp
```

After install, enable the daemon (per-user is the default supported
path):

```bash
systemctl --user enable --now memexd
```

System-wide install requires placing the unit file under
`/etc/systemd/system/` instead of the user path; that is documented
in `docs/specs/13-deployment.md`.

## Known limits

- The repository hosts only a single distribution component
  (`stable`/`main`). Beta/rc/alpha channels for self-update are kept
  separate and use direct GitHub release downloads, not APT.
- The repository expects a 4-year key rotation cadence. Schedule a
  follow-up task at the rotation date.
- arm64 packages currently target glibc 2.35+ (Ubuntu 22.04). Older
  distributions are out of scope.
