RangeForge
=========

RangeForge is a Rust workspace with three crates:

- `crates/rf-core`
- `crates/rf-engine`
- `crates/rf-cli`

This repository uses a **single top-level git repository** for all workspace crates.

Fresh machine setup (Windows, macOS, Linux)
-------------------------------------------

Install the required tooling
1. Install [Git](https://git-scm.com/downloads).
2. Install [Rust and Cargo](https://rustup.rs/) (recommended, same idea as `uv` for Python):
   ```bash
   # Windows (PowerShell)
   winget install Rustlang.Rustup
   # or use rustup installer from rust-lang.org
   ```
   Then verify:
   ```bash
   git --version
   rustc --version
   cargo --version
   ```

Optional but recommended:
```bash
rustup update
rustup component add clippy rustfmt
```

Clone and build
--------------

```bash
git clone <your-github-remote-url>
cd <repo-dir>
```

Build everything in the workspace:
```bash
cargo build --workspace
```

Run tests:
```bash
cargo test --workspace
```

Run the CLI:
```bash
cargo run --package rf-cli -- --help
```

Common maintenance commands
---------------------------
- `cargo check --workspace`
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets`
- `cargo build --release --workspace`

Working on branch and pushing
-----------------------------
```bash
git status
git add -A
git commit -m "Your commit message"
git remote add origin <your-github-remote-url>
git push -u origin <branch-name>
```

Notes
-----
- Build artifacts are ignored (`/target/`).
- Local project-level metadata from this repo (VS, agents) is ignored.

If you have an existing checkout from before this migration, remove any old nested
crate-level git metadata (`.git` directories under `crates/*/`) so this workspace
is treated as one repository.
