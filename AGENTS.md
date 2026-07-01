# AGENTS.md

These instructions apply to the entire RangeForge workspace.

## Project Shape

RangeForge is a Rust workspace with three crates:

- `crates/rf-core`: poker domain objects and currently implemented core poker logic.
- `crates/rf-engine`: probability distributions, metrics, action/evidence models, inference, and equity-style engine logic.
- `crates/rf-cli`: command-line entry point and presentation layer.

The guiding product idea is simple: keep a probability table over the opponent's possible two-card hands, remove impossible hands with blockers, update weights from observed actions, then report equity and information metrics.

## Design Rules

- Prefer the smallest implementation that satisfies the current milestone.
- Do not create a new module, enum, trait, or error type just because there is a conceptual seam.
- Keep related beginner-facing poker objects together unless there is a strong practical reason to split them.
- Preserve existing public names, aliases, and file layout unless the user explicitly asks for a rename or migration.
- If current code and older docs disagree, preserve the current code layout and mention the mismatch before moving code.
- Use one clear module docstring at the top of a file. Avoid doc-commenting every obvious getter or tiny helper.
- Helper functions used once should usually live near the calling code or stay inline unless extracting them clearly improves readability, testing, or hot-loop performance.
- Keep parser/model behavior explicit. Reject invalid or ambiguous input instead of silently guessing defaults.

## Crate Boundaries

`rf-core` should stay focused on objective poker state:

- Cards, card masks, hole cards, boards, streets, and known state.
- Combo identifiers, combo weights, and range notation expansion.
- Current evaluator/features modules are in this crate now; do not move them without an explicit request.

`rf-engine` should own calculations over those objects:

- `RangeDistribution` construction, normalization, conditioning, and iteration.
- Entropy, effective hand count, total variation, KL divergence, top-N summaries, and bucket aggregation.
- Action observations, legal action sets, validated bucketed action likelihood models, and Bayesian update logic.

`rf-cli` should own user interaction:

- Parse scenario/config input.
- Call core and engine APIs.
- Print deterministic human-readable or JSON reports.

## Rust Style

- Use safe Rust only unless the user explicitly asks otherwise.
- Use fixed-size arrays and bitsets for hot poker loops where that keeps the code clear.
- Keep hot paths allocation-free when practical, especially evaluator, combo iteration, and distribution loops.
- Use typed errors for public fallible APIs. Keep error variants meaningful but not excessive.
- Prefer deterministic ordering for output, tests, and summaries.
- Use `BTreeMap` when serialized or displayed order matters; use arrays when the universe is small and fixed.

## Testing

- Run only the tests or checks relevant to the requested change unless the user asks for a full suite.
- Before claiming a behavior is fixed, run the narrow test that proves it when feasible.
- For workspace-wide changes, use `cargo test --workspace`.
- For quick compiler feedback, use `cargo check --workspace`.
- Keep exhaustive or expensive tests ignored unless the user explicitly asks to run them.
- Performance claims must come from release-mode measurements.

## Common Commands

```bash
cargo check --workspace
cargo test --workspace
cargo fmt --all
cargo clippy --workspace --all-targets
cargo build --release --workspace
```

## Documentation Sources

Consult these docs when the task asks for product or architecture intent:

- `docs/rf.docx`: full product requirements and milestone plan.
- `docs/rf-fp.docx`: simplified first-principles explanation and anti-scope-creep guidance.
- `README.md`: fresh-machine setup and workspace commands.

The first-principles doc is the preferred guide for tone and scope: explain why an object exists before adding it, keep the module count low, and make the project understandable to a Rust beginner.

## Change Safety

- Do not revert user work unless explicitly asked.
- Do not rename functions, aliases, files, or modules as part of a test or cleanup change.
- Do not move code across crates unless the request is specifically about architecture or migration.
- If a requested implementation would require broad rewrites, first state the narrow version that satisfies the immediate milestone.
