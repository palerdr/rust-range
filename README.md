# RangeForge

RangeForge is a safe Rust engine for maintaining a probability distribution over an opponent's possible two-card Texas Hold'em hands, conditioning it on blockers, updating it from observed actions, and reporting equity and information metrics.

The project is intentionally narrow: heads-up Hold'em, one opponent, a fixed 1,326-combo hidden-hand universe, and an explicit illustrative action model. It is an inference and learning project, not a poker bot or wagering tool.

## What works

- Canonical cards, masks, hole cards, boards, streets, and known-state validation.
- Deterministic `ComboId` indexing for all `C(52, 2) = 1,326` unordered hands.
- MVP range notation: `random`, exact hands such as `AsKd`, pairs such as `QQ`, suited/offsuit classes such as `AKs` and `AKo`, combined `AK`, and positive weights such as `AKs:0.35`.
- Blocker conditioning against hero and public cards.
- Made-hand and draw features mapped to six deterministic model buckets.
- Validated JSON action models with 36 required postflop rows.
- Sequential Bayesian action updates using log-sum-exp normalization.
- Exact weighted equity on flop, turn, and river.
- Seeded Monte Carlo equity, including preflop support.
- Entropy, effective hand count, total variation, KL divergence, top combinations, bucket masses, action information, and next-card information.
- A thin `rangeforge` CLI with human and JSON reports.

## Architecture

```text
scenario JSON + model JSON
            |
            v
rf-cli  ->  rf-engine  ->  rf-core
report      beliefs,     cards, boards,
            equity,      combos, evaluator
            information
```

`rf-core` owns objective poker state and the fixed combo universe. `rf-engine` owns probability, inference, equity, and reports. `rf-cli` parses files and presents one engine report; it does not contain poker math.

## Mathematical contract

For hidden hand `H`, known cards `B`, and observations `E1..En`:

```text
P(H=h | B, E1..En) ∝ P(H=h | B) × ∏ P(Ei | H=h, contexti)
```

The implementation applies action likelihoods in log space and normalizes with log-sum-exp. A blocked combo always has zero posterior mass. Equity is showdown share only:

```text
equity = P(win) + 0.5 × P(tie)
```

The bundled model is synthetic and intentionally marked `"calibration": "illustrative"`. Its results demonstrate the pipeline and math; they are not claims about real-player behavior.

## Quick start

Install stable Rust with Cargo, then from the repository root:

```bash
cargo test --workspace
cargo run --package rf-cli --bin rangeforge -- --help
```

Validate the checked-in model and a scenario:

```bash
cargo run --package rf-cli --bin rangeforge -- validate-model examples/toy_postflop_v1.json
cargo run --package rf-cli --bin rangeforge -- validate examples/flop_large_bet.json --model examples/toy_postflop_v1.json
```

Run the main demo as human-readable text:

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/flop_large_bet.json --model examples/toy_postflop_v1.json
```

Run the same analysis as JSON:

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/flop_large_bet.json --model examples/toy_postflop_v1.json --format json --top 10
```

Other checked-in scenarios:

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/turn_exact.json --model examples/toy_postflop_v1.json
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/preflop_monte_carlo.json --model examples/toy_postflop_v1.json
```

Exact equity is deliberately rejected preflop; use `"method": "monte_carlo"` there.

## Scenario shape

```json
{
  "hero": "AsKh",
  "board": "QsJh2c",
  "prior": {"type": "notation", "value": "random"},
  "observations": [
    {"board": "QsJh2c", "decision": "unopened", "action": "bet_large"}
  ],
  "equity": {"method": "exact", "samples": 100000, "seed": 7}
}
```

Boards accept compact cards (`QsJh2c`) or whitespace-separated cards. Observation boards must be chronological prefixes of the current board. Legal actions are `check`, `bet_small`, `bet_large` when unopened and `fold`, `call`, `raise` when facing a bet.

## Testing and verification

Use the following commands for the normal quality gate:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

The ordinary suite covers invalid card state, exhaustive combo-ID round trips, range expansion, evaluator fixtures, distribution invariants, hand-calculated Bayes updates, model validation, exact/Monte Carlo equity, information identities, scenario reports, and CLI smoke tests. The exhaustive 2,598,960-five-card category-count test is intentionally ignored because it is a validation/performance job:

```bash
cargo test -p rf-core exhaustive_five_card_category_counts_match_reference -- --ignored
```

Performance claims should be measured in release mode. The evaluator has an ignored release throughput test:

```bash
cargo test --release -p rf-core perf_1_000_000_seven_card_evaluations -- --ignored --nocapture
```

## Workspace layout

```text
crates/rf-core/    cards, boards, combos, evaluator, features
crates/rf-engine/  distributions, metrics, actions, Bayes, equity, information, reports
crates/rf-cli/     rangeforge binary and CLI integration tests
examples/          checked-in scenario and action-model JSON
docs/              product requirements and first-principles design notes
```

## Deliberate limitations

Version 0.1 does not implement GTO solving, betting-tree optimization, multiway pots, stack/rake/side-pot dynamics, player calibration, live integrations, a web UI, or unsafe/SIMD/GPU acceleration. The action model omits position, pot size, stack depth, exact bet size, and player identity. Equity does not model future betting or fold equity.

The next responsible extension is model calibration against held-out data, followed by a benchmark-guided optimization of exact flop enumeration.
