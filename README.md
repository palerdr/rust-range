# rust-range

rust-range is a Rust command-line tool and library for analyzing a heads-up Texas Hold'em opponent range. It starts with the 1,326 possible two-card hands, removes hands blocked by known cards, updates the remaining probabilities from observed postflop actions, and reports showdown equity and information metrics.

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

## Run the demos

Install stable Rust with Cargo, then from the repository root:

```bash
cargo test --workspace
cargo run --package rf-cli --bin rangeforge -- --help
```

All examples use the bundled illustrative action model:

```bash
cargo run --package rf-cli --bin rangeforge -- validate-model examples/toy_postflop_v1.json
```

Run a demo as a human-readable report:

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/sequential_action_update.json --model examples/toy_postflop_v1.json
```

Use `--format json` for machine-readable output:

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/sequential_action_update.json --model examples/toy_postflop_v1.json --format json --top 10
```

### What the demos show

Sequential action inference applies a flop large bet and a turn raise to the same opponent range. The report shows the probability of each observation, how entropy and effective hand count change after every update, the final posterior's most likely hands, and equity against that posterior.

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/sequential_action_update.json --model examples/toy_postflop_v1.json
```

The blocker comparison holds the board and observed action fixed while changing Hero from `AsKh` to `AcKh`. This shows how holding the ace of spades removes opponent spade combinations and changes the inferred range and equity.

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/blocker_with_ace_of_spades.json --model examples/toy_postflop_v1.json
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/blocker_without_ace_of_spades.json --model examples/toy_postflop_v1.json
```

The draw-heavy board example (`JsTs7h`) first incorporates a large bet, then reports exact equity and expected next-card information across every legal turn card. It shows how much a future card is expected to reveal about the opponent's range.

```bash
cargo run --release --package rf-cli --bin rangeforge -- analyze examples/draw_heavy_next_card.json --model examples/toy_postflop_v1.json
```

The original simple examples remain available in `examples/`, including an exact turn calculation and seeded preflop Monte Carlo estimate. Exact equity is unavailable preflop; use `"method": "monte_carlo"` there.

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
