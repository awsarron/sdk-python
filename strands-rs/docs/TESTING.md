# Testing guidelines - Strands Rust SDK

## Test layout

- **Unit tests** are co-located with the code they cover, in a `#[cfg(test)] mod
  tests` block at the bottom of the module file (idiomatic Rust). This is the
  Rust counterpart to the TypeScript SDK's co-located `__tests__/` directories.
- **Integration tests** live in `strands/tests/` and exercise the public API
  (e.g. `tests/agent_loop.rs` drives `Agent::invoke` end to end).
- **Doc tests** in `///` examples are compiled (and run, unless marked
  `no_run`) by `cargo test`.

## Running tests

```bash
cargo test --all-features        # all unit, integration, and doc tests
cargo test -p strands-agents      # just the SDK crate
cargo clippy --all-targets --all-features   # lint (must be warning-free)
cargo fmt --check                 # formatting gate
```

## Conventions

- **Mirror source behaviors, not lines.** Each test that ports a TypeScript test
  carries a comment naming the source `describe`/`it` it corresponds to, so the
  behavior traceability is visible in the code.
- **Mock providers replay scripted events.** `TestModelProvider` (in
  `models/mod.rs` tests) and `MockMessageModel` (in `tests/agent_loop.rs`) stand
  in for the TypeScript `TestModelProvider` / `MockMessageModel` fixtures: they
  emit a fixed sequence of `ModelStreamEvent`s so aggregation and loop behavior
  can be asserted without a real model.
- **Regression tests** for a discovered bug state the behavior they guarantee;
  feature tests carry no issue reference (matching the monorepo evergreen-comment
  rule).
