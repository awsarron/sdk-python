# Agent Development Guide - Rust SDK

Guidance for AI agents working on the Strands Rust SDK (`strands-rs/`).

> **Cross-SDK rules live in the [root AGENTS.md](../AGENTS.md).** Plugin naming,
> cross-SDK parity, public/internal API marking, the structured-logging format,
> and the evergreen-comment rule apply to all SDKs and are stated once there.
> This file shows the Rust-idiomatic form and the rules unique to Rust.

## Product overview

The Rust SDK is the Rust port of the Strands Agents framework, derived from the
canonical TypeScript SDK. It mirrors the TypeScript SDK in concepts and names
(re-cased to Rust idiom) while being idiomatic Rust. It is currently a
**foundational vertical slice** — see [README.md](README.md#scope) for what is
and isn't ported.

## Workspace layout

```
strands-rs/
├── strands/          # the `strands-agents` crate
│   └── src/{types,models,tools,agent}/, errors.rs, lib.rs
├── strands-macros/   # the `#[tool]` procedural macro (`strands-agents-macros`)
└── docs/             # PORTING.md, TESTING.md
```

## Porting from TypeScript

The TypeScript SDK (`../strands-ts`) is the canonical source. When porting a
feature, read [docs/PORTING.md](docs/PORTING.md) for the construct mapping, then
the source file and its tests. The source tests are the behavioral spec — every
ported behavior gets a Rust test that names the source `it`/`describe` it mirrors.

## Coding conventions

- **Format with `rustfmt`** (`cargo fmt`); the default profile is the gate.
- **Lint clean with clippy** (`cargo clippy --all-targets --all-features` must be
  warning-free).
- **Errors:** one `#[non_exhaustive] enum StrandsError` (thiserror). Providers
  translate vendor errors into typed variants (`ModelThrottled`,
  `ContextWindowOverflow`) and preserve the cause via `#[source]` — never let a
  raw vendor error escape a provider boundary.
- **Async:** the SDK is async-only (Tokio). Model providers implement
  `#[async_trait] trait Model`; `stream` returns a boxed `Stream` built with
  `async-stream`.
- **Public vs internal API:** the public surface is what `lib.rs` re-exports plus
  `pub` items in public modules. Keep genuinely internal items non-`pub` or under
  a `#[doc(hidden)]` module (e.g. `reexport`, which exists only for macro-
  generated code).
- **Structured logging** uses `tracing` with fields, following the cross-SDK
  format in the root AGENTS.md.
- **Evergreen comments:** explain WHAT/WHY, never how the code changed.

## Adding a model provider

Implement `Model` (`models/mod.rs`): `model_id()` and `stream()`. Get
`stream_aggregated()` for free from the trait default. Map vendor errors to
`StrandsError` variants with the cause preserved. `models/bedrock.rs` is the
reference.

## Testing

See [docs/TESTING.md](docs/TESTING.md). Unit tests are co-located in
`#[cfg(test)] mod tests`; integration tests live in `strands/tests/`.

## Commands

```bash
cargo build --all-features
cargo test --all-features
cargo clippy --all-targets --all-features
cargo fmt --check
cargo run --example weather_agent   # needs AWS credentials
```
