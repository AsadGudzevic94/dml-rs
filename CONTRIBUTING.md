# Contributing to dml-rs

Thank you for your interest in improving dml-rs. Bug reports, feature requests and pull requests
are welcome.

## Reporting issues

Please open an issue at <https://github.com/AsadGudzevic94/dml-rs/issues> with a minimal
reproducible example, the versions of dml-rs (and of Python, NumPy and scikit-learn if you use the
Python package) and, for numerical questions, the result you expected.

## Repository layout

```
crates/
  dml-core/   Rust library: data, learners, cross-fitting, models, inference
  dml-py/     Python bindings (PyO3 + maturin), published as the `dml-rs` package
parity/       DoubleML fixture generator and benchmark scripts
```

## Development setup

You need a stable Rust toolchain (the minimum supported version is 1.80) and, for the Python
bindings and parity fixtures, Python 3.9 or newer.

```sh
# Rust: format, lint, test, document
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps

# Python bindings, in a virtual environment
python -m venv .venv
.venv/Scripts/pip install -r parity/requirements.txt pytest maturin   # .venv/bin/pip on Linux/macOS
cd crates/dml-py
maturin develop --release
pytest tests

# Type stubs (python/dml_rs/_dml_rs.pyi) must match the extension: update them with any change
# to a Python-facing signature in src/
pip install mypy pandas-stubs
python -m mypy.stubtest dml_rs._dml_rs
python -m mypy -p dml_rs
```

Every pull request runs the same checks in CI on Linux, macOS and Windows, plus a build on the
minimum supported Rust version.

## Numerical parity with DoubleML

Correctness is checked against the reference [DoubleML](https://docs.doubleml.org) package.
`parity/generate_fixtures.py` runs DoubleML on fixed data and fixed sample splits and writes JSON
fixtures to `crates/dml-core/tests/fixtures/`; `crates/dml-core/tests/parity.rs` and
`crates/dml-core/tests/staggered.rs` refit every case on the same folds and compare.

When adding a model or an option, please add a fixture case to the generator and regenerate:

```sh
.venv/Scripts/python parity/generate_fixtures.py
```

Regeneration is deterministic: existing fixtures should come out unchanged.

## Code style

- `cargo fmt` and `cargo clippy -- -D warnings` must pass.
- Public items have doc comments; new estimators come with unit tests, a DoubleML parity fixture
  where DoubleML offers the same estimator, and, for inference, a Monte-Carlo coverage test.
- Keep the core free of heavy dependencies: no BLAS or C libraries.

## Releasing (maintainers)

Releases are published from GitHub Actions with Trusted Publishing; no API tokens are involved.

1. Bump `version` in the root `Cargo.toml` and move the `Unreleased` notes in `CHANGELOG.md`
   under the new version.
2. Commit, then tag and push: `git tag -a v0.6.0 -m "dml-rs 0.6.0" && git push origin v0.6.0`.
3. Approve the two deployments in the Actions tab: `release-crate.yml` publishes `dml-core` to
   crates.io and `wheels.yml` publishes `dml-rs` to PyPI. Both stop if the tag does not match the
   `Cargo.toml` version.

## License

dml-rs is licensed under either of the [Apache License 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
