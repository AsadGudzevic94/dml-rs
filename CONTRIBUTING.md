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
r/dmlrs/      R package (extendr), on r-universe
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

### R package

`r/dmlrs` needs R 4.2 or newer and, on Windows, Rtools and the `x86_64-pc-windows-gnu` Rust
target. Inside the repository `r/dmlrs/src/.cargo/config.toml` builds it against the local
`dml-core`; the package tarball uses `dml-core` from crates.io, so R code that needs a new
`dml-core` API waits for a `dml-core` release. `R/extendr-wrappers.R` is written by hand: add a
wrapper for every new `#[extendr]` function.

```sh
R CMD INSTALL r/dmlrs
Rscript -e 'testthat::test_local("r/dmlrs", load_package = "installed")'
Rscript -e 'roxygen2::roxygenise("r/dmlrs", roclets = c("rd", "namespace"))'   # after doc changes

# The CRAN tarball: Rust crates bundled (offline build), inst/AUTHORS regenerated
r/build-cran.sh r
R CMD check --as-cran r/dmlrs_*.tar.gz
```

## Documentation

The site (https://asadgudzevic94.github.io/dml-rs/) is built with MkDocs from `docs/` and deployed
from `main` by `.github/workflows/docs.yml`. The example notebooks in `docs/examples/` are
committed with their outputs and are not executed during the build; re-run a notebook after
changing it or the API it uses:

```sh
pip install -r docs/requirements.txt
mkdocs serve                     # preview at http://127.0.0.1:8000
mkdocs build --strict            # what CI runs

# in the development environment (needs dml_rs, DoubleML, matplotlib, nbconvert)
jupyter nbconvert --to notebook --execute --inplace docs/examples/01_getting_started.ipynb
```

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

The weekly workflow "DoubleML main" runs the Python tests against DoubleML's main branch. A new
failure there means an upstream change to look at: a fix dml-rs needs too, or a known difference to
add to `crates/dml-py/tests/conftest.py` and `docs/validation.md`.

## Code style

- `cargo fmt` and `cargo clippy -- -D warnings` must pass.
- Public items have doc comments; new estimators come with unit tests, a DoubleML parity fixture
  where DoubleML offers the same estimator, and, for inference, a Monte-Carlo coverage test.
- Keep the core free of heavy dependencies: no BLAS or C libraries.

## API stability

The project follows semantic versioning. When changing a public API:

- Rust: new settings enums and structs with public fields are `#[non_exhaustive]`; give settings
  structs chained setters (`field_setters!`). Traits implemented by users (`Regressor`,
  `Classifier`, `LinearScoreModel`) only gain methods with default implementations.
- Renames keep the old name as a `#[deprecated(since = ...)]` alias (Rust) or a
  `DeprecationWarning` (Python) until the next major version; list them in the changelog.
- `dml_rs.native` (the low-level functions and result classes) may change in minor releases.
- The minimum supported Rust version is raised only in a minor release, with a changelog note.

## Releasing (maintainers)

Releases are published from GitHub Actions with Trusted Publishing; no API tokens are involved.

1. Bump `version` in the root `Cargo.toml` and move the `Unreleased` notes in `CHANGELOG.md`
   under the new version.
2. Commit, then tag and push: `git tag -a v0.6.0 -m "dml-rs 0.6.0" && git push origin v0.6.0`.
3. Approve the two deployments in the Actions tab: `release-crate.yml` publishes `dml-core` to
   crates.io and `wheels.yml` publishes `dml-rs` to PyPI. Both stop if the tag does not match the
   `Cargo.toml` version.

The R package is versioned separately (`r/dmlrs/DESCRIPTION`, `NEWS.md`). r-universe rebuilds it
from `main`. For CRAN: bump the version, run `r/build-cran.sh r`, commit any change to
`inst/AUTHORS`, check the tarball with `R CMD check --as-cran` (also on win-builder and
macbuilder), update `r/dmlrs/cran-comments.md` and upload the tarball at
https://cran.r-project.org/submit.html.

## License

dml-rs is licensed under either of the [Apache License 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
