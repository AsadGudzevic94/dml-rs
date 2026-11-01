## Submission

This is a new submission.

dmlrs is the R interface to the Rust crate 'dml-core' (on crates.io), which also powers the
Python package 'dml-rs' on PyPI.

## Rust

* The package follows the CRAN guidance for Rust packages
  (<https://cran.r-project.org/web/packages/using_rust.html>): all Rust crates are bundled in
  `src/rust/vendor.tar.xz` (xz compressed) and built offline (`cargo build --frozen`) with
  `-j 2`; nothing is downloaded at install time.
* `configure` reports the `rustc` and `cargo` versions before compiling and stops with
  installation instructions if they are missing or older than 1.80 (the minimum, stated in
  SystemRequirements).
* `CARGO_HOME` is a temporary directory inside the build tree, removed after the build; the
  Rust build directory is removed as well.
* The authors and licences of the bundled crates are listed in `inst/AUTHORS`, referenced from
  `Authors@R`. All are MIT, Apache-2.0 or BSD-2-Clause (most dual licensed).
* The package uses at most two threads when `_R_CHECK_LIMIT_CORES_` is set.

## Test environments

* GitHub Actions: Ubuntu, macOS (arm64) and Windows with R release and stable Rust, and Ubuntu
  with Rust 1.80: `R CMD check --as-cran`, Status: OK.
* Local: Windows 11, R 4.5.3, Rtools45.

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.
