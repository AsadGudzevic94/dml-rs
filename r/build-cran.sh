#!/usr/bin/env sh
# Builds the CRAN tarball of dmlrs: a copy of r/dmlrs with the Rust crates bundled in
# src/rust/vendor.tar.xz and their versions locked in src/rust/Cargo.lock, so that the package
# installs offline. dml-core comes from crates.io (as for r-universe), not from this repository.
#
# Usage: r/build-cran.sh [output directory, default r/]
# Needs R, cargo >= 1.84 (for the Rust-version-aware resolver) and python3 >= 3.11.
# Also rewrites r/dmlrs/inst/AUTHORS (authors and licences of the crates); commit any change.
set -eu

R="${R:-R}"
PYTHON="${PYTHON:-python3}"
here=$(cd "$(dirname "$0")" && pwd)
out=$(cd "${1:-$here}" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

# A copy without build products and without src/.cargo (which points dml-core at this repository).
tar -C "$here" --exclude=dmlrs/src/rust/target --exclude=dmlrs/src/.cargo \
  --exclude=dmlrs/src/rust/Cargo.lock --exclude='dmlrs/src/*.o' --exclude='dmlrs/src/*.so' \
  --exclude='dmlrs/src/*.dll' --exclude=dmlrs/src/Makevars --exclude=dmlrs/src/Makevars.win \
  -cf - dmlrs | tar -C "$stage" -xf -

cd "$stage/dmlrs/src/rust"
# The newest versions that still build with the rust-version of Cargo.toml (the package's MSRV).
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo generate-lockfile
cargo vendor --quiet --locked --versioned-dirs vendor > /dev/null
"$PYTHON" - vendor "$here/dmlrs/inst/AUTHORS" <<'EOF'
import pathlib, sys, tomllib

vendor, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = [
    "dmlrs bundles the following Rust crates (in src/rust/vendor.tar.xz of the package",
    "tarball). Their authors hold the copyright; each crate is used under the licence shown.",
    "",
]
for crate in sorted(vendor.iterdir(), key=lambda p: p.name.lower()):
    package = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))["package"]
    authors = ", ".join(package.get("authors", [])) or "the " + package["name"] + " developers"
    lines.append(f"{package['name']} {package['version']} ({package.get('license', 'see crate')})")
    lines.append(f"  {authors}")
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
EOF
mkdir -p "$stage/dmlrs/inst" && cp "$here/dmlrs/inst/AUTHORS" "$stage/dmlrs/inst/AUTHORS"
tar -cJf vendor.tar.xz vendor
rm -rf vendor

cd "$stage"
"$R" CMD build --no-manual dmlrs
mv dmlrs_*.tar.gz "$out/"
ls -l "$out"/dmlrs_*.tar.gz
