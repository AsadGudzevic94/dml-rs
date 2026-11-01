# Run by configure / configure.win: checks the Rust toolchain and writes src/Makevars(.win) from
# src/Makevars(.win).in.

desc <- read.dcf("DESCRIPTION")
msrv <- sub(".*rustc >= ([0-9.]+).*", "\\1", desc[, "SystemRequirements"])

install_hint <- c(
  "Install Rust from https://www.rust-lang.org/tools/install or with the system package manager:",
  "  Debian/Ubuntu: apt-get install cargo",
  "  Fedora: dnf install cargo",
  "  macOS: brew install rust"
)

# cargo installed by rustup lives in ~/.cargo/bin, which is not always on the PATH.
is_windows <- .Platform$OS.type == "windows"
Sys.setenv(PATH = paste(Sys.getenv("PATH"), file.path(Sys.getenv("HOME"), ".cargo", "bin"),
                        sep = .Platform$path.sep))
version_of <- function(tool) {
  out <- tryCatch(suppressWarnings(system2(tool, "--version", stdout = TRUE, stderr = TRUE)),
                  error = function(e) character())
  if (!length(out) || !is.null(attr(out, "status"))) {
    stop(paste(c(sprintf("'%s' was not found: dmlrs needs Rust %s or later.", tool, msrv),
                 install_hint), collapse = "\n"), call. = FALSE)
  }
  out[1]
}
rustc <- version_of("rustc")
cargo <- version_of("cargo")
message("Using ", rustc, "\nUsing ", cargo)

installed <- sub("^rustc ([0-9]+\\.[0-9]+(\\.[0-9]+)?).*", "\\1", rustc)
if (utils::compareVersion(installed, msrv) < 0) {
  stop(paste(c(sprintf("dmlrs needs Rust %s or later; found %s.", msrv, installed),
               "Update with `rustup update` or install a newer Rust:", install_hint[-1]),
             collapse = "\n"), call. = FALSE)
}

# A tarball made by r/build-cran.sh bundles the crates: build offline with the locked versions,
# two jobs (CRAN policy) and a temporary CARGO_HOME, and remove the build tree afterwards.
bundled <- file.exists("src/rust/vendor.tar.xz")
if (bundled) message("Building from the bundled Rust crates (offline).")

windows_target <- if (grepl("aarch", R.version$platform)) {
  "aarch64-pc-windows-gnullvm"
} else if (grepl("clang", Sys.getenv("R_COMPILED_BY"))) {
  "x86_64-pc-windows-gnullvm"
} else if (grepl("i386", R.version$platform)) {
  "i686-pc-windows-gnu"
} else {
  "x86_64-pc-windows-gnu"
}
target_dir <- if (is_windows) "$(TARGET_DIR)/$(TARGET)" else "$(TARGET_DIR)"

values <- c(
  "@LIBDIR@" = "release",
  "@WINDOWS_TARGET@" = windows_target,
  "@CARGO_HOME@" = if (bundled) "export CARGO_HOME=\"$(CARGOTMP)\" && " else "",
  "@CARGO_FLAGS@" = if (bundled) "-j 2 --frozen" else "",
  # Keep the compiled crates of a development build for the next install.
  "@CLEAN_TARGET@" = if (bundled) "\"$(TARGET_DIR)\"" else paste0(target_dir, "/release/build")
)

template <- if (is_windows) "src/Makevars.win.in" else "src/Makevars.in"
lines <- readLines(template)
for (key in names(values)) lines <- gsub(key, values[[key]], lines, fixed = TRUE)
out <- sub("\\.in$", "", template)
con <- file(out, open = "wb")
writeLines(lines, con, sep = "\n")
close(con)
message("Wrote ", out)
