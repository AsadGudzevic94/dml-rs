# Reference results of causalweight::medDML (Farbmacher et al. 2022; nested
# means debiased as in Qi and Zhang 2026) for the dml-rs Mediation model.
#
# MLmethod = "parametric" makes every nuisance an lm / glm(logit) fit, which
# dml-rs reproduces with Ols and LogisticRegression::unpenalized(). medDML draws
# its three folds with set.seed(1); sample(n) before any other random number,
# so the same call here gives the folds it used.
#
#   Rscript parity/generate_mediation_fixtures.R
#
# Needs causalweight (1.1.6) and jsonlite.
suppressPackageStartupMessages({
  library(causalweight)
  library(jsonlite)
})
stopifnot(packageVersion("causalweight") == "1.1.6")

simulate <- function(n, confounding) {
  set.seed(20261031)
  x <- matrix(rnorm(n * 4), n, 4)
  b <- c(0.5, -0.4, 0.3, 0)
  d <- as.numeric(confounding * (x %*% b) + rnorm(n) > 0)
  m1 <- 0.6 * d + 0.5 * x[, 1] - 0.3 * x[, 3] + rnorm(n)
  m2 <- as.numeric(0.4 * d + 0.5 * x[, 2] + rnorm(n) > 0)
  y <- 0.5 * d + m1 + 0.5 * m2 + 0.3 * d * m1 + x %*% c(0.4, 0.2, -0.3, 0.1) + rnorm(n)
  list(y = as.numeric(y), d = d, m = cbind(m1, m2), x = x)
}

medDML_folds <- function(n) {
  set.seed(1)
  idx <- sample(n)
  folds <- split(idx, cut(seq_along(idx), breaks = 3, labels = FALSE))
  lapply(folds, function(f) as.integer(f - 1))
}

cases <- list(
  list(name = "default", mediators = 1, args = list()),
  list(name = "two_mediators", mediators = 1:2, args = list()),
  list(name = "unnormalized_plain", mediators = 1,
       args = list(normalized = FALSE, debiasfirst = FALSE)),
  list(name = "fewsplits", mediators = 1:2, args = list(fewsplits = TRUE)),
  list(name = "strong_confounding_trim", mediators = 1:2, confounding = 1.6,
       args = list(trim = 0.08))
)

n <- 900
out <- list()
for (case in cases) {
  conf <- if (is.null(case$confounding)) 1 else case$confounding
  dat <- simulate(n, conf)
  m <- dat$m[, case$mediators, drop = FALSE]
  args <- c(list(y = dat$y, d = dat$d, m = if (ncol(m) == 1) m[, 1] else m, x = dat$x,
                 MLmethod = "parametric"), case$args)
  res <- do.call(medDML, args)
  out[[case$name]] <- list(
    y = dat$y, d = dat$d, m = unname(m), x = unname(dat$x),
    folds = unname(medDML_folds(n)),
    options = c(list(trim = 0.05, normalized = TRUE, fewsplits = FALSE, debiasfirst = TRUE)[
      setdiff(c("trim", "normalized", "fewsplits", "debiasfirst"), names(case$args))], case$args),
    effect = unname(res$results["effect", ]),
    se = unname(res$results["se", ]),
    ntrimmed = res$ntrimmed
  )
}
out$`_meta` <- list(causalweight = as.character(packageVersion("causalweight")),
                    R = R.version.string,
                    columns = c("total", "dir.treat", "dir.control", "indir.treat",
                                "indir.control", "Y(0,M(0))"))
dir <- file.path("crates", "dml-core", "tests", "fixtures")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor"),
           file.path(dir, "mediation_causalweight.json"))
for (nm in names(out)) if (nm != "_meta")
  cat(sprintf("%-26s ntrimmed=%3d  effects=%s\n", nm, out[[nm]]$ntrimmed,
              paste(sprintf("%.4f", out[[nm]]$effect), collapse = " ")))
