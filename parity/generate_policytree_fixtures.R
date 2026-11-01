# Reference trees of R's policytree (exact tree search) for dml-rs
# OptimalPolicyTree: node lists and predicted actions.
#
#   Rscript parity/generate_policytree_fixtures.R
suppressPackageStartupMessages({
  library(policytree)
  library(jsonlite)
})
stopifnot(packageVersion("policytree") == "1.2.5")

make <- function(n, p, d, seed, discrete = FALSE) {
  set.seed(seed)
  X <- matrix(if (discrete) sample(0:4, n * p, TRUE) else round(rnorm(n * p), 3), n, p)
  Gamma <- matrix(rnorm(n * d), n, d)
  Gamma[, 2] <- Gamma[, 2] + (X[, 1] > 0)
  if (d > 2) Gamma[, 3] <- Gamma[, 3] + 0.8 * (X[, 2] > 0.5)
  list(X = X, Gamma = Gamma)
}

cases <- list(
  list(name = "depth1", n = 300, p = 4, d = 3, depth = 1, seed = 1),
  list(name = "depth2", n = 300, p = 4, d = 3, depth = 2, seed = 2),
  list(name = "depth2_discrete_ties", n = 400, p = 3, d = 2, depth = 2, seed = 3, discrete = TRUE),
  list(name = "depth2_split_step_min_node", n = 400, p = 3, d = 3, depth = 2, seed = 4,
       split_step = 3, min_node_size = 20),
  list(name = "depth3_small", n = 80, p = 3, d = 3, depth = 3, seed = 5),
  list(name = "depth0", n = 50, p = 2, d = 4, depth = 0, seed = 6)
)
out <- list()
for (case in cases) {
  dat <- make(case$n, case$p, case$d, case$seed, isTRUE(case$discrete))
  step <- if (is.null(case$split_step)) 1 else case$split_step
  mns <- if (is.null(case$min_node_size)) 1 else case$min_node_size
  tree <- policy_tree(dat$X, dat$Gamma, depth = case$depth, split.step = step, min.node.size = mns,
                      verbose = FALSE)
  nodes <- lapply(tree$nodes, function(nd) {
    if (nd$is_leaf) list(is_leaf = TRUE, action = nd$action - 1)
    else list(is_leaf = FALSE, split_variable = nd$split_variable - 1, split_value = nd$split_value,
              left = nd$left_child - 1, right = nd$right_child - 1)
  })
  out[[case$name]] <- list(x = unname(dat$X), gamma = unname(dat$Gamma), depth = case$depth,
                           split_step = step, min_node_size = mns, nodes = nodes,
                           predictions = as.integer(predict(tree, dat$X)) - 1L)
  cat(sprintf("%-28s nodes %d\n", case$name, length(nodes)))
}
out$`_meta` <- list(policytree = as.character(packageVersion("policytree")), R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor"),
           file.path("crates", "dml-core", "tests", "fixtures", "policytree_r.json"))
