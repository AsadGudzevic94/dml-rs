# Data and Python estimates written by parity/generate_r_fixtures.py.

read_parity <- function() {
  n <- 400
  names <- c(paste0("x", 1:4), "y", "d", "dcont", "z", "d_iv", "ydid", "m", "ymed", "dose", "ycd",
             "d3", "y3", "s", "ylee", "strata")
  raw <- readBin(test_path("fixtures", "parity_data.bin"), "double", n = length(names) * n,
                 size = 8, endian = "little")
  cols <- matrix(raw, nrow = n)
  colnames(cols) <- names
  expected <- utils::read.csv(test_path("fixtures", "parity_expected.csv"))
  list(data = cols, x = cols[, 1:4], expected = split(expected$value, expected$name))
}
