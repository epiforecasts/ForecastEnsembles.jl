test_that("qra matches qrensemble default fixture", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "qra_input.csv"))
  target_date <- read.csv(file.path(ENS_REF_DIR, "qra_target.csv"))$target_date[1]
  in_df$target_date <- as.character(in_df$target_date)

  train <- in_df[in_df$target_date != target_date, ]
  target <- in_df[in_df$target_date == target_date, ]

  # Convert to ForecastTable shape.
  to_ft <- function(d) {
    data.frame(
      model_id       = d$model,
      output_type    = "quantile",
      output_type_id = d$quantile_level,
      location       = d$location,
      horizon        = d$horizon,
      target_date    = d$target_date,
      value          = d$predicted,
      stringsAsFactors = FALSE
    )
  }
  obs <- unique(train[, c("location", "horizon", "target_date", "observed")])

  out <- qra(
    training      = to_ft(train),
    target        = to_ft(target),
    observations  = obs,
    task_id_cols  = c("location", "horizon", "target_date"),
    per_quantile_weights = FALSE,
    enforce_normalisation = TRUE,
    intercept = FALSE,
    noncross = TRUE
  )

  ref <- read.csv(file.path(ENS_REF_DIR, "qra_default_output.csv"))
  ref$target_date <- as.character(ref$target_date)
  on_cols <- c("location", "horizon", "target_date", "output_type_id")
  ref$output_type_id <- ref$quantile_level
  m <- merge(
    out[, c(on_cols, "value")],
    ref[, c(on_cols, "predicted")],
    by = on_cols
  )
  expect_gt(nrow(m), 0)
  expect_lt(max(abs(m$value - m$predicted)), 1e-3)
})

test_that("a default qra() call does not warn about noncross", {
  skip_if_not_installed("pkgload")
  # The Julia-side warning arrives as subprocess stderr, not as an R condition,
  # so expect_no_warning() cannot see it -- it passes whether or not the warning
  # is emitted. Run the call in a fresh R process and inspect what it printed.
  pkg_dir <- normalizePath(file.path(testthat::test_path(), "..", ".."))
  script <- sprintf('
    options(ForecastEnsembles.jl_path = %s)
    if (requireNamespace("ForecastEnsembles", quietly = TRUE)) {
      suppressPackageStartupMessages(library(ForecastEnsembles))
    } else {
      pkgload::load_all(%s, quiet = TRUE)
    }
    ref <- %s
    in_df <- read.csv(file.path(ref, "qra_input.csv"))
    target_date <- read.csv(file.path(ref, "qra_target.csv"))$target_date[1]
    in_df$target_date <- as.character(in_df$target_date)
    train <- in_df[in_df$target_date != target_date, ]
    target <- in_df[in_df$target_date == target_date, ]
    to_ft <- function(d) data.frame(
      model_id = d$model, output_type = "quantile",
      output_type_id = d$quantile_level, location = d$location,
      horizon = d$horizon, target_date = d$target_date,
      value = d$predicted, stringsAsFactors = FALSE
    )
    obs <- unique(train[, c("location", "horizon", "target_date", "observed")])
    invisible(qra(training = to_ft(train), target = to_ft(target),
                  observations = obs,
                  task_id_cols = c("location", "horizon", "target_date")))
  ', deparse(ENS_REPO_ROOT), deparse(pkg_dir), deparse(ENS_REF_DIR))

  out <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"),
    args = c("-e", shQuote(script)),
    stdout = TRUE, stderr = TRUE
  ))

  # noncross defaults to TRUE here to mirror qrensemble, but per_quantile_weights
  # defaults to FALSE, where the flag does nothing. Forwarding it in that case
  # made the Julia constructor warn on a call the user configured no differently
  # from the documented default.
  expect_false(
    any(grepl("noncross", out, fixed = TRUE)),
    info = paste(out, collapse = "\n")
  )
})
