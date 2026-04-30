test_that("simple_ensemble matches the saved hubEnsembles output", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "simple_input.csv"))
  ref_mean <- read.csv(file.path(ENS_REF_DIR, "simple_mean_output.csv"))

  out <- simple_ensemble(
    in_df,
    agg_fun = "mean",
    task_id_cols = c("location", "horizon")
  )
  on_cols <- c("location", "horizon", "output_type_id")
  m <- merge(out[, c(on_cols, "value")],
             setNames(ref_mean[, c(on_cols, "value")], c(on_cols, "value_r")),
             by = on_cols)
  expect_equal(nrow(m), nrow(ref_mean))
  expect_lt(max(abs(m$value - m$value_r)), 1e-10)
})

test_that("simple_ensemble (median) matches", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "simple_input.csv"))
  ref_med <- read.csv(file.path(ENS_REF_DIR, "simple_median_output.csv"))

  out <- simple_ensemble(in_df, agg_fun = "median",
                         task_id_cols = c("location", "horizon"))
  on_cols <- c("location", "horizon", "output_type_id")
  m <- merge(out[, c(on_cols, "value")],
             setNames(ref_med[, c(on_cols, "value")], c(on_cols, "value_r")),
             by = on_cols)
  expect_lt(max(abs(m$value - m$value_r)), 1e-10)
})

test_that("simple_ensemble (weighted mean) matches", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "simple_input.csv"))
  weights <- read.csv(file.path(ENS_REF_DIR, "simple_weights.csv"))
  ref <- read.csv(file.path(ENS_REF_DIR, "simple_wmean_output.csv"))

  out <- simple_ensemble(in_df, weights = weights, agg_fun = "mean",
                         task_id_cols = c("location", "horizon"))
  on_cols <- c("location", "horizon", "output_type_id")
  m <- merge(out[, c(on_cols, "value")],
             setNames(ref[, c(on_cols, "value")], c(on_cols, "value_r")),
             by = on_cols)
  expect_lt(max(abs(m$value - m$value_r)), 1e-10)
})
