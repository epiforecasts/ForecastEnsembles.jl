test_that("linear_pool (quantile input) matches the saved hubEnsembles output", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "lp_quantile_input.csv"))

  out <- linear_pool(in_df, task_id_cols = c("location", "horizon"))
  expect_s3_class(out, "model_out_tbl")

  ref <- read.csv(file.path(ENS_REF_DIR, "lp_quantile_output.csv"))
  on_cols <- c("location", "horizon", "output_type_id")
  m <- merge(out[, c(on_cols, "value")],
             stats::setNames(ref[, c(on_cols, "value")], c(on_cols, "value_r")),
             by = on_cols)
  expect_equal(nrow(m), nrow(ref))
  # Julia inverts the mixture CDF exactly; the residual difference is the
  # R fixture's Monte Carlo noise plus spline-vs-PCHIP interiors.
  expect_lt(max(abs(m$value - m$value_r)), 0.05)
})

test_that("linear_pool (quantile input) is deterministic", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "lp_quantile_input.csv"))
  out1 <- linear_pool(in_df, task_id_cols = c("location", "horizon"))
  out2 <- linear_pool(in_df, task_id_cols = c("location", "horizon"))
  expect_identical(out1$value, out2$value)
})

test_that("linear_pool (sample input) is reproducible with a seed", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "lp_sample_input.csv"))
  out1 <- linear_pool(in_df, n_samples = 200, task_id_cols = "location",
                      seed = 42)
  out2 <- linear_pool(in_df, n_samples = 200, task_id_cols = "location",
                      seed = 42)
  expect_identical(out1$value, out2$value)
})

test_that("custom ensemble model_id is respected", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "lp_quantile_input.csv"))
  out <- linear_pool(in_df, task_id_cols = c("location", "horizon"),
                     model_id = "my-ensemble")
  expect_true(all(out$model_id == "my-ensemble"))
})
