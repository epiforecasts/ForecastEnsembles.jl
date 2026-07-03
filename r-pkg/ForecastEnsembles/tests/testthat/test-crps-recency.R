test_that("crps_weights with the lopensemble ramp matches the R fixture", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "crps_input.csv"))
  ft_df <- data.frame(
    model_id       = in_df$model,
    output_type    = "sample",
    output_type_id = in_df$sample_id,
    date           = in_df$date,
    value          = in_df$predicted,
    stringsAsFactors = FALSE
  )
  obs <- unique(in_df[, c("date", "observed")])

  out <- crps_weights(ft_df, obs, task_id_cols = "date",
                      lambda = "lopensemble", time_col = "date")

  ref <- read.csv(file.path(ENS_REF_DIR, "crps_weights_ramp_output.csv"))
  m <- merge(out, stats::setNames(ref, c("model_id", "weight_r")),
             by = "model_id")
  expect_equal(nrow(m), nrow(ref))
  expect_lt(max(abs(m$weight - m$weight_r)), 0.05)
})

test_that("lambda without time_col is caught in R", {
  df <- data.frame(model_id = "m1", output_type = "sample",
                   output_type_id = 1, t = 1, value = 1)
  obs <- data.frame(t = 1, observed = 0.5)
  expect_error(crps_weights(df, obs, task_id_cols = "t", lambda = 0.9),
               "time_col")
})
