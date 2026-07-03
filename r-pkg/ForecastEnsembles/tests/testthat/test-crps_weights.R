test_that("crps_weights matches lopensemble fixture within solver tolerance", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "crps_input.csv"))

  # Convert lopensemble shape (model, sample_id, predicted, observed, date)
  # into ForecastTable shape.
  ft_df <- data.frame(
    model_id       = in_df$model,
    output_type    = "sample",
    output_type_id = in_df$sample_id,
    date           = in_df$date,
    value          = in_df$predicted,
    stringsAsFactors = FALSE
  )
  obs <- unique(in_df[, c("date", "observed")])

  out <- crps_weights(
    training       = ft_df,
    observations   = obs,
    task_id_cols   = "date",
    dirichlet_alpha = 1.001
  )

  ref <- read.csv(file.path(ENS_REF_DIR, "crps_weights_output.csv"))
  m <- merge(out, setNames(ref, c("model_id", "weight_r")), by = "model_id")
  expect_equal(nrow(m), nrow(ref))
  expect_lt(max(abs(m$weight - m$weight_r)), 0.05)
})
