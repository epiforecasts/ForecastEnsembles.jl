test_that("blp recalibrates the target forecasts", {
  taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  n <- 60
  set.seed(1)
  y <- rnorm(n + 1)
  make_rows <- function(model_id, predictions) {
    do.call(rbind, lapply(taus, function(q) data.frame(
      model_id = model_id, output_type = "quantile",
      output_type_id = q, t = seq_along(predictions),
      value = predictions + qnorm(q), stringsAsFactors = FALSE
    )))
  }
  rows <- rbind(make_rows("m1", y + 0.3 * rnorm(n + 1)),
                make_rows("m2", y + rnorm(n + 1)))
  train  <- rows[rows$t <= n, ]
  target <- rows[rows$t == n + 1, ]
  obs <- data.frame(t = seq_len(n), observed = y[seq_len(n)])

  out <- blp(training = train, target = target, observations = obs,
             task_id_cols = "t")
  expect_s3_class(out, "data.frame")
  expect_equal(nrow(out), length(taus))
  expect_true(all(out$output_type == "quantile"))
  # Recalibrated quantiles are still monotone in the level.
  expect_false(is.unsorted(out$value[order(out$output_type_id)]))
})

test_that("blp validates observations and task_id_cols in R", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, t = 1, value = 1)
  expect_error(blp(df, df, data.frame(t = 1, truth = 0.5), task_id_cols = "t"),
               "observed")
  expect_error(blp(df, df, data.frame(t = 1, observed = 0.5)),
               "task_id_cols.*t")
})
