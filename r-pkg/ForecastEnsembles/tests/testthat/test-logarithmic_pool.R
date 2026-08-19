test_that("logarithmic_pool returns a well-formed quantile ensemble", {
  taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  df <- do.call(rbind, lapply(c("m1", "m2"), function(m) data.frame(
    model_id       = m,
    output_type    = "quantile",
    output_type_id = taus,
    location       = "A",
    value          = qnorm(taus, mean = if (m == "m1") 0 else 2, sd = 1),
    stringsAsFactors = FALSE
  )))

  out <- logarithmic_pool(df, task_id_cols = "location")
  expect_s3_class(out, "model_out_tbl")
  expect_equal(nrow(out), length(taus))
  expect_true(all(out$output_type == "quantile"))
  expect_true(all(out$model_id == "hub-ensemble"))
  # Pooled quantiles are monotone and lie between the two components' medians.
  expect_false(is.unsorted(out$value[order(out$output_type_id)]))
  med <- out$value[out$output_type_id == 0.5]
  expect_gt(med, 0)
  expect_lt(med, 2)
})

test_that("logarithmic_pool validates task_id_cols in R", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, location = "A", value = 1)
  expect_error(logarithmic_pool(df), "task_id_cols.*location")
})
