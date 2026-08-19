test_that("trimmed_mean drops the extremes before averaging", {
  # Five models at one quantile level; fraction 0.2 trims round(0.2 * 5) = 1
  # from each end, so the mean is over c(2, 3, 4) = 3 (the 1 and 100 dropped).
  df <- data.frame(
    model_id       = paste0("m", 1:5),
    output_type    = "quantile",
    output_type_id = 0.5,
    location       = "A",
    value          = c(1, 2, 3, 4, 100)
  )
  out <- trimmed_mean(df, fraction = 0.2, task_id_cols = "location")
  expect_s3_class(out, "model_out_tbl")
  expect_equal(nrow(out), 1L)
  # as.numeric() drops the JLDIM attribute JuliaConnectoR marks scalar columns
  # with, which expect_equal would otherwise flag against the plain literal.
  expect_equal(as.numeric(out$value), 3, tolerance = 1e-10)
  expect_true(all(out$model_id == "hub-ensemble"))
})

test_that("trimmed_mean winsorise clamps rather than drops", {
  df <- data.frame(
    model_id       = paste0("m", 1:5),
    output_type    = "quantile",
    output_type_id = 0.5,
    location       = "A",
    value          = c(1, 2, 3, 4, 100)
  )
  # Clamp the extreme (1 -> 2, 100 -> 4): mean of c(2, 2, 3, 4, 4) = 3.
  out <- trimmed_mean(df, fraction = 0.2, mode = "winsorise",
                      task_id_cols = "location")
  expect_equal(as.numeric(out$value), 3, tolerance = 1e-10)
})

test_that("trimmed_mean validates its arguments in R", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, location = "A", value = 1)
  expect_error(trimmed_mean(df), "task_id_cols.*location")
  expect_error(trimmed_mean(df, mode = "nonsense", task_id_cols = "location"),
               "should be one of")
})
