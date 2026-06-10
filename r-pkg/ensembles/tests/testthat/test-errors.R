# Error-path smoke tests. These exercise the R-side validation, so they
# must produce actionable R errors without touching Julia.

test_that("missing task_id_cols gives an actionable message", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, location = "A", value = 1)
  expect_error(simple_ensemble(df), "task_id_cols.*location")
})

test_that("scoringutils-style column names get a rename hint", {
  df <- data.frame(model = "m1", quantile_level = 0.5,
                   predicted = 1, location = "A")
  expect_error(simple_ensemble(df, task_id_cols = "location"),
               "model.*model_id")
})

test_that("task_id_cols absent from the data is caught in R", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, location = "A", value = 1)
  expect_error(simple_ensemble(df, task_id_cols = "horizon"),
               "not present.*horizon")
})

test_that("malformed weights are caught in R", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, location = "A", value = 1)
  expect_error(
    simple_ensemble(df, weights = data.frame(model = "m1", w = 1),
                    task_id_cols = "location"),
    "model_id.*weight")
  expect_error(
    simple_ensemble(df,
                    weights = data.frame(model_id = "m1", weight = -1),
                    task_id_cols = "location"),
    "non-negative")
})

test_that("observations without an `observed` column is caught in R", {
  df <- data.frame(model_id = "m1", output_type = "quantile",
                   output_type_id = 0.5, t = 1, value = 1)
  obs <- data.frame(t = 1, truth = 0.5)
  expect_error(qra(df, df, obs, task_id_cols = "t"),
               "observed")
})

test_that("mixture_from_samples requires weights", {
  df <- data.frame(model_id = "m1", output_type = "sample",
                   output_type_id = 1, t = 1, value = 1)
  expect_error(mixture_from_samples(df, task_id_cols = "t"),
               "weights.*required")
})
