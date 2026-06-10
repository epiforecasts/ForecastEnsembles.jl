test_that("qra attaches fitted weights as an attribute", {
  in_df <- read.csv(file.path(ENS_REF_DIR, "qra_input.csv"))
  target_date <- read.csv(file.path(ENS_REF_DIR, "qra_target.csv"))$target_date[1]
  in_df$target_date <- as.character(in_df$target_date)

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
  train <- in_df[in_df$target_date != target_date, ]
  target <- in_df[in_df$target_date == target_date, ]
  obs <- unique(train[, c("location", "horizon", "target_date", "observed")])

  out <- qra(
    training = to_ft(train), target = to_ft(target), observations = obs,
    task_id_cols = c("location", "horizon", "target_date")
  )
  w <- attr(out, "weights")
  expect_s3_class(w, "data.frame")
  expect_setequal(names(w), c("model_id", "weight"))
  expect_equal(sum(w$weight), 1, tolerance = 1e-6)
})
