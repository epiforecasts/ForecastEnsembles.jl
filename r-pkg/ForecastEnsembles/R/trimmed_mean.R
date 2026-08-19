#' Trimmed or winsorised cross-model mean
#'
#' R wrapper for the robust ensemble mean. At each task and `output_type_id`
#' (e.g. a quantile level), the per-model values are ordered and either
#' **trimmed** (the `fraction` most extreme from each end are dropped) or
#' **winsorised** (those extremes are clamped to the surviving boundary values)
#' before averaging. The aggregation runs in Julia.
#'
#' Because the count trimmed is `round(fraction * n)`, small ensembles need a
#' large enough `fraction` to trim anything at all: with the default
#' `fraction = 0.1`, nothing is trimmed until `n = 6` models.
#'
#' Operates on values that are comparable across models at a shared
#' `output_type_id`, so it supports `"quantile"` and `"cdf"` forecasts, not
#' `"sample"` (sample indices are not aligned across models).
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns.
#' @param fraction Proportion trimmed or clamped from each end, in `[0, 0.5)`.
#' @param mode One of `"trim"` (drop the extremes) or `"winsorise"` (clamp
#'   them).
#' @param task_id_cols Character vector of the column names that identify a
#'   forecast task, e.g. `c("location", "horizon", "target_date")`.
#' @param model_id Value for the `model_id` column of the output. Defaults to
#'   `"hub-ensemble"`.
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A data frame of class `model_out_tbl`, one row per
#'   (task, output_type_id), with `model_id` set as requested.
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   model_id       = paste0("m", 1:5),
#'   output_type    = "quantile",
#'   output_type_id = 0.5,
#'   location       = "A",
#'   value          = c(1, 2, 3, 4, 100)
#' )
#' trimmed_mean(df, fraction = 0.2, task_id_cols = "location")
#' }
#' @export
trimmed_mean <- function(model_out_tbl,
                         fraction = 0.1,
                         mode = c("trim", "winsorise"),
                         task_id_cols,
                         model_id = "hub-ensemble") {
  mode <- match.arg(mode)
  .validate_forecast_df(model_out_tbl, "model_out_tbl")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, model_out_tbl)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_trimmed")
  out <- fn(as.data.frame(model_out_tbl),
            as.list(task_id_cols),
            as.numeric(fraction),
            mode)
  res <- .julia_to_df(out)
  res$model_id <- model_id
  .as_model_out_tbl(res)
}
