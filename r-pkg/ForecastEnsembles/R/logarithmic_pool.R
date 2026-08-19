#' Logarithmic opinion pool
#'
#' R wrapper for the logarithmic opinion pool (a weighted *geometric* mean of
#' the component densities, in contrast to the arithmetic mean of
#' [linear_pool()]). The pooling runs in Julia: each model's quantile forecast
#' is turned into a density on a shared grid, the densities are combined in log
#' space, and quantiles are re-extracted.
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns.
#' @param weights Optional data frame with columns `model_id` and `weight`.
#'   Weights need not sum to 1; they are normalised internally.
#' @param ngrid Number of grid points used to evaluate the pooled density
#'   before re-extracting quantiles. Higher is more accurate but slower.
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
#' df <- expand.grid(
#'   model_id       = c("m1", "m2"),
#'   output_type_id = c(0.1, 0.25, 0.5, 0.75, 0.9),
#'   location       = "A",
#'   stringsAsFactors = FALSE
#' )
#' df$output_type <- "quantile"
#' df$value <- qnorm(df$output_type_id,
#'                   mean = ifelse(df$model_id == "m1", 0, 2), sd = 1)
#' logarithmic_pool(df, task_id_cols = "location")
#' }
#' @export
logarithmic_pool <- function(model_out_tbl,
                             weights = NULL,
                             ngrid = 2000L,
                             task_id_cols,
                             model_id = "hub-ensemble") {
  .validate_forecast_df(model_out_tbl, "model_out_tbl")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, model_out_tbl)
  .validate_weights(weights)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_logpool")
  out <- fn(as.data.frame(model_out_tbl),
            as.list(task_id_cols),
            if (is.null(weights)) NULL else as.data.frame(weights),
            as.integer(ngrid))
  res <- .julia_to_df(out)
  res$model_id <- model_id
  .as_model_out_tbl(res)
}
