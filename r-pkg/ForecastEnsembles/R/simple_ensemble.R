#' Hub-style simple/weighted ensemble
#'
#' R wrapper that mirrors `hubEnsembles::simple_ensemble`. The actual
#' aggregation runs in Julia.
#'
#' Differences from `hubEnsembles::simple_ensemble` to be aware of when
#' migrating:
#' \itemize{
#'   \item `task_id_cols` is required here; `hubEnsembles` infers it from
#'     the known hub schema columns.
#'   \item `agg_fun` is a string choice (`"mean"` or `"median"`), with
#'     `"mean"` as the default; `hubEnsembles` also accepts arbitrary
#'     aggregation functions, which this wrapper does not (the
#'     aggregation runs in Julia, so an R function can't cross the
#'     bridge).
#' }
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns.
#' @param weights Optional data frame with columns `model_id` and `weight`
#'   (optionally `output_type_id` for per-quantile weights). Weights need
#'   not sum to 1; they are normalised internally.
#' @param agg_fun One of `"mean"` or `"median"`.
#' @param task_id_cols Character vector of the column names that identify
#'   a forecast task, e.g. `c("location", "horizon", "target_date")`.
#' @param model_id Value for the `model_id` column of the output. Defaults
#'   to `"hub-ensemble"`, matching `hubEnsembles`.
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A data frame of class `model_out_tbl`, with one row per
#'   (task, output_type_id) and `model_id` set as requested.
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   model_id       = rep(c("m1", "m2", "m3"), each = 2),
#'   output_type    = "quantile",
#'   output_type_id = rep(c(0.25, 0.75), 3),
#'   location       = "A",
#'   value          = c(1.0, 3.0, 2.0, 4.0, 0.5, 2.5)
#' )
#' simple_ensemble(df, agg_fun = "mean", task_id_cols = "location")
#'
#' # Weighted variant.
#' w <- data.frame(model_id = c("m1", "m2", "m3"), weight = c(0.5, 0.3, 0.2))
#' simple_ensemble(df, weights = w, agg_fun = "median",
#'                 task_id_cols = "location")
#' }
#' @export
simple_ensemble <- function(model_out_tbl,
                            weights = NULL,
                            agg_fun = c("mean", "median"),
                            task_id_cols,
                            model_id = "hub-ensemble") {
  agg_fun <- match.arg(agg_fun)
  .validate_forecast_df(model_out_tbl, "model_out_tbl")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, model_out_tbl)
  .validate_weights(weights)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_simple")
  out <- fn(as.data.frame(model_out_tbl),
            as.list(task_id_cols),
            agg_fun,
            if (is.null(weights)) NULL else as.data.frame(weights))
  res <- .julia_to_df(out)
  res$model_id <- model_id
  .as_model_out_tbl(res)
}
