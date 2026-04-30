#' Hub-style simple/weighted ensemble
#'
#' Drop-in equivalent to `hubEnsembles::simple_ensemble`, computed by
#' Ensembles.jl over a JuliaConnectoR TCP bridge.
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns.
#' @param weights Optional data frame with columns `model_id` and `weight`.
#' @param agg_fun One of `"mean"` or `"median"`.
#' @param task_id_cols Character vector of task-id columns.
#'
#' @return A data frame, with `model_id = "hub-ensemble"`.
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
                            task_id_cols) {
  agg_fun <- match.arg(agg_fun)
  if (missing(task_id_cols))
    stop("`task_id_cols` is required.", call. = FALSE)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_simple")
  out <- fn(as.data.frame(model_out_tbl),
            as.list(task_id_cols),
            agg_fun,
            if (is.null(weights)) NULL else as.data.frame(weights))
  .julia_to_df(out)
}
