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
