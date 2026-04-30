#' Linear opinion pool
#'
#' Drop-in equivalent to `hubEnsembles::linear_pool`. Routes by
#' `output_type` of `model_out_tbl`: samples pooled, CDFs averaged, and
#' quantile inputs go through CDF reconstruction (PCHIP + normal tails).
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns.
#' @param weights Optional data frame with columns `model_id` and `weight`.
#' @param n_samples Pooled-sample size for the sampling/quantile paths.
#' @param task_id_cols Character vector of task-id columns.
#'
#' @return A data frame in the same shape, with `model_id = "hub-ensemble"`.
#' @export
linear_pool <- function(model_out_tbl,
                        weights = NULL,
                        n_samples = 10000L,
                        task_id_cols) {
  if (missing(task_id_cols))
    stop("`task_id_cols` is required.", call. = FALSE)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_linear_pool")
  out <- fn(as.data.frame(model_out_tbl),
            as.list(task_id_cols),
            as.integer(n_samples),
            if (is.null(weights)) NULL else as.data.frame(weights))
  .julia_to_df(out)
}
