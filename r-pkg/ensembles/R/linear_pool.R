#' Linear opinion pool
#'
#' Drop-in equivalent to `hubEnsembles::linear_pool`, computed by
#' Ensembles.jl. The kernel dispatches on the `output_type` of `model_out_tbl`:
#' samples are pooled, CDFs averaged, and quantile inputs go through a CDF
#' reconstruction (PCHIP + normal tails) before being re-extracted at the
#' input quantile levels.
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns. All rows must share the
#'   same `output_type`.
#' @param weights Optional data frame with columns `model_id` and `weight`.
#' @param n_samples Number of samples used for the pooled distribution when
#'   sampling/quantile paths are taken.
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

  .with_tempdir(function(td) {
    spec <- list(
      op = "linear_pool",
      input = .write_csv_temp(model_out_tbl, "input", td),
      task_id_cols = as.list(task_id_cols),
      n_samples = as.integer(n_samples),
      output = file.path(td, "out.csv"),
      .tempdir = td
    )
    if (!is.null(weights)) {
      spec$weights <- .write_csv_temp(weights, "weights", td)
    }
    .run_julia(spec)
  })
}
