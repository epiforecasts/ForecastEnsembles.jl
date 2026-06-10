#' Linear opinion pool
#'
#' R wrapper that mirrors `hubEnsembles::linear_pool`. Routes by
#' `output_type` of `model_out_tbl`: samples are pooled by weighted
#' resampling, CDFs averaged pointwise, and quantile inputs go through CDF
#' reconstruction (PCHIP + normal tails) followed by exact inversion of
#' the mixture CDF.
#'
#' Differences from `hubEnsembles::linear_pool` to be aware of when
#' migrating:
#' \itemize{
#'   \item The sample-count argument is named `n_samples` here;
#'     `hubEnsembles` calls it `n_output_samples`. A call passing
#'     `n_output_samples` errors rather than being silently ignored.
#'   \item `task_id_cols` is required here.
#'   \item For quantile input, the ensemble quantiles are computed exactly
#'     (no Monte Carlo step), so `n_samples` only affects the `"sample"`
#'     output type and results for quantile input are deterministic.
#' }
#'
#' @param model_out_tbl A data frame with columns `model_id`, `output_type`,
#'   `output_type_id`, `value`, plus task-id columns. All rows must share
#'   the same `output_type`.
#' @param weights Optional data frame with columns `model_id` and `weight`.
#'   Weights need not sum to 1; they are normalised internally.
#' @param n_samples Pooled-sample size when `output_type` is `"sample"`
#'   (named `n_output_samples` in `hubEnsembles`).
#' @param task_id_cols Character vector of the column names that identify
#'   a forecast task, e.g. `c("location", "horizon", "target_date")`.
#' @param model_id Value for the `model_id` column of the output. Defaults
#'   to `"hub-ensemble"`, matching `hubEnsembles`.
#' @param seed Optional integer seed for the Julia random number generator,
#'   used only by the `"sample"` path. Without it, sample-path results are
#'   not reproducible across sessions (R's `set.seed()` does not reach
#'   Julia).
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A data frame of class `model_out_tbl` in the same shape as the
#'   input, with `model_id` set as requested.
#' @examples
#' \dontrun{
#' # Quantile-input path: each model contributes a 5-quantile forecast; the
#' # linear pool reconstructs each CDF and inverts the mixture exactly.
#' df <- expand.grid(
#'   model_id       = c("m1", "m2"),
#'   output_type_id = c(0.1, 0.25, 0.5, 0.75, 0.9),
#'   location       = "A",
#'   stringsAsFactors = FALSE
#' )
#' df$output_type <- "quantile"
#' df$value <- qnorm(df$output_type_id,
#'                   mean = ifelse(df$model_id == "m1", 0, 2),
#'                   sd   = 1)
#' linear_pool(df, task_id_cols = "location")
#' }
#' @export
linear_pool <- function(model_out_tbl,
                        weights = NULL,
                        n_samples = 10000L,
                        task_id_cols,
                        model_id = "hub-ensemble",
                        seed = NULL) {
  .validate_forecast_df(model_out_tbl, "model_out_tbl")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, model_out_tbl)
  .validate_weights(weights)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_linear_pool")
  out <- fn(as.data.frame(model_out_tbl),
            as.list(task_id_cols),
            as.integer(n_samples),
            if (is.null(weights)) NULL else as.data.frame(weights),
            if (is.null(seed)) NULL else as.integer(seed))
  res <- .julia_to_df(out)
  res$model_id <- model_id
  .as_model_out_tbl(res)
}
