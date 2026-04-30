#' Quantile Regression Averaging
#'
#' Equivalent to `qrensemble::qra`. Fits a quantile regression on training
#' forecasts and applies the fitted weights to the target forecasts.
#'
#' @param training A data frame with columns `model_id`, `output_type`
#'   (must be `"quantile"`), `output_type_id`, `value`, plus task-id columns.
#' @param target A data frame in the same shape — the forecasts to combine
#'   using the fitted weights.
#' @param observations A data frame with task-id columns plus `observed`.
#' @param task_id_cols Character vector of task-id columns.
#' @param per_quantile_weights If TRUE, fit separate weights per quantile.
#' @param intercept Include an intercept term in the regression.
#' @param enforce_normalisation Constrain weights to lie on the simplex.
#' @param noncross Add cross-quantile monotonicity constraints (only used
#'   when `per_quantile_weights = TRUE`).
#' @param group Character vector of task dimensions over which to fit
#'   separate regressions.
#'
#' @return A data frame of fitted predictions on `target`.
#' @export
qra <- function(training,
                target,
                observations,
                task_id_cols,
                per_quantile_weights = FALSE,
                intercept = FALSE,
                enforce_normalisation = TRUE,
                noncross = TRUE,
                group = character(0)) {
  if (missing(task_id_cols))
    stop("`task_id_cols` is required.", call. = FALSE)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_qra")
  out <- fn(as.data.frame(training),
            as.data.frame(target),
            as.data.frame(observations),
            as.list(task_id_cols),
            isTRUE(per_quantile_weights),
            isTRUE(intercept),
            isTRUE(enforce_normalisation),
            isTRUE(noncross),
            as.list(group))
  .julia_to_df(out)
}
