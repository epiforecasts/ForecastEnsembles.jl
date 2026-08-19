#' Quantile Regression Averaging
#'
#' R wrapper that mirrors `qrensemble::qra`. Fits a quantile regression on
#' training forecasts and applies the fitted weights to the target
#' forecasts.
#'
#' @section Migrating from qrensemble:
#' `qrensemble::qra` takes a single `forecast_quantile` object containing
#' both training and holdout rows plus a `target` filter; this function
#' takes three explicit data frames. To convert: rename `model` to
#' `model_id`, `quantile_level` to `output_type_id` and `predicted` to
#' `value`; add an `output_type = "quantile"` column; split the rows into
#' `training` and `target` yourself (e.g. on `target_date`); and put the
#' observed values into a separate `observations` frame with the task-id
#' columns plus a column named exactly `observed`.
#'
#' The defaults (`per_quantile_weights = FALSE`,
#' `enforce_normalisation = TRUE`, `intercept = FALSE`, `noncross = TRUE`)
#' match `qrensemble::qra`, so a tuned call carries over unchanged.
#'
#' @section Default differences from the Julia API:
#' `ForecastEnsembles.QRA` in Julia defaults `noncross = false`, where this
#' wrapper defaults to `TRUE` to stay compatible with `qrensemble::qra`. The
#' two produce identical results: `noncross` does nothing unless
#' `per_quantile_weights = TRUE`, and both default that to off. Every other
#' default matches the Julia constructor.
#'
#' @param training A data frame with columns `model_id`, `output_type`
#'   (must be `"quantile"`), `output_type_id`, `value`, plus task-id
#'   columns.
#' @param target A data frame in the same shape: the forecasts to combine
#'   using the fitted weights.
#' @param observations A data frame with the task-id columns plus a column
#'   named exactly `observed`.
#' @param task_id_cols Character vector of the column names that identify
#'   a forecast task, e.g. `c("location", "horizon", "target_date")`.
#' @param per_quantile_weights If TRUE, fit separate weights per quantile.
#' @param intercept Include an intercept term in the regression.
#' @param enforce_normalisation Constrain weights to lie on the simplex.
#' @param noncross Add cross-quantile monotonicity constraints. Only takes
#'   effect when `per_quantile_weights = TRUE`; without effect otherwise, as
#'   the joint fit cannot cross by construction when
#'   `enforce_normalisation = TRUE`.
#' @param group Character vector of task dimensions over which to fit
#'   separate regressions (e.g. `"location"` for per-location weights).
#'   Leave empty to fit a single global model across all tasks.
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A data frame of fitted predictions on `target`. When the fit
#'   reduces to a clean weight vector (the default configuration does),
#'   the weights are attached as a data frame in `attr(result, "weights")`,
#'   mirroring `qrensemble`.
#' @examples
#' \dontrun{
#' # 50 training points, 1 holdout, 2 component models, 5 quantile levels.
#' set.seed(1)
#' taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
#' n <- 50
#' y <- rnorm(n + 1)
#' make_rows <- function(model_id, predictions) {
#'   do.call(rbind, lapply(taus, function(q) data.frame(
#'     model_id = model_id, output_type = "quantile",
#'     output_type_id = q, t = seq_along(predictions),
#'     value = predictions + qnorm(q),
#'     stringsAsFactors = FALSE
#'   )))
#' }
#' rows <- rbind(
#'   make_rows("m_good",  y + 0.3 * rnorm(n + 1)),
#'   make_rows("m_noisy", 2 * rnorm(n + 1))
#' )
#' train  <- rows[rows$t <= n, ]    # tasks 1..50 train
#' target <- rows[rows$t == n + 1, ] # task 51 is the holdout
#' obs <- data.frame(t = seq_len(n), observed = y[seq_len(n)])
#'
#' result <- qra(training = train, target = target, observations = obs,
#'               task_id_cols = "t")
#' attr(result, "weights")
#' }
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
  .validate_forecast_df(training, "training")
  .validate_forecast_df(target, "target")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, training, "training")
  .validate_observations(observations, task_id_cols)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_qra")
  out <- fn(as.data.frame(training),
            as.data.frame(target),
            as.data.frame(observations),
            as.list(task_id_cols),
            isTRUE(per_quantile_weights),
            isTRUE(intercept),
            isTRUE(enforce_normalisation),
            # Only forward `noncross` where it can act. The shared-weight fit is
            # already monotone in tau, so the Julia constructor warns that the
            # flag does nothing -- which on a default call would be a warning
            # about an argument the caller never set.
            isTRUE(noncross) && isTRUE(per_quantile_weights),
            as.list(group))
  res <- JuliaConnectoR::juliaGet(out)
  pred <- as.data.frame(res$pred, stringsAsFactors = FALSE)
  if (!is.null(res$weights)) {
    attr(pred, "weights") <- as.data.frame(res$weights,
                                           stringsAsFactors = FALSE)
  }
  pred
}
