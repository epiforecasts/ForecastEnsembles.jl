#' Beta-transformed linear pool (BLP)
#'
#' R wrapper for the beta-transformed linear pool. Fits a Beta recalibration on
#' the probability-integral-transform (PIT) values of the linear pool over the
#' `training` forecasts and `observations`, then applies that recalibration to
#' the `target` forecasts. Quantile forecasts only. The fit and combination run
#' in Julia.
#'
#' BLP is a *recalibration* of the pool rather than a re-weighting, so — unlike
#' [qra()] — it returns no per-model weights.
#'
#' @param training A data frame with columns `model_id`, `output_type` (must be
#'   `"quantile"`), `output_type_id`, `value`, plus task-id columns.
#' @param target A data frame in the same shape: the forecasts to recalibrate
#'   using the fitted transform.
#' @param observations A data frame with the task-id columns plus a column named
#'   exactly `observed`.
#' @param task_id_cols Character vector of the column names that identify a
#'   forecast task, e.g. `c("location", "horizon", "target_date")`.
#' @param weights Optional data frame with columns `model_id` and `weight`
#'   giving the underlying pool weights; equal weights if omitted. Weights need
#'   not sum to 1; they are normalised internally.
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A data frame of recalibrated predictions on `target`, one row per
#'   (task, output_type_id).
#' @examples
#' \dontrun{
#' taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
#' n <- 50
#' set.seed(1)
#' y <- rnorm(n + 1)
#' make_rows <- function(model_id, predictions) {
#'   do.call(rbind, lapply(taus, function(q) data.frame(
#'     model_id = model_id, output_type = "quantile",
#'     output_type_id = q, t = seq_along(predictions),
#'     value = predictions + qnorm(q), stringsAsFactors = FALSE
#'   )))
#' }
#' rows <- rbind(make_rows("m1", y + 0.3 * rnorm(n + 1)),
#'               make_rows("m2", y + rnorm(n + 1)))
#' train  <- rows[rows$t <= n, ]
#' target <- rows[rows$t == n + 1, ]
#' obs <- data.frame(t = seq_len(n), observed = y[seq_len(n)])
#' blp(training = train, target = target, observations = obs, task_id_cols = "t")
#' }
#' @export
blp <- function(training,
                target,
                observations,
                task_id_cols,
                weights = NULL) {
  .validate_forecast_df(training, "training")
  .validate_forecast_df(target, "target")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, training, "training")
  .validate_observations(observations, task_id_cols)
  .validate_weights(weights)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_blp")
  out <- fn(as.data.frame(training),
            as.data.frame(target),
            as.data.frame(observations),
            as.list(task_id_cols),
            if (is.null(weights)) NULL else as.data.frame(weights))
  .julia_to_df(out)
}
