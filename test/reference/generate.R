# Generate parity fixtures from the R packages so the Julia equivalents can be
# regression-tested against them. Run from the repo root:
#
#     Rscript test/reference/generate.R
#
# Each fixture is split into an `_input.csv` (the data passed to the R
# function) and an `_output.csv` (the result the R function returned).

suppressPackageStartupMessages({
  library(data.table)
  library(scoringutils)
  library(hubEnsembles)
  library(qrensemble)
})

set.seed(2026)
out_dir <- file.path("test", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Fixture 1: hubEnsembles::simple_ensemble — mean and median, weighted/unweighted
# ---------------------------------------------------------------------------

quantile_levels <- c(0.05, 0.25, 0.5, 0.75, 0.95)
locations <- c("A", "B")
horizons <- c(1, 2)

build_quantile_tbl <- function(seed = 1) {
  set.seed(seed)
  rows <- list()
  for (m in c("m1", "m2", "m3")) {
    for (loc in locations) {
      for (h in horizons) {
        offset <- rnorm(1)
        # Dispersion increases with horizon to look forecast-y.
        sd <- 1 + 0.3 * h
        for (q in quantile_levels) {
          rows[[length(rows) + 1L]] <- data.table(
            model_id       = m,
            output_type    = "quantile",
            output_type_id = q,
            location       = loc,
            horizon        = h,
            value          = qnorm(q, mean = offset, sd = sd)
          )
        }
      }
    }
  }
  rbindlist(rows)
}

mot <- build_quantile_tbl(1)
class(mot) <- c("model_out_tbl", class(mot))

fwrite(mot, file.path(out_dir, "simple_input.csv"))

# Mean ensemble
mean_out <- as.data.table(simple_ensemble(mot, agg_fun = "mean"))
fwrite(mean_out, file.path(out_dir, "simple_mean_output.csv"))

# Median ensemble
median_out <- as.data.table(simple_ensemble(mot, agg_fun = "median"))
fwrite(median_out, file.path(out_dir, "simple_median_output.csv"))

# Weighted mean ensemble
weights <- data.table(model_id = c("m1", "m2", "m3"), weight = c(0.5, 0.3, 0.2))
fwrite(weights, file.path(out_dir, "simple_weights.csv"))
wmean_out <- as.data.table(simple_ensemble(mot, weights = weights, agg_fun = "mean"))
fwrite(wmean_out, file.path(out_dir, "simple_wmean_output.csv"))

# ---------------------------------------------------------------------------
# Fixture 2: hubEnsembles::linear_pool — sample-input variant (deterministic)
# ---------------------------------------------------------------------------

build_sample_tbl <- function(seed = 2, K = 200) {
  set.seed(seed)
  rows <- list()
  for (m in c("m1", "m2", "m3")) {
    mu <- rnorm(1)
    for (loc in locations) {
      smp <- rnorm(K, mean = mu, sd = 1)
      for (k in seq_len(K)) {
        rows[[length(rows) + 1L]] <- data.table(
          model_id       = m,
          output_type    = "sample",
          output_type_id = as.character(k),
          location       = loc,
          value          = smp[k]
        )
      }
    }
  }
  rbindlist(rows)
}

smp <- build_sample_tbl(2)
class(smp) <- c("model_out_tbl", class(smp))
fwrite(smp, file.path(out_dir, "lp_sample_input.csv"))

# hubEnsembles::linear_pool on sample input requires equal weights, so we
# only generate the unweighted-sample fixture here (the weighted-sample case
# has no exact R analog).
set.seed(11)
lp_out <- as.data.table(linear_pool(
  smp,
  n_output_samples = 150,
  compound_taskid_set = c("location"),
))
fwrite(lp_out, file.path(out_dir, "lp_sample_output.csv"))

# linear_pool on quantile input — exercises distfromq via hubEnsembles.
mot_q <- build_quantile_tbl(seed = 41)
class(mot_q) <- c("model_out_tbl", class(mot_q))
fwrite(mot_q, file.path(out_dir, "lp_quantile_input.csv"))
set.seed(13)
lp_q_out <- as.data.table(linear_pool(
  mot_q,
  n_output_samples = 10000,
  compound_taskid_set = c("location", "horizon"),
))
fwrite(lp_q_out, file.path(out_dir, "lp_quantile_output.csv"))

# ---------------------------------------------------------------------------
# Fixture 3: qrensemble::qra — basic configurations
# ---------------------------------------------------------------------------

# `qra` uses one target_date as the holdout; everything else is training.
target_date_holdout <- as.Date("2024-01-01") + 70   # last point in series

build_su_full <- function(seed = 3, N = 70) {
  set.seed(seed)
  taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  rows <- list()
  for (loc in c("A", "B")) {
    for (h in c(1L, 2L)) {
      y <- rnorm(N)
      for (m in c("m_good", "m_noisy")) {
        pred <- if (m == "m_good") y + 0.4 * rnorm(N) else 2 * rnorm(N)
        for (t in seq_len(N)) {
          for (q in taus) {
            rows[[length(rows) + 1L]] <- data.table(
              model = m,
              quantile_level = q,
              predicted = pred[t] + qnorm(q),
              observed = y[t],
              location = loc,
              horizon = h,
              target_date = as.Date("2024-01-01") + t
            )
          }
        }
      }
    }
  }
  rbindlist(rows)
}

su_full <- build_su_full()
fwrite(su_full, file.path(out_dir, "qra_input.csv"))
fwrite(data.table(target_date = target_date_holdout),
       file.path(out_dir, "qra_target.csv"))

fc <- as_forecast_quantile(
  su_full,
  forecast_unit = c("model", "location", "horizon", "target_date"),
)
target_filter <- list(target_date = target_date_holdout)

# Default qra (per_quantile_weights = FALSE, enforce_normalisation = TRUE,
# intercept = FALSE, noncross = TRUE).
qra_default <- qra(fc, target = target_filter)
fwrite(as.data.table(qra_default),
       file.path(out_dir, "qra_default_output.csv"))
fwrite(attr(qra_default, "weights"),
       file.path(out_dir, "qra_default_weights.csv"))
ic <- attr(qra_default, "intercept")
if (is.null(ic)) ic <- data.table()
fwrite(ic, file.path(out_dir, "qra_default_intercepts.csv"))

# Per-quantile weights, no normalisation, with intercept, no noncross.
qra_perq <- qra(fc,
                target = target_filter,
                per_quantile_weights = TRUE,
                enforce_normalisation = FALSE,
                intercept = TRUE,
                noncross = FALSE)
fwrite(as.data.table(qra_perq),
       file.path(out_dir, "qra_perq_output.csv"))
fwrite(attr(qra_perq, "weights"),
       file.path(out_dir, "qra_perq_weights.csv"))
ic <- attr(qra_perq, "intercept")
if (is.null(ic)) ic <- data.table()
fwrite(ic, file.path(out_dir, "qra_perq_intercepts.csv"))

cat("Fixtures written to", out_dir, "\n")
