# Generate a CRPS-stacking fixture from `lopensemble`. Run from the repo root:
#
#     Rscript test/reference/generate_lopensemble.R
#
# Notes
# -----
# * `crps_weights` MAP-optimises a Stan model. Given a fixed data table and
#   default optimiser settings it is deterministic, so the fixture is
#   reproducible.
# * We pass `lambda = "equal"` to match Julia's untimed CRPS objective.
# * `dirichlet_alpha` is left at the package default (1.001) so the prior is
#   essentially uninformative.

suppressPackageStartupMessages({
  library(data.table)
  library(lopensemble)
})

set.seed(2027)
out_dir <- file.path("test", "reference")

# Two-model setup: m_good predicts y exactly with low noise; m_noisy adds a
# wide error. CRPS-stacking should put nearly all weight on m_good.
T_n <- 30        # tasks (dates)
S_n <- 100       # samples per (model, task)
dates <- seq.Date(as.Date("2024-01-01"), by = "day", length.out = T_n)
y <- rnorm(T_n)

rows <- list()
for (m in c("m_good", "m_noisy")) {
  for (t in seq_len(T_n)) {
    smp <- if (m == "m_good") y[t] + rnorm(S_n, sd = 0.4) else rnorm(S_n, sd = 3)
    for (k in seq_len(S_n)) {
      rows[[length(rows) + 1L]] <- data.table(
        model     = m,
        date      = dates[t],
        sample_id = k,
        predicted = smp[k],
        observed  = y[t]
      )
    }
  }
}
df <- rbindlist(rows)
fwrite(df, file.path(out_dir, "crps_input.csv"))

w <- crps_weights(df, lambda = "equal", dirichlet_alpha = 1.001)
out <- data.table(model = names(w), weight = as.numeric(w))
fwrite(out, file.path(out_dir, "crps_weights_output.csv"))

cat("CRPS fixture written\n")
print(out)
