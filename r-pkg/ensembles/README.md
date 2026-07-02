# ensembles (R)

<!-- badges: start -->
[![R-CI](https://github.com/sbfnk/ensembles.jl/actions/workflows/R-CI.yml/badge.svg)](https://github.com/sbfnk/ensembles.jl/actions/workflows/R-CI.yml)
<!-- badges: end -->

A thin R wrapper around the Julia package `Ensembles.jl`. The user-facing
functions mirror the originals:

| R wrapper                | Original                              |
|--------------------------|---------------------------------------|
| `simple_ensemble()`      | `hubEnsembles::simple_ensemble`       |
| `linear_pool()`          | `hubEnsembles::linear_pool`           |
| `qra()`                  | `qrensemble::qra`                     |
| `crps_weights()`         | `lopensemble::crps_weights`           |
| `mixture_from_samples()` | `lopensemble::mixture_from_samples`   |

## How it works

The wrapper talks to a long-lived Julia process via
[JuliaConnectoR](https://github.com/stefan-m-lenz/JuliaConnectoR). The
first call in a session takes ~10–15 s while Julia and Ensembles.jl warm
up; calls after that run in under a second.

## Requirements

- Julia ≥ 1.10 on `PATH` (or pass `julia_bindir` to `julia_setup()`).
- `R CMD INSTALL r-pkg/ensembles`.

The first call also instantiates the bridge Julia project (LBFGS, HiGHS,
CSV/JSON IO). That takes a few minutes, then it stays cached.

## Quick start

```r
library(ensembles)

df <- data.frame(
  model_id = rep(c("m1", "m2"), each = 4),
  output_type = "quantile",
  output_type_id = rep(c(0.1, 0.25, 0.75, 0.9), 2),
  location = "A",
  value = c(-1.5, -0.5, 0.5, 1.5,   # m1
            -1.0,  0.0, 1.0, 2.0)   # m2
)

simple_ensemble(df, agg_fun = "mean", task_id_cols = "location")
```

## Future work

- Persistent Julia daemon (DaemonMode.jl) so the first call is cheaper.
- In-process binding when JuliaCall supports Julia 1.12+.
