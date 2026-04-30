# ensembles (R)

Thin R wrapper around the Julia package `Ensembles.jl`. Mirrors the user-
facing API of `hubEnsembles`, `qrensemble`, and `lopensemble`:

| R wrapper                            | Equivalent in R packages                       |
|--------------------------------------|------------------------------------------------|
| `simple_ensemble()`                  | `hubEnsembles::simple_ensemble`                |
| `linear_pool()`                      | `hubEnsembles::linear_pool`                    |
| `qra()`                              | `qrensemble::qra`                              |
| `crps_weights()`                     | `lopensemble::crps_weights`                    |
| `mixture_from_samples()`             | `lopensemble::mixture_from_samples`            |

## How it works

Each public function spawns a short-lived Julia subprocess that runs the
operation through Ensembles.jl. Inputs go in as CSV, results come back as
CSV. There is one Julia startup per call (≈10–15 s on first use, ≈5–8 s
once Ensembles.jl is precompiled in the bridge environment).

We started with [JuliaCall](https://github.com/Non-Contradiction/JuliaCall)
(in-process embedding) but it segfaults on `using Ensembles` with Julia
1.12 due to native-library loading issues. The subprocess approach adds
startup cost but is reliable across Julia versions.

## Requirements

- Julia ≥ 1.10 on `PATH` (or pass `julia_bin` to `julia_setup()`).
- A working installation of this package (`R CMD INSTALL r-pkg/ensembles`).

The first call in a session also instantiates the bridge Julia project
(LBFGS solver, HiGHS LP solver, CSV/JSON IO). This takes a few minutes the
very first time and then stays cached.

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

- Persistent Julia daemon (DaemonMode.jl) to remove per-call startup cost.
- Optional in-process binding once JuliaCall is fixed for Julia 1.12+.
