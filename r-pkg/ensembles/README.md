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

The wrapper uses [JuliaConnectoR](https://github.com/stefan-m-lenz/JuliaConnectoR),
which talks to a single long-lived Julia process over a local TCP socket.
First call in a session pays a Julia + Ensembles.jl startup of ~10–15 s;
subsequent calls in the same session are sub-second.

We initially tried [JuliaCall](https://github.com/Non-Contradiction/JuliaCall)
(in-process embedding via `libjulia`), but it segfaults on `using
Ensembles` whenever the embedded Julia loads `RCall.jl`. RCall.jl maps R's
own `R_CStackLimit` symbol via `unsafe_store!`, and inside R-with-embedded-
Julia the same `libR` is already in the process — the resulting symbol
collision corrupts the stack-limit pointer and the next allocation
segfaults. JuliaConnectoR avoids this by keeping Julia and R in separate
processes.

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
