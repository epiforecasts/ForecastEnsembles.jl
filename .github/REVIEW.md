# Reviewing a change to ForecastEnsembles.jl

What to look for in a change to ForecastEnsembles.jl specifically — a Julia
package that combines probabilistic forecasts (quantiles, samples, CDFs, summary
statistics) from several component models, via `fit`/`combine` over the
`MixtureEnsemble`/`QuantileEnsemble` types with multiple dispatch on
`(output_type, method)`. The reviewing method — how to scope a review, what
counts as a finding, how to report and suggest the fix, the trust rules — is the
org half of this spec (`epiforecasts/.github` → `REVIEW.md`), and a review
follows both.

The package is experimental and its forecast data types are under active
redesign, so churn in internal representations is expected; do not hold a merge
on API instability itself.

## What to look for

- **Combination maths**: weights non-negative and normalised to sum to 1 where
  the method requires it; trimmed/windowed selection keeping the right
  components; degenerate inputs handled — a single component, all-equal
  forecasts, a zero or `NaN` weight. Combined **quantiles must stay monotonic**
  (no crossing) and keep their levels sorted; tail interpolation (`distfromq`)
  behaves at the extremes.
- **Pool numerics**: linear vs logarithmic opinion pool — log-pool underflow and
  the renormalisation step; mixture and quantile representations kept mutually
  consistent.
- **Estimation backends**: QRA's LP (JuMP/HiGHS) and CRPS-stacking (Optim.jl)
  set up with the right objective and constraints; an infeasible/unbounded LP or
  a non-converged optimise is caught rather than returning a silent garbage
  weight vector; the score optimised matches the method.
- **Output-type dispatch**: a new or changed `(output_type, method)` pair
  dispatches to the intended algorithm and does not fall through to a wrong
  generic method; a method claiming to cover an `output_type` actually does, or
  errors clearly where it does not; components with differing quantile levels or
  horizons are aligned before combining; `missing`/`NaN` forecasts handled.
- **Type stability and input mutation**: the suite runs JET, so a change should
  not introduce `Any`-typed returns or new dynamic dispatch on the compute path;
  and `fit`/`combine` should treat a caller's array or `DataFrame` as read-only
  rather than mutating it in place.
- **Package surface**: `Project.toml` `[compat]` updated when a dependency or a
  newly-used feature of one needs it; exports match what is genuinely public,
  with docstrings. (The `test/package` Aqua suite catches method ambiguities,
  unbound type parameters and undeclared dependencies on its own — read its
  failure rather than re-deriving it.)
- **The R wrapper (`r-pkg/`, `interop.jl`)**: when a Julia signature or type
  changes, the R interface is kept in step, and conversions across the boundary
  (quantile tables, weights) round-trip correctly.
- **Breaking changes to defaults**: a changed default value or method behaviour
  silently changes results for existing callers. Flag one that is not called out
  as breaking, and check whether the R wrapper's defaults still agree with the
  Julia ones.
- **Reference results**: `test/reference` and `test_backtest.jl` are regenerated
  only when combination output legitimately changes — a regenerated fixture is
  how a real regression gets papered over.
