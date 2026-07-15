# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# If this file defines `JET_REPORT_FILTER` (a `report -> Bool` predicate; a
# report is kept when it returns `true`), the runner switches from `test_package`
# to `report_package` + filter and fails only on reports the predicate keeps.
#
# `_default_score_fn` is an intentionally method-less hook: the MIT core declares
# it, and the `ForecastEnsemblesScoringRulesExt` extension supplies the only
# method once `using ScoringRules` runs. In this isolated JET environment the
# extension is not loaded, so the call in `backtest` reports "no matching
# method" — a by-design false positive. Drop exactly that report and keep every
# other one.
const JET_REPORT_FILTER = r -> !occursin("_default_score_fn", sprint(show, r))
