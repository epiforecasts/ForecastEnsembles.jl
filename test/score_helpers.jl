# Dependency-free scoring helpers for the tests. The score-driven estimators
# (Stacking, InverseScore, Hedge, PartialPooling) and `backtest` take any
# `score(samples, y; w)` / scorer callable; these local implementations exercise
# that interface without depending on ScoringRules (the documented companion),
# which keeps the test environment free of an unregistered git-pinned package.

# Weighted empirical (energy-form) CRPS of a sample forecast `dat` at `y`:
# CRPS = Σᵢ wᵢ|xᵢ − y| − ½ Σᵢⱼ wᵢ wⱼ |xᵢ − xⱼ|, with weights normalised. Accepts a
# per-sample weight vector `w` (as the estimators pass) or none (equal weights).
function crps(dat::AbstractVector, y::Real; w = nothing)
    n = length(dat)
    ww = w === nothing ? fill(1.0 / n, n) : w ./ sum(w)
    ex = sum(ww[i] * abs(dat[i] - y) for i in 1:n)
    ee = 0.0
    for i in 1:n, j in 1:n

        ee += ww[i] * ww[j] * abs(dat[i] - dat[j])
    end
    return ex - 0.5 * ee
end

# Pinball (quantile) loss at each level — the WIS kernel. Returns one value per
# level, matching ScoringRules' `quantile_score(levels, forecasts, y)` shape.
function quantile_score(levels::AbstractVector, forecasts::AbstractVector, y::Real)
    return [(y - forecasts[i]) * (levels[i] - (y < forecasts[i] ? 1.0 : 0.0))
            for i in eachindex(levels)]
end
