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

# Mean weighted-sample CRPS of an ensemble `fc` against observations `o` (joined
# on :t): score each task then average — the scorer shape `backtest` expects.
# `include`d into each backtest testitem, which brings `DataFrame` and `mean`
# into scope alongside `crps` above.
function sample_crps(fc, o)
    d = DataFrames.innerjoin(DataFrame(fc), o; on = :t)
    per = DataFrames.combine(DataFrames.groupby(d, :t),
        [:value, :observed] => ((v, y) -> crps(Float64.(v), Float64(first(y)))) => :s)
    return mean(per.s)
end

# Pinball (quantile) loss at each level — the WIS kernel. Returns one value per
# level, matching ScoringRules' `quantile_score(levels, forecasts, y)` shape.
function quantile_score(levels::AbstractVector, forecasts::AbstractVector, y::Real)
    return [(y - forecasts[i]) * (levels[i] - (y < forecasts[i] ? 1.0 : 0.0))
            for i in eachindex(levels)]
end

# Shared backtest fixture: a non-stationary regime where model A is sharp in the
# first half of the time window and model B in the second. A scheme that learns
# weights from recent performance should beat equal weighting out of sample.
# `include`d into each backtest testitem, which brings `DataFrame`, `MersenneTwister`
# and `ForecastTable` into scope.
function _bt_sample_data(; T = 30, K = 80, seed = 4)
    rng = MersenneTwister(seed)
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for t in 1:T
        y = obs.observed[t]
        a_good = t <= T ÷ 2
        for (mid, sharp) in (("m_a", a_good), ("m_b", !a_good))
            sd = sharp ? 0.4 : 3.0
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "sample",
                    output_type_id = 1:K,
                    t = t,
                    value = y .+ sd .* randn(rng, K)
                )
            )
        end
    end
    return ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
end
