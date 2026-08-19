# Logarithmic (geometric) opinion pool: the ensemble density is the normalised
# weighted product of the member densities,
#
#     f_ens(x) ∝ Πᵢ fᵢ(x)^wᵢ,
#
# the geometric-mean counterpart of the linear pool's arithmetic mean of
# densities ([`MixtureEnsemble`](@ref)/[`LinearPool`](@ref)). It is a product of
# experts: mass survives only where every weighted member puts mass, so the pool
# is sharper than the linear pool and precision-weights toward the tightest
# member. Unlike the linear pool it has no closed form for reconstructed quantile
# forecasts, so the density product is formed on a grid, renormalised, and
# inverted — which also keeps the output quantiles monotone by construction.

"""
    LogarithmicPool(; weights = nothing, ngrid = 2000)

Logarithmic (geometric) opinion pool of quantile forecasts: the ensemble is the
normalised weighted product of the member densities, `f_ens(x) ∝ Πᵢ fᵢ(x)^wᵢ`.

The geometric-mean counterpart of [`LinearPool`](@ref) (which averages the
densities): the log pool is a product of experts, so it concentrates where the
members agree and is typically sharper than the linear pool. `weights` are
per-model (equal by default, or any per-model [`EnsembleWeights`](@ref) / fitted
method) and act as the density exponents.

Each member's density is reconstructed from its quantiles (PCHIP interior,
Normal tails), the log-product is formed on a grid of `ngrid` points,
renormalised to integrate to one, and inverted at the requested levels. Quantile
forecasts only.

# Fields

- `weights`: per-model exponent weights, or `nothing` for equal weights.
- `ngrid`: number of grid points for the density product (higher is more
  accurate but slower).
"""
struct LogarithmicPool <: UnfittedMethod
    weights::Union{Nothing, EnsembleWeights}
    ngrid::Int
end

function LogarithmicPool(; weights = nothing, ngrid::Integer = 2000)
    w = _resolve_weights(weights)
    if w !== nothing && is_per_quantile(w)
        throw(ArgumentError("LogarithmicPool takes per-model weights only; " *
                            "per-quantile weights are not meaningful for a product pool."))
    end
    ngrid >= 100 || throw(ArgumentError("ngrid must be >= 100 (got $ngrid)"))
    return LogarithmicPool(w, Int(ngrid))
end

"""
    combine(ft::ForecastTable, m::LogarithmicPool; rng = default_rng()) -> ForecastTable

Apply the logarithmic (geometric) opinion pool at each task. See
[`LogarithmicPool`](@ref). The output `model_id` is `"hub-ensemble"`.
"""
function combine(ft::ForecastTable, m::LogarithmicPool; rng::AbstractRNG = default_rng())
    output_type(ft) === :quantile ||
        throw(ArgumentError("LogarithmicPool supports :quantile forecasts."))
    wdict = _weights_vector(m.weights, ft, ft.data)

    out_groups = DataFrame[]
    for tg in DataFrames.groupby(ft.data, ft.task_id_cols)
        levels = sort(unique(tg.output_type_id))
        models = unique(tg[!, ft.model_id_col])
        ws = [wdict[mod] for mod in models]
        ws ./= sum(ws)

        dists = Vector{QuantileDistribution}(undef, length(models))
        for sub in DataFrames.groupby(tg, ft.model_id_col)
            i = findfirst(==(first(sub[!, ft.model_id_col])), models)
            s = sort(sub, :output_type_id)
            dists[i] = QuantileDistribution(Float64.(s.output_type_id), Float64.(s.value))
        end

        out = DataFrame(tg[1:1, ft.task_id_cols])
        out = repeat(out, length(levels))
        out.output_type = fill(:quantile, length(levels))
        out.output_type_id = levels
        out.value = _log_pool_quantiles(dists, ws, Float64.(levels), m.ngrid)
        out[!, ft.model_id_col] .= "hub-ensemble"
        push!(out_groups, out)
    end
    out = reduce(vcat, out_groups)
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(
        out;
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

# Grid-based geometric pool: reconstruct log Σᵢ wᵢ fᵢ on a grid, renormalise,
# integrate to a CDF, and invert at the requested levels.
function _log_pool_quantiles(
        dists::Vector{QuantileDistribution},
        ws::AbstractVector{<:Real},
        levels::AbstractVector{<:Real},
        ngrid::Int
)
    # Key the grid to the requested levels, not a fixed 1e-4 tail: a level more
    # extreme than 1e-4 would otherwise fall outside the grid and be silently
    # clamped to its edge. `plo`/`phi` never shrink the default tail.
    plo = min(1.0e-4, minimum(levels))
    phi = max(1 - 1.0e-4, maximum(levels))
    lo = minimum(quantile(d, plo) for d in dists)
    hi = maximum(quantile(d, phi) for d in dists)
    hi > lo || throw(ArgumentError(
        "logarithmic pool: the combined forecast range is degenerate (the span " *
        "between the lowest and highest requested-tail percentile across components " *
        "is zero, lo = hi = $lo), so no integration grid can be built."))
    pad = 0.05 * (hi - lo)
    xs = range(lo - pad, hi + pad; length = ngrid)
    dx = step(xs)

    logdens = zeros(ngrid)
    F = Vector{Float64}(undef, ngrid)
    dens = Vector{Float64}(undef, ngrid)
    for (d, w) in zip(dists, ws)
        @inbounds for j in 1:ngrid
            F[j] = cdf(d, xs[j])
        end
        dens[1] = (F[2] - F[1]) / dx
        dens[ngrid] = (F[ngrid] - F[ngrid - 1]) / dx
        @inbounds for j in 2:(ngrid - 1)
            dens[j] = (F[j + 1] - F[j - 1]) / (2dx)
        end
        @. logdens += w * log(max(dens, 1e-300))
    end

    logdens .-= maximum(logdens)
    g = exp.(logdens)
    # Reach check: with the grid keyed to the requested levels the pooled density
    # should be negligible at the edges. If it is not, the grid is too narrow to
    # hold the full mass, so renormalisation would inflate it and the extreme
    # quantiles would be truncated — flag it rather than fail silently.
    edge = max(g[1], g[ngrid])        # relative to the unit peak
    edge < 1.0e-3 || @warn "logarithmic pool: the density is not negligible at " *
          "the integration-grid edge (relative $(round(edge; sigdigits = 2))), so the " *
          "most extreme requested quantiles may be truncated; raise `ngrid`." maxlog = 1
    g ./= _trapz(g, dx)               # normalise to a density

    # cumulative-trapezoid CDF, forced to [0, 1]
    cum = similar(g)
    cum[1] = 0.0
    @inbounds for j in 2:ngrid
        cum[j] = cum[j - 1] + dx * (g[j] + g[j - 1]) / 2
    end
    cum ./= cum[end]

    return [_invert_grid(xs, cum, τ) for τ in levels]
end

_trapz(g::AbstractVector, dx::Real) = dx * (sum(g) - 0.5 * (g[1] + g[end]))

function _invert_grid(xs::AbstractVector, cum::AbstractVector, τ::Real)
    τ <= cum[1] && return first(xs)
    τ >= cum[end] && return last(xs)
    j = searchsortedlast(cum, τ)
    cum[j + 1] == cum[j] && return xs[j]
    frac = (τ - cum[j]) / (cum[j + 1] - cum[j])
    return xs[j] + frac * (xs[j + 1] - xs[j])
end
