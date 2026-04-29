# Reconstruct an approximate continuous distribution from a small number of
# quantile pairs (τ_k, q_k), then sample from it. This is the Julia
# counterpart of the bits of the R `distfromq` package that hubEnsembles uses
# for its quantile-input linear-pool path.
#
# v0.1 implementation:
#   - interior: linear interpolation between consecutive quantile knots
#     (a monotone, parameter-free choice)
#   - left tail (u < τ_1):  normal tail matched to (τ_1, τ_2)
#   - right tail (u > τ_K): normal tail matched to (τ_{K-1}, τ_K)
#
# Normal tails mean: pick a Normal(μ, σ) such that its inverse CDF at the two
# innermost tail probabilities equals the corresponding quantile values;
# extrapolate using that Normal.
#
# This matches `distfromq`'s default `tail_dist = "norm"` qualitatively. It is
# not yet PCHIP in the interior — that is a follow-up before claiming
# numerical parity with the R package.

using Distributions: Normal
import Distributions: cdf
import Statistics: quantile
using Random: AbstractRNG, default_rng


"""
    QuantileDistribution(probs, vals)

A 1-D distribution reconstructed from the (probability, value) quantile pairs
`probs` and `vals`. `probs` must be strictly increasing in (0,1); `vals` must
be non-decreasing and the same length.
"""
struct QuantileDistribution
    probs::Vector{Float64}
    vals::Vector{Float64}
    left_tail::Normal{Float64}
    right_tail::Normal{Float64}
end

function QuantileDistribution(probs::AbstractVector, vals::AbstractVector)
    p = collect(Float64.(probs))
    v = collect(Float64.(vals))
    length(p) == length(v) || throw(ArgumentError("probs and vals must be the same length"))
    length(p) >= 2 || throw(ArgumentError("need at least two quantile pairs"))
    issorted(p) || throw(ArgumentError("probs must be increasing"))
    all(0 .< p .< 1) || throw(ArgumentError("probs must lie strictly in (0,1)"))
    issorted(v) || throw(ArgumentError("vals must be non-decreasing"))

    lt = _fit_normal_tail(p[1], v[1], p[2], v[2])
    rt = _fit_normal_tail(p[end-1], v[end-1], p[end], v[end])
    return QuantileDistribution(p, v, lt, rt)
end

# Solve for (μ, σ) such that Φ((v_i − μ)/σ) = p_i for i = 1, 2.
function _fit_normal_tail(p1, v1, p2, v2)
    z1 = quantile(Normal(0, 1), p1)
    z2 = quantile(Normal(0, 1), p2)
    if v1 == v2 || z1 == z2
        # Degenerate: collapse to a near-point mass tail.
        return Normal(v1, eps(typeof(v1)))
    end
    σ = (v2 - v1) / (z2 - z1)
    μ = v1 - z1 * σ
    return Normal(μ, σ)
end

"""
    quantile(d::QuantileDistribution, u)

Return the value of `d` at probability `u ∈ (0,1)`.
"""
function quantile(d::QuantileDistribution, u::Real)
    p = d.probs; v = d.vals
    if u <= p[1]
        return quantile(d.left_tail, u)
    elseif u >= p[end]
        return quantile(d.right_tail, u)
    else
        i = searchsortedlast(p, u)
        # Interior linear interpolation.
        if p[i+1] == p[i]
            return v[i]
        end
        t = (u - p[i]) / (p[i+1] - p[i])
        return v[i] + t * (v[i+1] - v[i])
    end
end

"""
    cdf(d::QuantileDistribution, x)

Return P(X ≤ x). Inverse of [`Base.quantile(::QuantileDistribution, ::Real)`](@ref).
"""
function cdf(d::QuantileDistribution, x::Real)
    p = d.probs; v = d.vals
    if x <= v[1]
        return cdf(d.left_tail, x)
    elseif x >= v[end]
        return cdf(d.right_tail, x)
    else
        i = searchsortedlast(v, x)
        if v[i+1] == v[i]
            return p[i]
        end
        t = (x - v[i]) / (v[i+1] - v[i])
        return p[i] + t * (p[i+1] - p[i])
    end
end

"""
    rand(rng, d::QuantileDistribution, n)

Draw `n` samples from `d` by inverse-CDF sampling.
"""
function Base.rand(rng::AbstractRNG, d::QuantileDistribution, n::Integer)
    return [quantile(d, rand(rng)) for _ in 1:n]
end

Base.rand(d::QuantileDistribution, n::Integer) = rand(default_rng(), d, n)
