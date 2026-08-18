# Reconstruct an approximate continuous distribution from a small number of
# quantile pairs (τ_k, q_k), then sample from it. This is the Julia
# counterpart of the bits of the R `distfromq` package that hubEnsembles uses
# for its quantile-input linear-pool path.
#
# Design:
#   - interior: monotone cubic Hermite (PCHIP, Fritsch–Carlson) interpolation
#     between consecutive knots — matches `distfromq`'s default behaviour
#     and keeps the reconstructed distribution monotone (so the quantile
#     function and sampler are well-defined).
#   - left tail (u < τ_1):  normal tail matched to (τ_1, τ_2)
#   - right tail (u > τ_K): normal tail matched to (τ_{K-1}, τ_K)
#
# Two PCHIP splines are stored per distribution: one for the forward map
# u ↦ value (used by `quantile` and sampling), one for the inverse map
# value ↦ u (used by `cdf`). Both must be monotone increasing, which is
# guaranteed by the Fritsch–Carlson construction when the input pairs are
# strictly increasing.

"""
    QuantileDistribution(probs, vals)

A 1-D distribution reconstructed from the (probability, value) quantile pairs
`probs` and `vals`. `probs` must be strictly increasing in (0,1); `vals` must
be non-decreasing and the same length.
"""
struct QuantileDistribution
    probs::Vector{Float64}     # τ_k
    vals::Vector{Float64}      # q_k
    fwd_d::Vector{Float64}     # PCHIP slopes for u ↦ value at each knot
    inv_d::Vector{Float64}     # PCHIP slopes for value ↦ u at each knot
    left_tail::Normal{Float64}
    right_tail::Normal{Float64}
end

function QuantileDistribution(probs::AbstractVector, vals::AbstractVector)
    p = collect(Float64.(probs))
    v = collect(Float64.(vals))
    length(p) == length(v) || throw(ArgumentError("probs and vals must be the same length"))
    length(p) >= 2 || throw(ArgumentError("need at least two quantile pairs"))
    (issorted(p) && allunique(p)) ||
        throw(ArgumentError("probs must be strictly increasing"))
    all(0 .< p .< 1) || throw(ArgumentError("probs must lie strictly in (0,1)"))
    issorted(v) || throw(ArgumentError("vals must be non-decreasing"))

    fwd_d = _pchip_slopes(p, v)
    inv_d = _pchip_slopes(v, p)

    lt = _left_tail(p, v)
    rt = _right_tail(p, v)

    return QuantileDistribution(p, v, fwd_d, inv_d, lt, rt)
end

# Solve for (μ, σ) such that Φ((v_i − μ)/σ) = p_i for i = 1, 2.
function _fit_normal_tail(p1, v1, p2, v2)
    z1 = quantile(Normal(0, 1), p1)
    z2 = quantile(Normal(0, 1), p2)
    if v1 == v2 || z1 == z2
        return Normal(v1, eps(typeof(v1)))
    end
    σ = (v2 - v1) / (z2 - z1)
    μ = v1 - z1 * σ
    return Normal(μ, σ)
end

# Fit the outer Normal tails from the outermost pair of *distinct* knot values.
# When the outer two quantiles tie (e.g. Q(0.1) = Q(0.25) for a count forecast),
# fitting from the tied pair gives a degenerate spike whose median lands at the
# knot, so `cdf` at the boundary wrongly reads 0.5. Reaching to the first distinct
# knot instead makes `cdf(boundary) = p`; a non-tied boundary is unchanged.
# Fully-degenerate input (all values equal) has no distinct knot, so both tails
# fall back to a near-point spike and `cdf` at the value reads ≈ 0.5 — unsupported,
# but no better answer exists from quantiles alone.
function _left_tail(p::AbstractVector, v::AbstractVector)
    lo = firstindex(v)
    j = findnext(k -> v[k] != v[lo], eachindex(v), lo + 1)
    j === nothing && return Normal(v[lo], eps(one(float(v[lo]))))  # fully degenerate
    return _fit_normal_tail(p[lo], v[lo], p[j], v[j])
end

function _right_tail(p::AbstractVector, v::AbstractVector)
    hi = lastindex(v)
    j = findprev(k -> v[k] != v[hi], eachindex(v), hi - 1)
    j === nothing && return Normal(v[hi], eps(one(float(v[hi]))))  # fully degenerate
    return _fit_normal_tail(p[j], v[j], p[hi], v[hi])
end

# Fritsch–Carlson monotone cubic Hermite slopes for (x, y) with strictly
# increasing x and non-decreasing y. Returns the derivative `d_i` at each
# knot. References: Fritsch & Carlson 1980 SIAM JNA.
function _pchip_slopes(x::AbstractVector, y::AbstractVector)
    n = length(x)
    h = diff(x)
    # The inverse map passes quantile *values* as `x`; tied values give a
    # zero-width interval, so `s = Inf` at a tied knot. That is safe: `cdf`/
    # `quantile` never evaluate such a knot (the tail or neighbouring interval
    # covers it), so no `Inf` slope reaches a Hermite evaluation.
    s = diff(y) ./ h
    d = zeros(Float64, n)

    # With only two knots there is a single secant; the monotone Hermite
    # interpolant with d = s at both ends is the straight line through the
    # knots. (The three-point endpoint formulae below need n >= 3.) Both
    # tails are then fitted from the same two pairs, so the distribution is
    # a single Normal with a matching linear mid-section.
    if n == 2
        d[1] = s[1]
        d[2] = s[1]
        return d
    end

    # Endpoints (one-sided three-point estimate, then enforce monotonicity).
    d[1] = ((2h[1] + h[2]) * s[1] - h[1] * s[2]) / (h[1] + h[2])
    if sign(d[1]) != sign(s[1])
        d[1] = 0.0
    elseif sign(s[1]) != sign(s[2]) && abs(d[1]) > abs(3 * s[1])
        d[1] = 3 * s[1]
    end

    d[n] = ((2h[n - 1] + h[n - 2]) * s[n - 1] - h[n - 1] * s[n - 2]) / (h[n - 1] + h[n - 2])
    if sign(d[n]) != sign(s[n - 1])
        d[n] = 0.0
    elseif sign(s[n - 1]) != sign(s[n - 2]) && abs(d[n]) > abs(3 * s[n - 1])
        d[n] = 3 * s[n - 1]
    end

    # Interior: weighted harmonic-mean of neighbouring secants when they
    # share sign; zero otherwise.
    for i in 2:(n - 1)
        if s[i - 1] * s[i] <= 0
            d[i] = 0.0
        else
            w1 = 2h[i] + h[i - 1]
            w2 = h[i] + 2h[i - 1]
            d[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i])
        end
    end

    # Final monotonicity pass (region-of-monotonicity test): rescale
    # so that (d_i / s_i)^2 + (d_{i+1} / s_i)^2 ≤ 9.
    for i in 1:(n - 1)
        s[i] == 0 && continue
        α = d[i] / s[i]
        β = d[i + 1] / s[i]
        r = α^2 + β^2
        if r > 9
            τ = 3 / sqrt(r)
            d[i] = τ * α * s[i]
            d[i + 1] = τ * β * s[i]
        end
    end

    return d
end

# Evaluate Hermite cubic on [x_i, x_{i+1}] at x.
function _hermite(xi, xi1, yi, yi1, di, di1, x)
    h = xi1 - xi
    t = (x - xi) / h
    h00 = 2t^3 - 3t^2 + 1
    h10 = t^3 - 2t^2 + t
    h01 = -2t^3 + 3t^2
    h11 = t^3 - t^2
    return h00 * yi + h10 * h * di + h01 * yi1 + h11 * h * di1
end

"""
    quantile(d::QuantileDistribution, u)

Return the value of `d` at probability `u ∈ (0,1)`.
"""
function quantile(d::QuantileDistribution, u::Real)
    p = d.probs;
    v = d.vals
    if u <= p[1]
        return quantile(d.left_tail, u)
    elseif u >= p[end]
        return quantile(d.right_tail, u)
    else
        i = searchsortedlast(p, u)
        i == length(p) && return v[end]
        return _hermite(p[i], p[i + 1], v[i], v[i + 1], d.fwd_d[i], d.fwd_d[i + 1], u)
    end
end

"""
    cdf(d::QuantileDistribution, x)

Return P(X ≤ x).
"""
function cdf(d::QuantileDistribution, x::Real)
    p = d.probs;
    v = d.vals
    if x <= v[1]
        return cdf(d.left_tail, x)
    elseif x >= v[end]
        return cdf(d.right_tail, x)
    else
        i = searchsortedlast(v, x)
        i == length(v) && return p[end]
        return _hermite(v[i], v[i + 1], p[i], p[i + 1], d.inv_d[i], d.inv_d[i + 1], x)
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
