@testitem "QuantileDistribution" begin
    using Random: MersenneTwister
    # Round-trip: build a QD from quantiles of a known Normal, check that
    # `quantile` and `cdf` recover values close to that Normal at interior
    # points.
    using Distributions
    d = Normal(2.0, 1.5)
    probs = collect(0.05:0.05:0.95)
    vals = quantile.(Ref(d), probs)
    qd = ForecastEnsembles.QuantileDistribution(probs, vals)

    # Interior points should match exactly at the knot probabilities.
    for (p, v) in zip(probs, vals)
        @test quantile(qd, p) ≈ v atol = 1e-9
    end
    # CDF round-trip.
    @test ForecastEnsembles.cdf(qd, vals[5]) ≈ probs[5] atol = 1e-9

    # Tail extrapolation should be reasonable for a normal-distributed sample.
    @test quantile(qd, 0.001) < vals[1]
    @test quantile(qd, 0.999) > vals[end]

    # Sampling: empirical mean should be close to true mean.
    rng = MersenneTwister(42)
    s = rand(rng, qd, 20_000)
    @test mean(s) ≈ 2.0 atol = 0.1
    @test std(s) ≈ 1.5 atol = 0.15
end
