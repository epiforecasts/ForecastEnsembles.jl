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

@testitem "QuantileDistribution: tied outer quantiles read the step, not 0.5" begin
    # A count-like forecast piling at zero: Q(0.1) = Q(0.25) = 0.
    probs = [0.1, 0.25, 0.5, 0.9]
    vals = [0.0, 0.0, 5.0, 10.0]
    qd = ForecastEnsembles.QuantileDistribution(probs, vals)
    # cdf at the tied boundary is the step probability at that knot, not the
    # median (0.5) of a degenerate spike.
    @test ForecastEnsembles.cdf(qd, 0.0) ≈ 0.1 atol = 1e-8
    # Just above the tie, cdf has jumped toward the next level (the atom mass).
    @test ForecastEnsembles.cdf(qd, 1e-6) > 0.2
    # Still a valid CDF: monotone and in [0, 1].
    cs = [ForecastEnsembles.cdf(qd, x) for x in range(-2.0, 12.0; length = 60)]
    @test issorted(cs)
    @test all(0 .<= cs .<= 1)
end
