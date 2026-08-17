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
    # Just above the tie, cdf has jumped toward the next level. The atom mass on
    # the step from p = 0.1 to p = 0.25 is 0.15, so cdf(0⁺) approaches ~0.25; a
    # loose `> 0.2` bound catches a regression back to the old 0.1 without being
    # brittle to the interpolation detail.
    @test ForecastEnsembles.cdf(qd, 1e-6) > 0.2
    # Still a valid CDF: monotone and in [0, 1].
    cs = [ForecastEnsembles.cdf(qd, x) for x in range(-2.0, 12.0; length = 60)]
    @test issorted(cs)
    @test all(0 .<= cs .<= 1)
end

@testitem "QuantileDistribution: fully-degenerate input is unsupported (pinned)" begin
    # Every quantile value identical: there is no distinct knot to fit a tail from,
    # so both tails fall back to a near-point spike centred on the value. This is
    # documented as unsupported — cdf at the value reads ≈ 0.5 (the spike's median).
    # Pin that behaviour so a future reader does not mistake it for a real tail
    # probability.
    probs = [0.1, 0.25, 0.5, 0.9]
    vals = fill(3.0, 4)
    qd = ForecastEnsembles.QuantileDistribution(probs, vals)
    @test ForecastEnsembles.cdf(qd, 3.0) ≈ 0.5 atol = 1e-8
end
