using Optim

"""
    FittedCRPSStacking(weights, models)

Output of `fit(::CRPSStacking, …)`. Stores the simplex-constrained ensemble
weights as a `DataFrame` with columns `model_id` and `weight`. Plug into
`combine(ft, fitted)` (sample inputs) — internally a [`LinearPool`](@ref)
with these weights.
"""
struct FittedCRPSStacking <: UnfittedMethod
    weights::DataFrame
    models::Vector{String}
    crps::Float64       # mean training CRPS at the optimum
end

"""
    fit(m::CRPSStacking, training::ForecastTable, observations::AbstractDataFrame) -> FittedCRPSStacking

Estimate ensemble weights by minimising mean CRPS of the weighted mixture
distribution against `:observed` values in `observations`. Expects
`output_type = :sample` for `training`.

The optimisation is performed in softmax space (so weights are unconstrained
during search and projected to the simplex). The per-task CRPS for a mixture
F = Σᵢ wᵢ Fᵢ is computed in closed form from the empirical-sample formulation:

    CRPS_t(w) = Σᵢ wᵢ aᵢᵗ − ½ Σᵢⱼ wᵢ wⱼ Bᵢⱼᵗ

with `aᵢᵗ = mean(|Xᵢₖ − yₜ|)` and `Bᵢⱼᵗ = mean(|Xᵢₖ − Xⱼₗ|)`.

`m.dirichlet_alpha` adds a Dirichlet(α, …, α) log-prior penalty to the loss,
matching `lopensemble`'s MAP estimator. With `α = 1` (uniform prior) the
penalty vanishes; with `α > 1` it pushes the optimum toward the interior of
the simplex.
"""
function fit(m::CRPSStacking, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :sample ||
        throw(ArgumentError("CRPSStacking expects sample forecasts; got $(output_type(training))"))
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations frame must have an :observed column"))

    join_cols = training.task_id_cols
    df = innerjoin(training.data, obs[:, [join_cols..., :observed]]; on = join_cols)
    isempty(df) && throw(ArgumentError("no overlap between training and observations"))

    models = sort(unique(df[!, training.model_id_col]))
    M = length(models)

    # Per task t: A_t (length M) and B_t (M×M).
    a_list = Vector{Float64}[]
    b_list = Matrix{Float64}[]
    for tg in DataFrames.groupby(df, join_cols)
        y = first(tg.observed)
        A = zeros(M)
        B = zeros(M, M)
        samples_per_model = Vector{Vector{Float64}}(undef, M)
        for (i, mod) in enumerate(models)
            samples_per_model[i] = Float64.(tg[tg[!, training.model_id_col] .== mod, :value])
        end
        for i in 1:M
            si = samples_per_model[i]
            isempty(si) && continue
            A[i] = mean(abs.(si .- y))
            for j in i:M
                sj = samples_per_model[j]
                isempty(sj) && continue
                # mean over (i,k) × (j,l) of |X_ik - X_jl|
                # vector-vector outer absolute mean
                bval = mean(abs.(si .- reshape(sj, 1, :)))
                B[i, j] = bval
                B[j, i] = bval
            end
        end
        push!(a_list, A)
        push!(b_list, B)
    end

    α = m.dirichlet_alpha
    T = length(a_list)
    # Per-task CRPS is on the order of |y| (~1 in standard units). The
    # Dirichlet penalty is summed over models without a 1/T factor, so it
    # competes correctly with the *total* training CRPS rather than its
    # mean. We minimise the negative log-posterior up to constants:
    #   loss(z) = mean_t CRPS_t(w) − ((α − 1) / T) · Σᵢ log wᵢ
    function loss(z)
        m_z = maximum(z)
        log_w = z .- (m_z + log(sum(exp.(z .- m_z))))
        w = exp.(log_w)
        s = 0.0
        @inbounds for t in 1:T
            A = a_list[t]; B = b_list[t]
            s += dot(w, A) - 0.5 * dot(w, B * w)
        end
        return s / T - ((α - 1) / T) * sum(log_w)
    end

    z0 = zeros(M)
    res = optimize(loss, z0, LBFGS())
    w_hat = _softmax(Optim.minimizer(res))
    crps_hat = Optim.minimum(res)

    weights_df = DataFrame(model_id = models, weight = w_hat)
    return FittedCRPSStacking(weights_df, String.(models), crps_hat)
end

"""
    combine(ft::ForecastTable, m::FittedCRPSStacking; rng = default_rng()) -> ForecastTable

Apply CRPS-stacked weights to a (sample-typed) forecast table. Equivalent to
`combine(ft, LinearPool(weights = m.weights))`.
"""
function combine(ft::ForecastTable, m::FittedCRPSStacking; rng::AbstractRNG = default_rng())
    return combine(ft, LinearPool(; weights = m.weights); rng = rng)
end

# CRPSStacking is by construction a single per-model weight vector on the
# simplex — directly reusable by any method that accepts `weights`.
weights(m::FittedCRPSStacking) = EnsembleWeights(m.weights)

# ---- helpers ----

function _softmax(z::AbstractVector)
    m = maximum(z)
    e = exp.(z .- m)
    return e ./ sum(e)
end

# `dot` from LinearAlgebra is needed for the CRPS expression; pulled in via
# Optim's transitive dependency.
using LinearAlgebra: dot
