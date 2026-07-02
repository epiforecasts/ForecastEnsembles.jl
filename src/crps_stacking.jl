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

with `aᵢᵗ = mean(|Xᵢₖ − yₜ|)`, `Bᵢⱼᵗ = mean(|Xᵢₖ − Xⱼₗ|)` for i ≠ j, and the
within-model term `Bᵢᵢᵗ` averaged over distinct sample pairs k ≠ l only (the
k = l pairs are identically zero and would bias the diagonal downward by
O(1/K) otherwise).

The objective is a concave quadratic in w minimised over the simplex, so the
optimum can sit at a vertex (all weight on one model). A single L-BFGS run
from uniform weights can stall in nearly flat landscapes, so the optimiser is
restarted from each vertex-leaning start as well and the best minimum is kept.

`m.dirichlet_alpha` adds a Dirichlet(α, …, α) log-prior penalty,
`−((α−1)/T) Σᵢ log wᵢ`, matching the scale of the mean per-task CRPS. With
`α = 1` the penalty vanishes exactly; the default 1.001 is a weak
regularisation that keeps the optimum off the simplex boundary (and matches
`lopensemble`'s default).

Per-task weights (recency weighting via `lambda`/`time_col`, or arbitrary
weights via `task_weights`) reweight the task-mean objective; see
[`CRPSStacking`](@ref) for the options. The Dirichlet prior scale uses the
effective sample size of the weights.
"""
function fit(m::CRPSStacking, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :sample || throw(
        ArgumentError(
            "CRPSStacking expects sample forecasts; got $(output_type(training))",
        ),
    )
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations frame must have an :observed column"))

    join_cols = training.task_id_cols
    df = innerjoin(training.data, obs[:, [join_cols..., :observed]]; on = join_cols)
    isempty(df) && throw(ArgumentError("no overlap between training and observations"))

    models = sort(unique(df[!, training.model_id_col]))
    M = length(models)

    # Per task t: A_t (length M) and B_t (M×M), plus the task identifiers
    # (aligned with a_list/b_list) for task weighting.
    a_list = Vector{Float64}[]
    b_list = Matrix{Float64}[]
    task_rows = DataFrame[]
    for tg in DataFrames.groupby(df, join_cols)
        push!(task_rows, DataFrame(tg[1:1, join_cols]))
        y = first(tg.observed)
        A = zeros(M)
        B = zeros(M, M)
        samples_per_model = Vector{Vector{Float64}}(undef, M)
        for (i, mod) in enumerate(models)
            samples_per_model[i] =
                Float64.(tg[tg[!, training.model_id_col] .== mod, :value])
        end
        for i = 1:M
            si = samples_per_model[i]
            isempty(si) && continue
            A[i] = mean(abs.(si .- y))
            for j = i:M
                sj = samples_per_model[j]
                isempty(sj) && continue
                total = sum(abs.(si .- reshape(sj, 1, :)))
                if i == j
                    # Within-model: the k = l pairs are identically zero, so
                    # the unbiased estimator of E|X − X′| averages over the
                    # K(K−1) distinct pairs only.
                    K = length(si)
                    B[i, i] = K > 1 ? total / (K * (K - 1)) : 0.0
                else
                    B[i, j] = total / (length(si) * length(sj))
                    B[j, i] = B[i, j]
                end
            end
        end
        push!(a_list, A)
        push!(b_list, B)
    end

    α = m.dirichlet_alpha
    T = length(a_list)
    tasks = reduce(vcat, task_rows)
    λ = _task_lambda(m, tasks, join_cols)
    sum(λ) > 0 || throw(ArgumentError("task weights sum to zero"))
    λ̃ = λ ./ sum(λ)

    # The objective only depends on the (weighted) task-mean A and B, so
    # collapse them once: loss(w) = w·Ā − ½ w'B̄w − ((α−1)/T_eff) Σ log wᵢ.
    # The prior scales with the effective sample size (Σλ)²/Σλ², so
    # down-weighting history does not quietly strengthen the prior; with
    # equal weights T_eff = T and this reduces to the unweighted case.
    Ā = sum(λ̃[t] .* a_list[t] for t = 1:T)
    B̄ = sum(λ̃[t] .* b_list[t] for t = 1:T)
    T_eff = sum(λ)^2 / sum(abs2, λ)
    prior_scale = (α - 1) / T_eff

    function _logsoftmax(z)
        m_z = maximum(z)
        return z .- (m_z + log(sum(exp.(z .- m_z))))
    end

    function loss(z)
        log_w = _logsoftmax(z)
        w = exp.(log_w)
        return dot(w, Ā) - 0.5 * dot(w, B̄ * w) - prior_scale * sum(log_w)
    end

    # ∂L/∂w, then the softmax Jacobian: ∂L/∂zⱼ = wⱼ(gⱼ − Σₖ wₖ gₖ).
    function grad!(G, z)
        w = exp.(_logsoftmax(z))
        g_w = Ā .- B̄ * w .- prior_scale ./ w
        wg = dot(w, g_w)
        @. G = w * (g_w - wg)
        return G
    end

    # Multi-start: uniform weights plus one vertex-leaning start per model.
    starts = [zeros(M)]
    for i = 1:M
        z = zeros(M)
        z[i] = 4.0
        push!(starts, z)
    end

    best_res = nothing
    for z0 in starts
        res = optimize(loss, grad!, z0, LBFGS())
        if best_res === nothing || Optim.minimum(res) < Optim.minimum(best_res)
            best_res = res
        end
    end
    Optim.converged(best_res) || @warn(
        "CRPSStacking: L-BFGS did not converge " *
        "($(Optim.iterations(best_res)) iterations); weights are the best " *
        "iterate found."
    )

    w_hat = _softmax(Optim.minimizer(best_res))
    crps_hat = Optim.minimum(best_res)

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

# Resolve the per-task weight vector (unnormalised), aligned with the rows
# of `tasks` (one row per training task, in a_list order).
function _task_lambda(m::CRPSStacking, tasks::DataFrame, join_cols)
    T = nrow(tasks)

    if m.task_weights !== nothing
        wdf = m.task_weights
        absent = setdiff(join_cols, propertynames(wdf))
        isempty(absent) ||
            throw(ArgumentError("`task_weights` is missing task-id column(s): $absent"))
        tasks_idx = copy(tasks)
        tasks_idx.__row__ = 1:T
        joined = leftjoin(tasks_idx, wdf[:, [join_cols..., :weight]]; on = join_cols)
        sort!(joined, :__row__)
        any(ismissing, joined.weight) && throw(
            ArgumentError(
                "`task_weights` is missing a weight for at least one training task",
            ),
        )
        return Float64.(joined.weight)
    end

    m.lambda === nothing && return ones(T)

    m.time_col in join_cols || throw(
        ArgumentError(
            "time_col $(m.time_col) is not one of the task-id columns $join_cols",
        ),
    )
    tvals = tasks[!, m.time_col]
    ut = sort(unique(tvals))
    Tt = length(ut)
    per_time = if m.lambda isa Float64
        [m.lambda^(Tt - i) for i = 1:Tt]
    elseif m.lambda === :lopensemble
        [2 - (1 - i / Tt)^2 for i = 1:Tt]
    elseif m.lambda === :equal
        ones(Tt)
    elseif m.lambda isa Vector{Float64}
        length(m.lambda) == Tt || throw(
            ArgumentError(
                "`lambda` has length $(length(m.lambda)) but the training data " *
                "has $Tt unique values of $(m.time_col)",
            ),
        )
        m.lambda
    else # Function of the normalised time rank
        w = [Float64(m.lambda(i / Tt)) for i = 1:Tt]
        all(>=(0), w) ||
            throw(ArgumentError("`lambda` function returned a negative weight"))
        w
    end
    rank = Dict(v => i for (i, v) in enumerate(ut))
    return [per_time[rank[v]] for v in tvals]
end

function _softmax(z::AbstractVector)
    m = maximum(z)
    e = exp.(z .- m)
    return e ./ sum(e)
end
