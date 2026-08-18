# Partial pooling / hierarchical stacking. Instead of one global weight vector
# (complete pooling) or an independent weight vector per stratum (no pooling),
# learn per-stratum weights that shrink toward a shared global vector. In softmax
# logit space each stratum `s` has logits `zₛ`, a global `z₀` anchors them, and a
# penalty `lambda · Σₛ‖zₛ − z₀‖²` controls the shrinkage: `lambda → 0` gives
# independent per-stratum stacking, `lambda → ∞` collapses every stratum onto the
# global weights. The score is any caller-supplied `score(samples, y; w)` — the
# `fit` references no scoring library, so this is plain MIT-core Julia. This file
# holds the type, the stratum-aware `combine` (each stratum gets its own weights,
# unseen strata fall back to the global vector), and the fit. ScoringRules.jl is
# the natural companion for the score.

"""
    PartialPooling(score; strata, lambda = 1.0, dirichlet_alpha = 1.0)

Hierarchical (partially pooled) stacking: learn a weight vector per stratum that
shrinks toward a shared global vector, so a data-sparse stratum borrows strength
from the rest.

`fit(PartialPooling(score; strata), training, observations)` jointly optimises,
in softmax space, one logit vector per distinct combination of the `strata`
columns plus a global logit vector, minimising the mean `score` of each
stratum's linearly-pooled forecast plus a shrinkage penalty pulling every
stratum toward the global vector. The [`FittedPartialPooling`](@ref) result
plugs into [`combine`](@ref), which applies each stratum's own weights (an unseen
stratum falls back to the global vector).

`score` is any callable `score(samples, y; w)`;
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) is the natural
companion, not a dependency of this package. Generalises [`Stacking`](@ref): a
single stratum, or `lambda → ∞`, recovers global stacking.

# Fields

- `score`: the scoring-rule function to minimise (negatively oriented).
- `strata`: task-id columns whose value combinations define the strata (e.g.
  `[:location]`, `[:location, :age_group]`).
- `lambda`: shrinkage strength toward the global vector. `0` fits each stratum
  independently; large values pool them toward one shared vector. Must be `≥ 0`.
  The objective weights `lambda` as a per-stratum-averaged penalty against a
  per-task-averaged data loss, so it is the trade-off between a stratum's typical
  squared logit distance from the global vector and the typical per-task score —
  a scale that does not drift with the number of strata. Tune it on held-out
  score (e.g. via [`backtest`](@ref)); there is no universally correct value.
- `dirichlet_alpha`: strength of a symmetric-Dirichlet prior on each stratum's
  weights, pulling them toward the simplex centre; `1.0` is the flat,
  uninformative prior.
"""
struct PartialPooling{F} <: TrainedMethod
    score::F
    strata::Vector{Symbol}
    lambda::Float64
    dirichlet_alpha::Float64
end

function PartialPooling(
        score;
        strata,
        lambda::Real = 1.0,
        dirichlet_alpha::Real = 1.0
)
    lambda >= 0 || throw(ArgumentError("lambda (shrinkage) must be >= 0 (got $lambda)"))
    dirichlet_alpha >= 1 ||
        throw(ArgumentError("dirichlet_alpha must be >= 1 (1 is the flat, " *
                            "uninformative prior; larger values pull the weights " *
                            "toward the simplex centre)"))
    st = Symbol.(collect(strata))
    isempty(st) && throw(ArgumentError("strata must name at least one task-id column"))
    return PartialPooling{typeof(score)}(score, st, Float64(lambda), Float64(dirichlet_alpha))
end

"""
    FittedPartialPooling(weights, global_weights, strata, models, score_value)

Output of `fit(::PartialPooling, …)`. Stores the per-stratum `weights` (a
`DataFrame` with the `strata` columns plus `model_id` and `weight`), the pooled
`global_weights` (`model_id`, `weight`) used for strata not seen in training, the
`strata` columns, the component `models`, and the mean `score_value` at the
optimum. Plug into `combine(ft, fitted)` — it applies each stratum's weights,
falling back to `global_weights` for an unseen stratum. `weights(fitted)` returns
the pooled global vector as an [`EnsembleWeights`](@ref).
"""
struct FittedPartialPooling <: UnfittedMethod
    weights::DataFrame
    global_weights::DataFrame
    strata::Vector{Symbol}
    models::Vector{String}
    score_value::Float64
end

function combine(ft::ForecastTable, m::FittedPartialPooling; rng::AbstractRNG = default_rng())
    absent = setdiff(m.strata, ft.task_id_cols)
    isempty(absent) || throw(ArgumentError(
        "combine target is missing strata column(s) $absent; its task-id columns " *
        "are $(ft.task_id_cols)"))

    outs = DataFrame[]
    for g in DataFrames.groupby(ft.data, m.strata)
        key = g[1, m.strata]
        wdf = _stratum_weights(m, key)
        sub = ForecastTable(
            DataFrame(g);
            task_id_cols = ft.task_id_cols,
            model_id_col = ft.model_id_col
        )
        push!(outs, combine(sub, LinearPool(; weights = wdf); rng = rng).data)
    end
    return ForecastTable(
        reduce(vcat, outs);
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

# The weights for one stratum: the matching rows of the per-stratum table, or the
# pooled global vector when the stratum was not seen in training.
function _stratum_weights(m::FittedPartialPooling, key)
    w = m.weights
    mask = trues(nrow(w))
    for s in m.strata
        mask .&= (w[!, s] .== key[s])
    end
    sub = w[mask, [:model_id, :weight]]
    return nrow(sub) == 0 ? m.global_weights : sub
end

# The pooled global vector is the natural single-vector summary.
weights(m::FittedPartialPooling) = EnsembleWeights(m.global_weights)

# Per-stratum logits `zₛ` plus a global `z₀`, optimised jointly: the mean
# weighted-sample score of each stratum's mixture, plus `lambda · Σₛ‖zₛ − z₀‖²`
# shrinking strata toward the global vector. `lambda → 0` fits strata
# independently; `lambda → ∞` pools them onto one vector (the optimum sets
# `z₀ = meanₛ zₛ`). The score is the user's callable, so this stays in the core.
function fit(m::PartialPooling, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :sample || throw(ArgumentError(
        "PartialPooling supports :sample forecasts (weighted-sample scores)."))
    absent = setdiff(m.strata, task_id_cols(training))
    isempty(absent) || throw(ArgumentError(
        "strata $absent must be among the task-id columns $(task_id_cols(training))"))

    tcols = task_id_cols(training)
    mid = training.model_id_col
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    d = innerjoin(training.data, obs[:, [tcols..., :observed]]; on = tcols)
    isempty(d) && throw(ArgumentError("no overlap between forecasts and observations"))
    models = sort(unique(d[!, mid]))
    M = length(models)
    M >= 2 || throw(ArgumentError("need at least two models to stack (got $M)"))
    idx = Dict(mm => i for (i, mm) in enumerate(models))

    # Distinct strata (as ordered tuples of the strata-column values).
    skeys = [Tuple(r[s] for s in m.strata) for r in eachrow(unique(d[:, m.strata]))]
    S = length(skeys)
    sidx = Dict(k => i for (i, k) in enumerate(skeys))

    # Per task: pooled samples, each sample's model index, per-model counts, y,
    # and the stratum index (a task lies wholly in one stratum: strata ⊆ tcols).
    task_data = map(collect(DataFrames.groupby(d, tcols))) do g
        midx = [idx[mm] for mm in g[!, mid]]
        counts = [count(==(i), midx) for i in 1:M]
        s = sidx[Tuple(g[1, col] for col in m.strata)]
        (samples = Float64.(g.value), midx = midx, counts = counts,
            y = Float64(first(g.observed)), s = s)
    end

    score = m.score
    α = m.dirichlet_alpha
    λ = m.lambda
    ntask = length(task_data)

    # z packs the global logits (first M) then S per-stratum blocks of M.
    gview(z) = @view z[1:M]
    sview(z, s) = @view z[(M * s + 1):(M * (s + 1))]

    function loss(z)
        z0 = gview(z)
        data = zero(eltype(z))
        for td in task_data
            w = _softmax(sview(z, td.s))
            sw = [w[i] / td.counts[i] for i in td.midx]
            data += score(td.samples, td.y; w = sw)
        end
        data /= ntask
        shrink = zero(eltype(z))
        penalty = zero(eltype(z))
        for s in 1:S
            zs = sview(z, s)
            shrink += sum(abs2, zs .- z0)
            if α > 1
                penalty -= (α - 1) * sum(log, _softmax(zs))
            end
        end
        shrink *= λ / S
        penalty /= (S * ntask)
        return data + shrink + penalty
    end

    res = optimize(loss, zeros(M * (S + 1)), LBFGS())
    z_hat = Optim.minimizer(res)

    # Per-stratum weights table (strata columns + model_id + weight).
    on_rows = DataFrame[]
    for (k, s) in sidx
        w = _softmax(sview(z_hat, s))
        df = DataFrame(model_id = String.(models), weight = w)
        for (j, col) in enumerate(m.strata)
            df[!, col] .= k[j]
        end
        push!(on_rows, df)
    end
    weights_df = reduce(vcat, on_rows)
    global_df = DataFrame(model_id = String.(models), weight = _softmax(gview(z_hat)))
    return FittedPartialPooling(weights_df, global_df, copy(m.strata),
        String.(models), Optim.minimum(res))
end
