
"""
    FittedQRA(coefs, intercepts, models, levels, group_cols, per_quantile_weights, enforce_normalisation, has_intercept)

Output of `fit(::QRA, …)`. `coefs` is a `Dict{NamedTuple => Vector{Float64}}`
mapping a `(group..., quantile_level)` key to a vector of model coefficients
in the order `models`. When `per_quantile_weights` is `false`, all keys
sharing the same group share the same coefficients (and have the same key
under a sentinel `:any` quantile_level).

# Fields
- `coefs`: `Dict` mapping each `(group, quantile_level)` key to its vector of
  model coefficients (ordered as `models`).
- `intercepts`: `Dict` mapping each `(group, quantile_level)` key to its
  intercept; empty (all zero) when the fit has no intercept.
- `models`: component model ids, giving the order of the coefficient vectors.
- `levels`: the quantile levels the fit was trained on.
- `group_cols`: the grouping task columns used when fitting; may be empty.
- `per_quantile_weights`: `Bool`, whether coefficients vary across quantile
  levels.
- `enforce_normalisation`: `Bool`, whether the simplex constraint
  (non-negative coefficients summing to one) was imposed.
- `has_intercept`: `Bool`, whether an intercept was estimated.
"""
struct FittedQRA <: UnfittedMethod
    coefs::Dict{Tuple, Vector{Float64}}
    intercepts::Dict{Tuple, Float64}
    models::Vector{String}
    levels::Vector{Float64}
    group_cols::Vector{Symbol}
    per_quantile_weights::Bool
    enforce_normalisation::Bool
    has_intercept::Bool
end

"""
    fit(m::QRA, training::ForecastTable, observations::AbstractDataFrame) -> FittedQRA

Estimate quantile regression weights from a training set of forecasts paired
with observed values.

`observations` must contain the task-id columns of `training` plus a column
named `:observed` with the realised value.

Mirrors the core of `qrensemble::qra`: per group (and per quantile level if
`per_quantile_weights = true`), solve a quantile regression LP with the
component forecasts as predictors. `enforce_normalisation = true` constrains
the coefficients to lie on the simplex (non-negative, sum to one); `noncross`
adds across-quantile monotonicity constraints when `per_quantile_weights = true`.

# Arguments
- `m`: a `QRA` configuration specifying grouping, per-quantile weighting,
  intercept, normalisation and non-crossing options.
- `training`: a `ForecastTable` of quantile forecasts to fit weights from.
- `observations`: a `DataFrame` containing the task-id columns of `training`
  plus an `:observed` column with the realised value.

# Examples
```@example
using ForecastEnsembles, DataFrames, Random, Distributions
rng = MersenneTwister(1)
n = 60; levels = [0.1, 0.5, 0.9]
y = randn(rng, n)
rows = DataFrame[]
for (mid, pred) in (("m1", y .+ 0.3 .* randn(rng, n)), ("m2", y .+ randn(rng, n)))
    for τ in levels
        push!(rows, DataFrame(model_id = mid, output_type = "quantile",
            output_type_id = τ, t = 1:n, value = pred .+ quantile(Normal(), τ)))
    end
end
train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
obs = DataFrame(t = 1:n, observed = y)
fit(QRA(; enforce_normalisation = true, intercept = false), train, obs)
```
"""
function fit(m::QRA, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :quantile ||
        throw(ArgumentError("QRA expects quantile forecasts; got $(output_type(training))"))

    join_cols = training.task_id_cols
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations frame must have an :observed column"))

    df = innerjoin(training.data, obs[:, [join_cols..., :observed]]; on = join_cols)
    isempty(df) &&
        throw(ArgumentError("no overlap between training and observations on task_id_cols"))

    models = sort(unique(df[!, training.model_id_col]))
    levels = sort(unique(Float64.(df.output_type_id)))

    group_cols = m.group
    coefs = Dict{Tuple, Vector{Float64}}()
    intercepts = Dict{Tuple, Float64}()

    for group_df in DataFrames.groupby(df, isempty(group_cols) ? Symbol[] : group_cols)
        gkey = isempty(group_cols) ? () :
               NamedTuple(c => group_df[1, c] for c in group_cols)

        if m.per_quantile_weights
            if m.noncross
                # Joint LP across τ with cross-τ monotonicity constraints.
                βs,
                β0s = _per_quantile_regression_noncross(
                    group_df,
                    models,
                    levels,
                    training.model_id_col;
                    intercept = m.intercept,
                    simplex = m.enforce_normalisation
                )
                for (k, τ) in enumerate(levels)
                    coefs[(gkey, τ)] = βs[k]
                    intercepts[(gkey, τ)] = β0s[k]
                end
            else
                for τ in levels
                    gτ = group_df[group_df.output_type_id .== τ, :]
                    X, y = _design_matrix(gτ, models, training.model_id_col)
                    β,
                    β0 = _quantile_regression(
                        X,
                        y,
                        τ;
                        intercept = m.intercept,
                        simplex = m.enforce_normalisation
                    )
                    coefs[(gkey, τ)] = β
                    intercepts[(gkey, τ)] = β0
                end
            end
        else
            # Joint fit: stack rows over τ levels with τ-tilted loss.
            β,
            β0 = _joint_quantile_regression(
                group_df,
                models,
                levels,
                training.model_id_col;
                intercept = m.intercept,
                simplex = m.enforce_normalisation
            )
            for τ in levels
                coefs[(gkey, τ)] = β
                intercepts[(gkey, τ)] = β0
            end
        end
    end

    return FittedQRA(
        coefs,
        intercepts,
        String.(models),
        levels,
        group_cols,
        m.per_quantile_weights,
        m.enforce_normalisation,
        m.intercept
    )
end

"""
    combine(ft::ForecastTable, m::FittedQRA) -> ForecastTable

Apply fitted QRA weights to a new set of forecasts. The output has
`output_type = :quantile` and one row per (task, quantile_level).
"""
function combine(ft::ForecastTable, m::FittedQRA; rng::AbstractRNG = default_rng())
    output_type(ft) === :quantile ||
        throw(ArgumentError("FittedQRA expects quantile forecasts; got $(output_type(ft))"))

    df = ft.data
    out_groups = DataFrame[]
    other_cols = vcat(m.group_cols, ft.task_id_cols)
    other_cols = unique(other_cols)

    for tg in DataFrames.groupby(df, other_cols)
        gkey = isempty(m.group_cols) ? () : NamedTuple(c => tg[1, c] for c in m.group_cols)
        # Readable label for error messages: `()` is opaque for an ungrouped fit.
        group_label = isempty(m.group_cols) ? "the ungrouped fit" : "group $gkey"
        levels_present = sort(unique(Float64.(tg.output_type_id)))

        unseen = setdiff(levels_present, m.levels)
        isempty(unseen) || throw(
            ArgumentError(
            "FittedQRA was not trained on quantile levels $unseen " *
            "(trained levels: $(m.levels)).",
        ),
        )
        haskey(m.coefs, (gkey, first(levels_present))) || throw(
            ArgumentError(
            "FittedQRA was not trained on group $gkey " *
            "(group columns: $(m.group_cols)).",
        ),
        )

        # A model entirely absent from the input; the per-τ loop below catches a
        # model that is present but missing at a specific level.
        models_present = unique(tg[!, ft.model_id_col])
        missing_models = setdiff(m.models, models_present)
        isempty(missing_models) || throw(
            ArgumentError(
            "FittedQRA models $(join(missing_models, ", ")) are absent from the input table.",
        ),
        )

        values = Float64[]
        for τ in levels_present
            sub = tg[tg.output_type_id .== τ, :]
            x = Float64[]
            for mod in m.models
                vals = sub[sub[!, ft.model_id_col] .== mod, :value]
                isempty(vals) && throw(ArgumentError(
                    "FittedQRA cannot combine: model $mod is missing at " *
                    "output_type_id $τ for $group_label. QRA needs every model at " *
                    "every quantile level; supply a complete input or drop the model.",
                ))
                # A model must appear exactly once per (τ, group); flag duplicate
                # rows rather than silently picking one.
                length(vals) == 1 || throw(ArgumentError(
                    "FittedQRA cannot combine: model $mod appears $(length(vals)) " *
                    "times at output_type_id $τ for $group_label; expected exactly " *
                    "one row per model per level."))
                push!(x, first(vals))
            end
            β = m.coefs[(gkey, τ)]
            β0 = m.intercepts[(gkey, τ)]
            push!(values, β0 + sum(β .* x))
        end
        # Non-crossing is only guaranteed at training points (and only when
        # fitted with noncross = true). At new points, per-quantile fits
        # without the simplex constraint can produce crossing quantiles,
        # which hub validators typically reject.
        issorted(values) ||
            @warn("FittedQRA produced crossing quantiles for at least one task; " *
                  "consider per_quantile_weights = true with noncross = true, or " *
                  "post-process (e.g. sort) before submission.",
                maxlog = 1,)
        out = DataFrame(tg[1:1, ft.task_id_cols])
        out = repeat(out, length(levels_present))
        out.output_type = fill(:quantile, length(levels_present))
        out.output_type_id = levels_present
        out.value = values
        out[!, ft.model_id_col] .= "qra"
        push!(out_groups, out)
    end
    res = reduce(vcat, out_groups)
    select!(res, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(
        res;
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

"""
    weights(m::FittedQRA) -> Union{DataFrame, Nothing}

Return the fitted weights as a `DataFrame`, or `nothing` if the fit shape
isn't a clean weight vector. Two shapes are produced:

- *Per-model* (cols `:model_id, :weight`) — when the fit is joint
  (`per_quantile_weights = false`), simplex (`enforce_normalisation = true`),
  no intercept. The same weights apply at every quantile level.
- *Per-quantile* (cols `:model_id, :output_type_id, :weight`) — when the
  fit is per-τ (`per_quantile_weights = true`), simplex, no intercept.
  Weights vary across quantile levels.

Returns `nothing` when the fit has an intercept, isn't simplex-constrained,
or has more than one task group (different groups give different fits and
the user has to disambiguate).
"""
function weights(m::FittedQRA)
    m.enforce_normalisation || return nothing
    m.has_intercept && return nothing

    # If the fit was grouped, there is more than one (gkey, τ) → β. Refuse.
    isempty(m.group_cols) || _has_single_group(m) || return nothing

    if m.per_quantile_weights
        # One β per τ, all under the same gkey. Long-format frame.
        out = DataFrame(model_id = String[], output_type_id = Float64[], weight = Float64[])
        for τ in m.levels
            β = _coefs_for(m, τ)
            β === nothing && return nothing
            for (i, mod) in enumerate(m.models)
                push!(out, (mod, τ, β[i]))
            end
        end
        return EnsembleWeights(out)
    else
        # Joint fit: every τ shares the same β.
        βs = unique(values(m.coefs))
        length(βs) == 1 || return nothing
        β = first(βs)
        return EnsembleWeights(DataFrame(model_id = m.models, weight = β))
    end
end

# Report which of the three disqualifying cases this fit is in, since the caller
# cannot tell them apart from a bare `nothing`.
function _no_weights_reason(m::FittedQRA)
    m.enforce_normalisation ||
        return "this QRA fit is unconstrained, so its coefficients are a " *
               "regression, not weights summing to one; refit with " *
               "`enforce_normalisation = true` for a weight vector"
    m.has_intercept &&
        return "this QRA fit has an intercept, so the coefficients alone do " *
               "not define the combination; refit without one for a weight " *
               "vector"
    return "this QRA fit is grouped and the groups have different " *
           "coefficients, so there is no single weight vector; fit one group " *
           "at a time to get weights"
end

# m.coefs keys are (gkey, τ) where gkey is `()` when no group_cols were set.
function _coefs_for(m::FittedQRA, τ)
    for (k, β) in m.coefs
        k[2] == τ && return β
    end
    return nothing
end

# True when m.coefs has only one distinct gkey across its keys.
function _has_single_group(m::FittedQRA)
    gkeys = unique(first(k) for k in keys(m.coefs))
    return length(gkeys) == 1
end

# ---------- LP helpers ------------------------------------------------------

# HiGHS returns NaN coefficients without complaint when the LP is infeasible
# or hits a limit; surface that as an error instead.
function _check_lp_solution(model, context::String)
    st = termination_status(model)
    st == MOI.OPTIMAL || error(
        "QRA $context LP did not solve to optimality (status: $st). " *
        "Check the training data for degeneracy (e.g. collinear forecasts " *
        "with enforce_normalisation = true).",
    )
end

# Returns (X, y) where X is n×M with one column per model in `models`
# (in the given order) and rows aligned with `df` by row order. Assumes `df`
# is the long-format slice for one (group, τ) and contains the :value and
# :observed columns.
function _design_matrix(
        df::AbstractDataFrame,
        models::Vector{<:AbstractString},
        model_id_col::Symbol
)
    # Pivot so each (task) has all M model values on a single row.
    other_cols = setdiff(propertynames(df), [
        model_id_col, :output_type, :output_type_id, :value])
    wide = unstack(df, other_cols, model_id_col, :value)
    # A model with no forecasts in this slice has no column after unstack; add it
    # as all-missing so the complete-case filter drops those rows rather than
    # treating the absent model as zero.
    for mod in models
        hasproperty(wide, Symbol(mod)) || (wide[!, Symbol(mod)] = fill(missing, nrow(wide)))
    end
    cols = Symbol.(models)
    # Complete-case: keep only tasks where every model submitted a value; partial
    # submissions leave `missing` cells that break the numeric conversion. Callers
    # wanting imputation must do it first.
    keep = DataFrames.completecases(wide[:, cols])
    any(keep) || throw(ArgumentError(
        "QRA has no training tasks where all of $(join(models, ", ")) submitted a " *
        "forecast; supply complete cases or drop the incomplete models."))
    wide = wide[keep, :]
    # Columns carry a `Union{Missing, Float64}` eltype, so fill the matrix column
    # by column; a single `Matrix{Float64}(…)` cannot narrow the union.
    X = Matrix{Float64}(undef, nrow(wide), length(cols))
    for (j, c) in enumerate(cols)
        X[:, j] = Float64.(wide[!, c])
    end
    y = Float64.(wide.observed)
    return X, y
end

# Single-τ quantile regression LP.
function _quantile_regression(
        X::AbstractMatrix,
        y::AbstractVector,
        τ::Real;
        intercept::Bool = true,
        simplex::Bool = false
)
    n, M = size(X)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, β[1:M])
    @variable(model, β0)
    if simplex
        @constraint(model, [j = 1:M], β[j] >= 0)
        @constraint(model, sum(β) == 1)
    end
    if !intercept
        @constraint(model, β0 == 0)
    end
    @variable(model, u[1:n] >= 0)
    @variable(model, v[1:n] >= 0)
    @constraint(model,
        [i = 1:n],
        y[i] - β0 - sum(X[i, j] * β[j] for j in 1:M) == u[i] - v[i])
    @objective(model, Min, sum(τ * u[i] + (1 - τ) * v[i] for i in 1:n))
    optimize!(model)
    _check_lp_solution(model, "single-τ")
    return value.(β), value(β0)
end

# Per-τ quantile regression with non-crossing constraints: separate
# coefficients (β_τ, β0_τ) per τ, but for every training point the predicted
# quantiles must be non-decreasing in τ. Returns (Vector{Vector{Float64}},
# Vector{Float64}) keyed in the order of `levels`.
function _per_quantile_regression_noncross(
        group_df::AbstractDataFrame,
        models::Vector{<:AbstractString},
        levels::Vector{Float64},
        model_id_col::Symbol;
        intercept::Bool = true,
        simplex::Bool = false
)
    # Build aligned design matrices per τ. Assumes all τ share the same set
    # of training points (typical when forecasts are produced jointly).
    Xs = Matrix{Float64}[]
    ys = Vector{Float64}[]
    for τ in levels
        sub = group_df[group_df.output_type_id .== τ, :]
        X, y = _design_matrix(sub, models, model_id_col)
        push!(Xs, X)
        push!(ys, y)
    end
    M = size(first(Xs), 2)
    K = length(levels)
    n = size(first(Xs), 1)
    all(size(X, 1) == n for X in Xs) ||
        throw(ArgumentError(
            "noncross requires the same training tasks at every quantile level; " *
            "got mismatched sizes $(map(X -> size(X, 1), Xs)). This can happen with " *
            "partial submissions where a model skips some tasks at some levels — " *
            "ensure every model submits at every level, or drop the incomplete models."))

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, β[1:M, 1:K])
    @variable(model, β0[1:K])
    if simplex
        @constraint(model, [j = 1:M, k = 1:K], β[j, k] >= 0)
        @constraint(model, [k = 1:K], sum(β[:, k]) == 1)
    end
    if !intercept
        @constraint(model, [k = 1:K], β0[k] == 0)
    end

    obj = AffExpr(0.0)
    for (k, τ) in enumerate(levels)
        u = @variable(model, [1:n], lower_bound = 0.0)
        v = @variable(model, [1:n], lower_bound = 0.0)
        for i in 1:n
            @constraint(model,
                ys[k][i] - β0[k] - sum(Xs[k][i, j] * β[j, k] for j in 1:M) == u[i] - v[i])
            add_to_expression!(obj, τ, u[i])
            add_to_expression!(obj, 1 - τ, v[i])
        end
    end

    # Non-crossing: predicted quantile at τ_k ≤ predicted at τ_{k+1} for
    # every training point.
    for k in 1:(K - 1)
        for i in 1:n
            @constraint(model,
                β0[k] + sum(Xs[k][i, j] * β[j, k] for j in 1:M) <=
                β0[k + 1] + sum(Xs[k + 1][i, j] * β[j, k + 1] for j in 1:M))
        end
    end

    @objective(model, Min, obj)
    optimize!(model)
    _check_lp_solution(model, "noncross")

    βs = [collect(value.(β[:, k])) for k in 1:K]
    β0s = collect(value.(β0))
    return βs, β0s
end

# Joint quantile regression: stack rows for all τ in levels, with τ-tilted
# losses but a single shared coefficient vector. Component forecasts on the
# rhs are the τ-specific quantile values. Mirrors qrensemble's
# `per_quantile_weights = FALSE` mode.
function _joint_quantile_regression(
        group_df::AbstractDataFrame,
        models::Vector{<:AbstractString},
        levels::Vector{Float64},
        model_id_col::Symbol;
        intercept::Bool = true,
        simplex::Bool = false
)
    Xs = Matrix{Float64}[]
    ys = Vector{Float64}[]
    τs = Float64[]
    for τ in levels
        sub = group_df[group_df.output_type_id .== τ, :]
        X, y = _design_matrix(sub, models, model_id_col)
        push!(Xs, X)
        push!(ys, y)
        push!(τs, τ)
    end
    M = size(first(Xs), 2)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, β[1:M])
    @variable(model, β0)
    if simplex
        @constraint(model, [j = 1:M], β[j] >= 0)
        @constraint(model, sum(β) == 1)
    end
    if !intercept
        @constraint(model, β0 == 0)
    end

    obj = AffExpr(0.0)
    for (k, τ) in enumerate(τs)
        n = size(Xs[k], 1)
        u = @variable(model, [1:n], lower_bound = 0.0)
        v = @variable(model, [1:n], lower_bound = 0.0)
        for i in 1:n
            @constraint(model,
                ys[k][i] - β0 - sum(Xs[k][i, j] * β[j] for j in 1:M) == u[i] - v[i])
            add_to_expression!(obj, τ, u[i])
            add_to_expression!(obj, 1 - τ, v[i])
        end
    end
    @objective(model, Min, obj)
    optimize!(model)
    _check_lp_solution(model, "joint")
    return value.(β), value(β0)
end
