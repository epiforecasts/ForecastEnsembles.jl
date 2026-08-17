"""
    EnsembleWeights(data; shape = :auto)

A typed container for per-model or per-quantile ensemble weights. The
underlying `data::DataFrame` always has columns `:model_id` and `:weight`,
and optionally `:output_type_id`. Two shapes are supported:

- `:per_model` — columns `:model_id, :weight`. A single weight per model
  applied at every quantile/sample/etc.
- `:per_quantile` — columns `:model_id, :output_type_id, :weight`. Weights
  vary across quantile levels.

Construct from a DataFrame (shape inferred from columns by default) or by
calling `weights(m)` on a fitted method. `MixtureEnsemble` and
`QuantileEnsemble` accept any of: a raw DataFrame, an `EnsembleWeights`,
or any fitted method for which `weights(m)` returns one. The conversion
happens at construction
of the method, not at `combine` time, so invalid inputs fail fast.
"""
struct EnsembleWeights
    data::DataFrame
    shape::Symbol   # :per_model or :per_quantile
end

function EnsembleWeights(df; shape::Symbol = :auto)
    df = DataFrame(df)
    cols = propertynames(df)
    has_model_id = :model_id in cols
    has_weight = :weight in cols
    has_oti = :output_type_id in cols
    (has_model_id && has_weight) || throw(
        ArgumentError(
        "EnsembleWeights requires :model_id and :weight columns " *
        "(got $(collect(cols))).",
    ),
    )

    all(w -> !ismissing(w) && isfinite(w) && w >= 0, df.weight) || throw(
        ArgumentError(
        "EnsembleWeights requires finite, non-negative, non-missing weights; " *
        "mixing operations are undefined for negative or non-finite weights.",
    ),
    )

    inferred = has_oti ? :per_quantile : :per_model
    if shape === :auto
        shape = inferred
    elseif shape == :per_quantile && !has_oti
        throw(ArgumentError("shape = :per_quantile requires an :output_type_id column"))
    elseif shape == :per_model && has_oti
        throw(
            ArgumentError(
            "shape = :per_model is incompatible with an :output_type_id column",
        ),
        )
    elseif shape ∉ (:per_model, :per_quantile)
        throw(ArgumentError("shape must be :per_model, :per_quantile, or :auto"))
    end

    return EnsembleWeights(df, shape)
end

# Constructor passthrough: idempotent on EnsembleWeights values.
EnsembleWeights(w::EnsembleWeights) = w

is_per_quantile(w::EnsembleWeights) = w.shape === :per_quantile

DataFrames.DataFrame(w::EnsembleWeights) = w.data

function Base.show(io::IO, ::MIME"text/plain", w::EnsembleWeights)
    println(io, "EnsembleWeights($(w.shape), $(nrow(w.data)) rows)")
    show(io, MIME("text/plain"), w.data)
end
