"""
    ForecastTable

A long-format probabilistic-forecast table aligned with the hubverse
`model_out_tbl` schema.

Required columns
----------------

- the model identifier column (default `:model_id`)
- `:output_type`            — symbol, one of `Ensembles.KNOWN_OUTPUT_TYPES`
- `:output_type_id`         — quantile level, sample index, threshold, …
- `:value`                  — the forecast value
- one or more task-id columns (e.g. `:location`, `:horizon`, `:target_date`)

Construction
------------

```julia
ForecastTable(df; task_id_cols, model_id_col = :model_id)
```

`task_id_cols` may be omitted, in which case it is inferred as every column
that is not one of the required columns.
"""
struct ForecastTable
    data::DataFrame
    task_id_cols::Vector{Symbol}
    model_id_col::Symbol
end

const REQUIRED_NON_TASK_COLS = (:output_type, :output_type_id, :value)

function ForecastTable(
    df::AbstractDataFrame;
    task_id_cols::Union{Nothing,AbstractVector{Symbol}} = nothing,
    model_id_col::Symbol = :model_id,
)
    df = DataFrame(df) # defensive copy / materialise
    _validate_columns!(df, model_id_col)

    inferred_task = setdiff(
        Symbol.(propertynames(df)),
        [model_id_col, REQUIRED_NON_TASK_COLS...],
    )
    chosen_task = task_id_cols === nothing ? inferred_task : Symbol.(task_id_cols)
    isempty(chosen_task) && throw(ArgumentError(
        "ForecastTable needs at least one task-id column; none could be inferred.",
    ))
    for c in chosen_task
        hasproperty(df, c) || throw(ArgumentError("task_id_col $c not present in data"))
    end

    # Normalise output_type to Symbol so dispatch is cheap.
    if !(eltype(df.output_type) <: Symbol)
        df.output_type = Symbol.(df.output_type)
    end
    for ot in unique(df.output_type)
        is_known_output_type(ot) || throw(ArgumentError(
            "unknown output_type :$ot; allowed: $(KNOWN_OUTPUT_TYPES)",
        ))
    end

    return ForecastTable(df, collect(chosen_task), model_id_col)
end

ForecastTable(t; kwargs...) = ForecastTable(DataFrame(t); kwargs...)

function _validate_columns!(df::DataFrame, model_id_col::Symbol)
    required = (model_id_col, REQUIRED_NON_TASK_COLS...)
    missing_cols = [c for c in required if !hasproperty(df, c)]
    isempty(missing_cols) || throw(ArgumentError(
        "ForecastTable is missing required columns: $(missing_cols)",
    ))
end

# ---- accessors ----

DataFrames.DataFrame(ft::ForecastTable) = ft.data
task_id_cols(ft::ForecastTable) = ft.task_id_cols
model_ids(ft::ForecastTable) = unique(ft.data[!, ft.model_id_col])

"""
    output_type(ft::ForecastTable) -> Symbol

The single output_type present in `ft`. Throws if the table mixes output types,
since most ensemble methods are defined on a single type at a time.
"""
function output_type(ft::ForecastTable)
    types = unique(ft.data.output_type)
    length(types) == 1 || throw(ArgumentError(
        "ForecastTable contains mixed output_types $types; split before combining.",
    ))
    return types[1]
end

# ---- Tables.jl interface ----

Tables.istable(::Type{ForecastTable}) = true
Tables.columnaccess(::Type{ForecastTable}) = true
Tables.columns(ft::ForecastTable) = Tables.columns(ft.data)
Tables.schema(ft::ForecastTable) = Tables.schema(ft.data)

# ---- pretty printing ----

function Base.show(io::IO, ::MIME"text/plain", ft::ForecastTable)
    println(io, "ForecastTable(")
    println(io, "  models       = ", model_ids(ft))
    println(io, "  output_type  = ", output_type_summary(ft))
    println(io, "  task_id_cols = ", ft.task_id_cols)
    println(io, "  rows         = ", nrow(ft.data))
    print(io, ")")
end

function output_type_summary(ft::ForecastTable)
    types = unique(ft.data.output_type)
    length(types) == 1 ? string(types[1]) : "mixed: $types"
end
