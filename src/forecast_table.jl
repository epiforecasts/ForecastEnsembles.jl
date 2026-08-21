"""
    ForecastTable

A long-format probabilistic-forecast table aligned with the hubverse
`model_out_tbl` schema.

Required columns
----------------

- the model identifier column (default `:model_id`)
- `:output_type`            — symbol, one of `ForecastEnsembles.KNOWN_OUTPUT_TYPES`
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

Getting the data out
--------------------

`DataFrame(ft)` returns a copy, safe to mutate, as does
`DataFrame(ft; copycols = true)`. Two routes share the backing store
instead, as the Tables.jl zero-copy contract expects: `Tables.columns(ft)`
and `DataFrame(ft; copycols = false)` both hand back columns aliasing
`ft.data`. Mutating those in place corrupts `ft` and bypasses the
constructor's validation, so pass `copycols = false` only when you know the
result is read-only.

Fields
------

- `data`: the underlying long-format `DataFrame` holding the forecasts.
- `task_id_cols`: the task-id columns identifying a forecast target.
- `model_id_col`: the column naming the model that produced each forecast.
"""
struct ForecastTable
    data::DataFrame
    task_id_cols::Vector{Symbol}
    model_id_col::Symbol
end

const REQUIRED_NON_TASK_COLS = (:output_type, :output_type_id, :value)

function ForecastTable(
        df::AbstractDataFrame;
        task_id_cols::Union{Nothing, AbstractVector{Symbol}} = nothing,
        model_id_col::Symbol = :model_id
)
    df = DataFrame(df) # defensive copy / materialise
    _validate_columns!(df, model_id_col)
    nrow(df) > 0 || throw(ArgumentError("ForecastTable must contain at least one row"))
    if any(v -> ismissing(v) || (v isa AbstractFloat && isnan(v)), df.value)::Bool
        throw(
            ArgumentError(
            "ForecastTable :value column contains missing or NaN entries; " *
            "filter or impute before constructing.",
        ),
        )
    end

    inferred_task = setdiff(Symbol.(propertynames(df)), [
        model_id_col, REQUIRED_NON_TASK_COLS...])
    chosen_task = task_id_cols === nothing ? inferred_task : Symbol.(task_id_cols)
    isempty(chosen_task) && throw(
        ArgumentError(
        "ForecastTable needs at least one task-id column; none could be inferred.",
    ),
    )
    for c in chosen_task
        hasproperty(df, c) || throw(ArgumentError("task_id_col $c not present in data"))
    end

    # Normalise output_type to Symbol so dispatch is cheap.
    if !(eltype(df.output_type) <: Symbol)
        df.output_type = Symbol.(df.output_type)
    end
    for ot in unique(df.output_type)
        is_known_output_type(ot) ||
            throw(ArgumentError("unknown output_type :$ot; allowed: $(KNOWN_OUTPUT_TYPES)"))
    end

    return ForecastTable(df, collect(chosen_task), model_id_col)
end

ForecastTable(t; kwargs...) = ForecastTable(DataFrame(t); kwargs...)

function _validate_columns!(df::DataFrame, model_id_col::Symbol)
    required = (model_id_col, REQUIRED_NON_TASK_COLS...)
    missing_cols = [c for c in required if !hasproperty(df, c)]
    isempty(missing_cols) ||
        throw(ArgumentError("ForecastTable is missing required columns: $(missing_cols)"))
end

# ---- accessors ----

# Defensive copy: callers must not be able to mutate the table's backing store
# (which would bypass the constructor's validation) through the accessor.
# `copy` copies the column vectors, so both column replacement and in-place
# element writes on the returned frame leave `ft.data` untouched.
#
# `copycols` is declared here rather than left to the generic Tables.jl
# constructor, which a keyword call would otherwise fall through to: without it
# `DataFrame(ft; copycols = false)` aliased `ft.data` by accident of dispatch,
# giving no hint that the result shares the store.
function DataFrames.DataFrame(ft::ForecastTable; copycols::Bool = true)
    return copycols ? copy(ft.data) : DataFrames.DataFrame(ft.data; copycols = false)
end

"""
    task_id_cols(ft::ForecastTable) -> Vector{Symbol}

The task-id columns of `ft`, i.e. the columns that together identify a forecast
target (e.g. `:location`, `:horizon`, `:target_date`).

# Arguments

- `ft`: a [`ForecastTable`](@ref).

# Example

```@example
using ForecastEnsembles, DataFrames
df = DataFrame(
    location = "A", horizon = 1,
    model_id = repeat(["m1", "m2", "m3"], inner = 2),
    output_type = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5]
)
ft = ForecastTable(df; task_id_cols = [:location, :horizon])
task_id_cols(ft)
```
"""
task_id_cols(ft::ForecastTable) = ft.task_id_cols

"""
    model_ids(ft::ForecastTable) -> Vector

The distinct model identifiers present in `ft`, taken from its model-id column.

# Arguments

- `ft`: a [`ForecastTable`](@ref).

# Example

```@example
using ForecastEnsembles, DataFrames
df = DataFrame(
    location = "A", horizon = 1,
    model_id = repeat(["m1", "m2", "m3"], inner = 2),
    output_type = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5]
)
ft = ForecastTable(df; task_id_cols = [:location, :horizon])
model_ids(ft)
```
"""
model_ids(ft::ForecastTable) = unique(ft.data[!, ft.model_id_col])

"""
    output_type(ft::ForecastTable) -> Symbol

The single output_type present in `ft`. Throws if the table mixes output types,
since most ensemble methods are defined on a single type at a time.

# Arguments

- `ft`: a [`ForecastTable`](@ref).

# Examples

```@example
using ForecastEnsembles, DataFrames
df = DataFrame(
    location = "A", horizon = 1,
    model_id = repeat(["m1", "m2", "m3"], inner = 2),
    output_type = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5]
)
ft = ForecastTable(df; task_id_cols = [:location, :horizon])
output_type(ft)
```
"""
function output_type(ft::ForecastTable)
    types = unique(ft.data.output_type)
    length(types) == 1 || throw(
        ArgumentError(
        "ForecastTable contains mixed output_types $types; split before combining.",
    ),
    )
    return types[1]
end

# ---- Tables.jl interface ----

Tables.istable(::Type{ForecastTable}) = true
Tables.columnaccess(::Type{ForecastTable}) = true
# Zero-copy, as the Tables.jl contract expects: the returned columns alias the
# table's backing store, so mutating them in place corrupts `ft`. Use
# `DataFrame(ft)` (above) for a copy you own.
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
