# Converters between the hubverse-aligned ForecastTable schema and the
# alternative shapes used by scoringutils (R) and lopensemble (R).

"""
    from_scoringutils(df; task_id_cols = nothing) -> ForecastTable

Convert a `scoringutils::forecast_quantile`-shaped frame
(columns `model`, `quantile_level`, `predicted`, plus task vars) to a
`ForecastTable` with `output_type = :quantile`.
"""
function from_scoringutils(df; task_id_cols = nothing)
    out = DataFrame(df)
    rename!(out, :model => :model_id, :quantile_level => :output_type_id, :predicted => :value)
    out.output_type = fill(:quantile, nrow(out))
    # observed is metadata for scoring, not part of the forecast — drop it
    if hasproperty(out, :observed)
        select!(out, Not(:observed))
    end
    return ForecastTable(out; task_id_cols = task_id_cols)
end

"""
    from_samples(df; task_id_cols = nothing,
                  model_col = :model, sample_col = :sample, value_col = :predicted)
    -> ForecastTable

Convert a sample-shaped frame (one row per draw) to a `ForecastTable` with
`output_type = :sample`. Mirrors the input expected by
`lopensemble::mixture_from_samples`.
"""
function from_samples(df;
                      task_id_cols = nothing,
                      model_col::Symbol = :model,
                      sample_col::Symbol = :sample,
                      value_col::Symbol = :predicted)
    out = DataFrame(df)
    rename!(out, model_col => :model_id, sample_col => :output_type_id, value_col => :value)
    out.output_type = fill(:sample, nrow(out))
    if hasproperty(out, :observed)
        select!(out, Not(:observed))
    end
    return ForecastTable(out; task_id_cols = task_id_cols)
end
