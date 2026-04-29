module Ensembles

using DataFrames
using Statistics
using StatsBase
using Tables

import DataFrames: combine

export ForecastTable,
       output_type,
       task_id_cols,
       model_ids,
       EnsembleMethod,
       UnfittedMethod,
       TrainedMethod,
       SimpleEnsemble,
       LinearPool,
       QRA,
       CRPSStacking,
       combine,
       fit

include("output_types.jl")
include("forecast_table.jl")
include("methods.jl")
include("simple.jl")
include("interop.jl")

end # module
