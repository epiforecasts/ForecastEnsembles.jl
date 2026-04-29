module Ensembles

using DataFrames
using Statistics
using StatsBase
using Tables

import DataFrames: combine
import StatsBase: fit

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
       FittedQRA,
       CRPSStacking,
       FittedCRPSStacking,
       combine,
       fit

include("output_types.jl")
include("forecast_table.jl")
include("methods.jl")
include("simple.jl")
include("distfromq.jl")
include("linear_pool.jl")
include("qra.jl")
include("crps_stacking.jl")
include("interop.jl")

end # module
