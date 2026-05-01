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
       EnsembleWeights,
       QuantileEnsemble,
       MixtureEnsemble,
       SimpleEnsemble,    # alias for QuantileEnsemble
       LinearPool,        # alias for MixtureEnsemble
       QRA,
       FittedQRA,
       CRPSStacking,
       FittedCRPSStacking,
       combine,
       fit,
       weights

include("output_types.jl")
include("forecast_table.jl")
include("weights.jl")
include("methods.jl")
include("simple.jl")
include("distfromq.jl")
include("linear_pool.jl")
include("qra.jl")
include("crps_stacking.jl")
include("interop.jl")

end # module
