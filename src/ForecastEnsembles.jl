module ForecastEnsembles

using DataFrames
using LinearAlgebra: dot
using Random: AbstractRNG, default_rng
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
       LinearPool,        # alias for MixtureEnsemble
       QRA,
       FittedQRA,
       CRPSStacking,
       FittedCRPSStacking,
       combine,
       fit,
       weights,
       ScoringRule,
       CRPS,
       WIS,
       score,
       mean_score,
       backtest

include("output_types.jl")
include("forecast_table.jl")
include("weights.jl")
include("methods.jl")
include("simple.jl")
include("distfromq.jl")
include("linear_pool.jl")
include("qra.jl")
include("crps_stacking.jl")
include("scores.jl")
include("backtest.jl")
include("interop.jl")

end # module
