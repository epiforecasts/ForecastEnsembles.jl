module ForecastEnsembles

using DataFrames: DataFrames, AbstractDataFrame, DataFrame, Not, innerjoin,
                  leftjoin, nrow, rename!, select!, unstack
using Distributions: Normal, Beta, fit_mle, params
using HiGHS: HiGHS
using JuMP: JuMP, @constraint, @objective, @variable, AffExpr, MOI, Model,
            add_to_expression!, optimize!, set_silent, termination_status, value
using LinearAlgebra: dot
using Optim: Optim, LBFGS, optimize
using Random: AbstractRNG, default_rng
using Statistics: Statistics
using StatsBase: StatsBase, mean, median
using Tables: Tables

import DataFrames: combine
import Distributions: cdf
import Statistics: quantile
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
       LinearPool,        # documented alias; MixtureEnsemble is primary
       LogarithmicPool,
       TrimmedMean,
       QRA,
       FittedQRA,
       CRPSStacking,
       FittedCRPSStacking,
       BLP,
       FittedBLP,
       Stacking,
       FittedStacking,
       InverseScore,
       FittedInverseScore,
       Hedge,
       FittedHedge,
       PartialPooling,
       FittedPartialPooling,
       Windowed,
       combine,
       fit,
       weights,
       backtest,
       effective_num_models,
       weight_stability

include("output_types.jl")
include("forecast_table.jl")
include("weights.jl")
include("methods.jl")
include("simple.jl")
include("distfromq.jl")
include("linear_pool.jl")
include("logarithmic_pool.jl")
include("blp.jl")
include("trimmed.jl")
include("qra.jl")
include("crps_stacking.jl")
include("stacking.jl")
include("online.jl")
include("partial_pooling.jl")
include("diagnostics.jl")
include("backtest.jl")
include("windowed.jl")
include("interop.jl")

end # module
