module ForecastEnsembles

using DataFrames: DataFrames, AbstractDataFrame, DataFrame, Not, innerjoin,
                  leftjoin, nrow, rename!, select!, unstack
using Distributions: Normal
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
       LinearPool,        # alias for MixtureEnsemble
       QRA,
       FittedQRA,
       CRPSStacking,
       FittedCRPSStacking,
       Stacking,
       FittedStacking,
       InverseScore,
       FittedInverseScore,
       Hedge,
       FittedHedge,
       Windowed,
       combine,
       fit,
       weights,
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
include("stacking.jl")
include("online.jl")
include("backtest.jl")
include("windowed.jl")
include("interop.jl")

end # module
