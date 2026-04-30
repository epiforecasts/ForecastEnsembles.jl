# Bridge script for the R `ensembles` package.
#
# Reads a JSON spec (path passed as argv[1]) describing the operation and
# input file paths, runs Ensembles.jl, and writes the output as CSV at the
# path given in the spec. Each invocation is one operation — startup cost
# (~5–10 s with the full LP/Optim deps) is amortised by the caller.
#
# Spec schema (extra fields ignored):
#
#   {
#     "op": "simple_ensemble" | "linear_pool" | "qra" | "crps_weights",
#     "input":         "<path-to-input.csv>",
#     "weights":       "<path-to-weights.csv>"   (optional),
#     "training":      "<path-to-training.csv>"  (qra/crps_weights),
#     "target":        "<path-to-target.csv>"    (qra),
#     "observations":  "<path-to-obs.csv>"       (qra/crps_weights),
#     "task_id_cols":  ["loc", "horizon", ...],
#     "agg":           "mean" | "median",        (simple_ensemble),
#     "n_samples":     10000,                    (linear_pool),
#     "per_quantile_weights": false, "intercept": false,
#     "enforce_normalisation": true, "noncross": true,
#     "group":         ["loc", ...],
#     "dirichlet_alpha": 1.001,
#     "output":        "<path-to-output.csv>"
#   }

using JSON
using CSV
using DataFrames
using Ensembles

const SPEC = JSON.parsefile(ARGS[1])

read_input(path) = begin
    df = CSV.read(path, DataFrame; stringtype = String)
    if hasproperty(df, :output_type) && !(eltype(df.output_type) <: Symbol)
        df.output_type = Symbol.(df.output_type)
    end
    df
end

task_cols = Symbol.(SPEC["task_id_cols"])

build_ft(df) = ForecastTable(df; task_id_cols = task_cols)

function run_simple()
    df  = read_input(SPEC["input"])
    ft  = build_ft(df)
    agg = Symbol(SPEC["agg"])
    method = if haskey(SPEC, "weights") && SPEC["weights"] !== nothing
        SimpleEnsemble(agg; weights = read_input(SPEC["weights"]))
    else
        SimpleEnsemble(agg)
    end
    out = combine(ft, method)
    write_out(out)
end

function run_linear_pool()
    df = read_input(SPEC["input"])
    ft = build_ft(df)
    n  = Int(get(SPEC, "n_samples", 10_000))
    method = if haskey(SPEC, "weights") && SPEC["weights"] !== nothing
        LinearPool(; n_samples = n, weights = read_input(SPEC["weights"]))
    else
        LinearPool(; n_samples = n)
    end
    out = combine(ft, method)
    write_out(out)
end

function run_qra()
    train_ft  = build_ft(read_input(SPEC["training"]))
    target_ft = build_ft(read_input(SPEC["target"]))
    obs       = read_input(SPEC["observations"])
    method = QRA(;
        per_quantile_weights  = Bool(get(SPEC, "per_quantile_weights", false)),
        intercept             = Bool(get(SPEC, "intercept", false)),
        enforce_normalisation = Bool(get(SPEC, "enforce_normalisation", true)),
        noncross              = Bool(get(SPEC, "noncross", true)),
        group                 = Symbol.(get(SPEC, "group", String[])),
    )
    fitted = fit(method, train_ft, obs)
    out = combine(target_ft, fitted)
    write_out(out)
end

function run_crps()
    train_ft = build_ft(read_input(SPEC["training"]))
    obs      = read_input(SPEC["observations"])
    method = CRPSStacking(;
        dirichlet_alpha = Float64(get(SPEC, "dirichlet_alpha", 1.001)),
    )
    fitted = fit(method, train_ft, obs)
    CSV.write(SPEC["output"], fitted.weights)
end

function write_out(out_ft::ForecastTable)
    df = DataFrame(out_ft)
    if hasproperty(df, :output_type)
        df.output_type = String.(df.output_type)
    end
    CSV.write(SPEC["output"], df)
end

const OP = SPEC["op"]
if OP == "simple_ensemble"
    run_simple()
elseif OP == "linear_pool"
    run_linear_pool()
elseif OP == "qra"
    run_qra()
elseif OP == "crps_weights"
    run_crps()
else
    error("unknown op: $OP")
end
