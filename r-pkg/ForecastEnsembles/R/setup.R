#' Initialise the Julia bridge
#'
#' Activates the bundled `inst/julia/` project (which pins the exact
#' versions of ForecastEnsembles.jl, DataFrames.jl, etc. that this package was
#' tested against), starts the JuliaConnectoR server with that project
#' active, and loads the bridge helper functions. Subsequent calls are
#' no-ops.
#'
#' Heavy lifting (binary detection, subprocess instantiate, lazy-init
#' guard) is delegated to [juliaready::julia_ready()].
#'
#' @param julia_bindir Path to the directory containing the `julia`
#'   executable. If supplied, sets `JULIACONNECTOR_JULIABIN` so the
#'   bundled JuliaConnectoR uses that binary.
#' @param ensembles_jl_path Optional path to a checkout of ForecastEnsembles.jl.
#'   When supplied, the bundled project is reconfigured to develop that
#'   source instead of the version pinned in the manifest. Defaults to
#'   `getOption("ForecastEnsembles.jl_path")`, so a session-wide
#'   `options(ForecastEnsembles.jl_path = "...")` (e.g. set by the test helper or
#'   in CI) is picked up automatically.
#'
#' @section Startup time:
#' Two distinct delays, easy to conflate:
#' \itemize{
#'   \item The very first use on a machine instantiates and precompiles the
#'     bundled Julia project (LP solver, optimiser, ForecastEnsembles.jl). This can
#'     take a few minutes and then stays cached in the Julia depot.
#'   \item Every fresh R session pays a Julia startup of roughly 10--15
#'     seconds on the first call to any function in this package. Later
#'     calls in the same session run in about a second.
#' }
#' Requires Julia (>= 1.10) on the `PATH`, or `julia_bindir`. If Julia is
#' not installed, install it via juliaup
#' (\url{https://github.com/JuliaLang/juliaup}) before using this package.
#'
#' @return Invisible NULL.
#' @examples
#' \dontrun{
#' # Default: pick up the bundled bridge project and a Julia binary from
#' # the env or PATH.
#' julia_setup()
#'
#' # Develop a local checkout of ForecastEnsembles.jl into the bundled project.
#' julia_setup(ensembles_jl_path = "~/code/ForecastEnsembles.jl")
#' }
#' @export
julia_setup <- function(julia_bindir = NULL,
                        ensembles_jl_path = getOption("ForecastEnsembles.jl_path")) {
  if (isTRUE(.pkg_env$ready)) return(invisible(NULL))

  if (!is.null(julia_bindir)) {
    bin <- file.path(julia_bindir,
                     if (.Platform$OS.type == "windows") "julia.exe" else "julia")
    Sys.setenv(JULIACONNECTOR_JULIABIN = bin)
  }

  bridge <- .resolve_bridge()

  # If a development checkout of ForecastEnsembles.jl was supplied, dev it into
  # the bundled project before instantiation.
  if (!is.null(ensembles_jl_path)) {
    abs <- normalizePath(ensembles_jl_path, mustWork = TRUE)
    abs_jl <- gsub("\\\\", "/", abs, fixed = TRUE)
    bridge_jl <- gsub("\\\\", "/", bridge, fixed = TRUE)
    juliaready:::julia_subprocess(sprintf(
      'import Pkg; Pkg.activate("%s"); Pkg.develop(path="%s")',
      bridge_jl, abs_jl
    ))
  }

  juliaready::julia_ready(
    packages  = c("ForecastEnsembles", "DataFrames"),
    state_env = .pkg_env,
    project   = bridge,
    install   = FALSE,
    verbose   = FALSE
  )

  # Define all helper functions in one shot so subsequent calls don't pay
  # the parse cost.
  juliaready::eval_julia(.bridge_helpers_jl())

  invisible(NULL)
}

.bridge_helpers_jl <- function() {
'
import Random

function _ens_to_string!(df::DataFrame)
    if hasproperty(df, :output_type) && eltype(df.output_type) == Symbol
        df.output_type = String.(df.output_type)
    end
    df
end

_ens_pack(df::DataFrame) = (; (Symbol(c) => df[!, c] for c in names(df))...)

function _ens_to_symbol!(df::DataFrame)
    if hasproperty(df, :output_type) && !(eltype(df.output_type) <: Symbol)
        df.output_type = Symbol.(df.output_type)
    end
    df
end

function _ens_simple(df_in, task_id_cols::Vector, agg::String, weights_in)
    df = _ens_to_symbol!(DataFrame(df_in))
    cols = Symbol.(task_id_cols)
    ft = ForecastEnsembles.ForecastTable(df; task_id_cols = cols)
    method = if weights_in === nothing
        ForecastEnsembles.QuantileEnsemble(Symbol(agg))
    else
        ForecastEnsembles.QuantileEnsemble(Symbol(agg); weights = DataFrame(weights_in))
    end
    _ens_pack(_ens_to_string!(DataFrame(combine(ft, method))))
end

function _ens_linear_pool(df_in, task_id_cols::Vector, n_samples::Int, weights_in,
                          seed)
    df = _ens_to_symbol!(DataFrame(df_in))
    cols = Symbol.(task_id_cols)
    ft = ForecastEnsembles.ForecastTable(df; task_id_cols = cols)
    method = if weights_in === nothing
        ForecastEnsembles.LinearPool(; n_samples = n_samples)
    else
        ForecastEnsembles.LinearPool(; n_samples = n_samples, weights = DataFrame(weights_in))
    end
    rng = seed === nothing ? Random.default_rng() :
                              Random.MersenneTwister(Int(seed))
    _ens_pack(_ens_to_string!(DataFrame(combine(ft, method; rng = rng))))
end

function _ens_qra(train_in, target_in, obs_in, task_id_cols::Vector,
                  per_quantile_weights::Bool, intercept::Bool,
                  enforce_normalisation::Bool, noncross::Bool,
                  group::Vector)
    cols = Symbol.(task_id_cols)
    train_df  = _ens_to_symbol!(DataFrame(train_in))
    target_df = _ens_to_symbol!(DataFrame(target_in))
    obs_df    = DataFrame(obs_in)
    train_ft  = ForecastEnsembles.ForecastTable(train_df;  task_id_cols = cols)
    target_ft = ForecastEnsembles.ForecastTable(target_df; task_id_cols = cols)
    method = ForecastEnsembles.QRA(;
        per_quantile_weights = per_quantile_weights,
        intercept = intercept,
        enforce_normalisation = enforce_normalisation,
        noncross = noncross,
        group = Symbol.(group),
    )
    fitted = fit(method, train_ft, obs_df)
    pred = _ens_pack(_ens_to_string!(DataFrame(combine(target_ft, fitted))))
    w = ForecastEnsembles.weights(fitted)
    (pred = pred,
     weights = w === nothing ? nothing : _ens_pack(DataFrame(w)))
end

function _ens_crps(train_in, obs_in, task_id_cols::Vector, dirichlet_alpha::Float64,
                   lambda, time_col, task_weights_in)
    cols = Symbol.(task_id_cols)
    train_ft = ForecastEnsembles.ForecastTable(_ens_to_symbol!(DataFrame(train_in));
                                        task_id_cols = cols)
    obs_df   = DataFrame(obs_in)
    lam = lambda isa AbstractString ? Symbol(lambda) : lambda
    method = ForecastEnsembles.CRPSStacking(;
        dirichlet_alpha = dirichlet_alpha,
        lambda = lam,
        time_col = time_col === nothing ? nothing : Symbol(time_col),
        task_weights = task_weights_in === nothing ? nothing :
                       DataFrame(task_weights_in),
    )
    fitted = fit(method, train_ft, obs_df)
    _ens_pack(DataFrame(fitted.weights))
end

nothing
'
}
