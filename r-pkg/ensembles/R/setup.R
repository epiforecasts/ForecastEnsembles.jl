#' Initialise the Julia bridge
#'
#' Starts a Julia process via JuliaConnectoR (TCP-bridged, out-of-process),
#' activates the bundled bridge project, and brings Ensembles.jl into the
#' Julia session. Subsequent calls in the same R session are no-ops.
#'
#' @param julia_bindir Path to the directory containing the `julia`
#'   executable. Defaults to whatever JuliaConnectoR finds via the
#'   `JULIA_BINDIR` env var or `PATH`.
#' @param ensembles_jl_path Optional path to a checkout of Ensembles.jl.
#'   When supplied, the bridge project is reconfigured to develop that
#'   source instead of the version bundled at install time.
#'
#' @return Invisible NULL.
#' @export
julia_setup <- function(julia_bindir = NULL, ensembles_jl_path = NULL) {
  if (.pkg_env$initialised) return(invisible(NULL))

  if (!is.null(julia_bindir)) Sys.setenv(JULIA_BINDIR = julia_bindir)

  bridge <- .resolve_bridge()
  bridge_jl <- gsub("\\\\", "/", bridge, fixed = TRUE)

  JuliaConnectoR::juliaEval(sprintf(
    'using Pkg; Pkg.activate("%s"); nothing', bridge_jl
  ))

  if (!is.null(ensembles_jl_path)) {
    abs <- normalizePath(ensembles_jl_path, mustWork = TRUE)
    abs <- gsub("\\\\", "/", abs, fixed = TRUE)
    JuliaConnectoR::juliaEval(sprintf(
      'using Pkg; Pkg.develop(path="%s"); nothing', abs
    ))
  }

  JuliaConnectoR::juliaEval('using Ensembles, DataFrames; nothing')

  # Define all helper functions in one shot so subsequent calls don't pay
  # the parse cost.
  JuliaConnectoR::juliaEval(.bridge_helpers_jl())

  .pkg_env$initialised <- TRUE
  invisible(NULL)
}

.bridge_helpers_jl <- function() {
'
function _ens_to_string!(df::DataFrame)
    if hasproperty(df, :output_type) && eltype(df.output_type) == Symbol
        df.output_type = String.(df.output_type)
    end
    df
end

# Convert a DataFrame to a NamedTuple of vectors so JuliaConnectoR
# round-trips it as a proper named list (which R then turns into a
# data.frame trivially).
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
    ft = Ensembles.ForecastTable(df; task_id_cols = cols)
    method = if weights_in === nothing
        Ensembles.SimpleEnsemble(Symbol(agg))
    else
        Ensembles.SimpleEnsemble(Symbol(agg); weights = DataFrame(weights_in))
    end
    _ens_pack(_ens_to_string!(DataFrame(combine(ft, method))))
end

function _ens_linear_pool(df_in, task_id_cols::Vector, n_samples::Int, weights_in)
    df = _ens_to_symbol!(DataFrame(df_in))
    cols = Symbol.(task_id_cols)
    ft = Ensembles.ForecastTable(df; task_id_cols = cols)
    method = if weights_in === nothing
        Ensembles.LinearPool(; n_samples = n_samples)
    else
        Ensembles.LinearPool(; n_samples = n_samples, weights = DataFrame(weights_in))
    end
    _ens_pack(_ens_to_string!(DataFrame(combine(ft, method))))
end

function _ens_qra(train_in, target_in, obs_in, task_id_cols::Vector,
                  per_quantile_weights::Bool, intercept::Bool,
                  enforce_normalisation::Bool, noncross::Bool,
                  group::Vector)
    cols = Symbol.(task_id_cols)
    train_df  = _ens_to_symbol!(DataFrame(train_in))
    target_df = _ens_to_symbol!(DataFrame(target_in))
    obs_df    = DataFrame(obs_in)
    train_ft  = Ensembles.ForecastTable(train_df;  task_id_cols = cols)
    target_ft = Ensembles.ForecastTable(target_df; task_id_cols = cols)
    method = Ensembles.QRA(;
        per_quantile_weights = per_quantile_weights,
        intercept = intercept,
        enforce_normalisation = enforce_normalisation,
        noncross = noncross,
        group = Symbol.(group),
    )
    fitted = fit(method, train_ft, obs_df)
    _ens_pack(_ens_to_string!(DataFrame(combine(target_ft, fitted))))
end

function _ens_crps(train_in, obs_in, task_id_cols::Vector, dirichlet_alpha::Float64)
    cols = Symbol.(task_id_cols)
    train_ft = Ensembles.ForecastTable(_ens_to_symbol!(DataFrame(train_in));
                                        task_id_cols = cols)
    obs_df   = DataFrame(obs_in)
    method = Ensembles.CRPSStacking(; dirichlet_alpha = dirichlet_alpha)
    fitted = fit(method, train_ft, obs_df)
    _ens_pack(DataFrame(fitted.weights))
end

nothing
'
}
