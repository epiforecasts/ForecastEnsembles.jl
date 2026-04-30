#' Initialise the Julia bridge
#'
#' Activates the bundled `inst/julia/` project (which pins the exact
#' versions of Ensembles.jl, DataFrames.jl, etc. that this package was
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
#' @param ensembles_jl_path Optional path to a checkout of Ensembles.jl.
#'   When supplied, the bundled project is reconfigured to develop that
#'   source instead of the version pinned in the manifest.
#'
#' @return Invisible NULL.
#' @export
julia_setup <- function(julia_bindir = NULL, ensembles_jl_path = NULL) {
  if (isTRUE(.pkg_env$ready)) return(invisible(NULL))

  if (!is.null(julia_bindir)) {
    bin <- file.path(julia_bindir,
                     if (.Platform$OS.type == "windows") "julia.exe" else "julia")
    Sys.setenv(JULIACONNECTOR_JULIABIN = bin)
  }

  bridge <- .resolve_bridge()

  # If a development checkout of Ensembles.jl was supplied, dev it into
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
    packages  = c("Ensembles", "DataFrames"),
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
