#' Initialise the Julia bridge
#'
#' Locates a Julia executable and resolves paths to the bundled bridge
#' project and runner script. Each public function in `ensembles`
#' subsequently spawns a short-lived Julia process for the actual work.
#' Call this once per session if you need to control which Julia is used or
#' to point at a custom Ensembles.jl source tree.
#'
#' @param julia_bin Path to the `julia` executable. Defaults to the first
#'   `julia` on `PATH`.
#' @param ensembles_jl_path Optional path to a checkout of Ensembles.jl. If
#'   supplied, the bridge project is reconfigured to develop that source
#'   instead of the one bundled at install time. Useful when iterating on
#'   the Julia package alongside the R wrapper.
#'
#' @return Invisible NULL.
#' @export
julia_setup <- function(julia_bin = NULL, ensembles_jl_path = NULL) {
  if (is.null(julia_bin)) {
    julia_bin <- Sys.which("julia")
    if (!nzchar(julia_bin)) {
      stop("`julia` not found on PATH. Install Julia (>= 1.10) or pass `julia_bin`.",
           call. = FALSE)
    }
  }

  bridge <- .resolve_bridge()
  if (!is.null(ensembles_jl_path)) {
    abs <- normalizePath(ensembles_jl_path, mustWork = TRUE)
    res <- system2(
      julia_bin,
      args = c(sprintf("--project=%s", bridge), "-e",
               shQuote(sprintf('using Pkg; Pkg.develop(path="%s")', abs))),
      stdout = TRUE, stderr = TRUE
    )
  }

  .pkg_env$julia_bin <- unname(julia_bin)
  .pkg_env$bridge_project <- bridge
  .pkg_env$runner_script <- file.path(bridge, "runner.jl")
  invisible(NULL)
}
