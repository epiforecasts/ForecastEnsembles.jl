.pkg_env <- new.env(parent = emptyenv())
.pkg_env$julia_bin <- NULL
.pkg_env$bridge_project <- NULL
.pkg_env$runner_script <- NULL

.ensure_setup <- function() {
  if (!is.null(.pkg_env$julia_bin)) return(invisible(NULL))
  julia_setup()
  invisible(NULL)
}

# Resolve where the bridge Project.toml + runner.jl live. After installation
# they are inside the package's `inst/julia/` directory. During development
# they sit next to the package source.
.resolve_bridge <- function() {
  candidates <- c(
    system.file("julia", package = "ensembles"),
    file.path(getwd(), "..", "inst", "julia"),
    file.path(getwd(), "inst", "julia")
  )
  for (p in candidates) {
    if (nzchar(p) && file.exists(file.path(p, "runner.jl"))) {
      return(normalizePath(p, mustWork = TRUE))
    }
  }
  stop("Could not find inst/julia/runner.jl. Reinstall the ensembles package.",
       call. = FALSE)
}
