.pkg_env <- new.env(parent = emptyenv())
.pkg_env$initialised <- FALSE

.ensure_setup <- function() {
  if (.pkg_env$initialised) return(invisible(NULL))
  julia_setup()
  invisible(NULL)
}

# Resolve where the bridge Project.toml lives. After installation it is
# inside the package's `inst/julia/`. During development it sits next to
# the package source.
.resolve_bridge <- function() {
  candidates <- c(
    system.file("julia", package = "ensembles"),
    file.path(getwd(), "..", "inst", "julia"),
    file.path(getwd(), "inst", "julia")
  )
  for (p in candidates) {
    if (nzchar(p) && file.exists(file.path(p, "Project.toml"))) {
      return(normalizePath(p, mustWork = TRUE))
    }
  }
  stop("Could not find inst/julia/Project.toml. Reinstall the ensembles package.",
       call. = FALSE)
}

.onLoad <- function(libname, pkgname) {
  reg.finalizer(.pkg_env, function(env) {
    tryCatch(JuliaConnectoR::stopJulia(), error = function(e) NULL)
  }, onexit = TRUE)
}
