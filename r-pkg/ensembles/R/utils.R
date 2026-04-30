# Subprocess bridge: write JSON spec + CSVs to a tempdir, run julia, read
# the resulting CSV. Each call pays a Julia startup of ~5–10 s (more on the
# first invocation in a session if the env needs precompilation).

.write_csv_temp <- function(df, name, dir) {
  path <- file.path(dir, paste0(name, ".csv"))
  utils::write.csv(df, path, row.names = FALSE)
  path
}

.run_julia <- function(spec) {
  .ensure_setup()
  json <- jsonlite::toJSON(spec, auto_unbox = TRUE, null = "null", na = "null")
  spec_path <- file.path(spec$.tempdir, "spec.json")
  writeLines(json, spec_path)
  res <- system2(
    .pkg_env$julia_bin,
    args = c(
      sprintf("--project=%s", .pkg_env$bridge_project),
      .pkg_env$runner_script,
      spec_path
    ),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop("Julia bridge failed:\n", paste(res, collapse = "\n"), call. = FALSE)
  }
  out_path <- spec$output
  if (!file.exists(out_path)) {
    stop("Julia bridge produced no output at ", out_path, call. = FALSE)
  }
  utils::read.csv(out_path, stringsAsFactors = FALSE)
}

.with_tempdir <- function(fun) {
  d <- tempfile("ensembles_")
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  fun(d)
}
