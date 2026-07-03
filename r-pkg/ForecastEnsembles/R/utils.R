.julia_to_df <- function(jl_named_list) {
  cols <- JuliaConnectoR::juliaGet(jl_named_list)
  as.data.frame(cols, stringsAsFactors = FALSE)
}

# ---- R-side pre-flight validation -------------------------------------
# All numerical validation happens in Julia, but a Julia backtrace is not a
# helpful error for an R user. These checks catch the common mistakes
# (wrong column names, missing arguments, malformed weights) before any
# data crosses the bridge, with messages that say what to do.

.required_cols <- c("model_id", "output_type", "output_type_id", "value")

# Column names used by the R packages this one mirrors, mapped to ours.
.col_hints <- c(
  model = "model_id",
  quantile_level = "output_type_id",
  sample_id = "output_type_id",
  sample = "output_type_id",
  predicted = "value"
)

.validate_forecast_df <- function(df, arg) {
  if (!is.data.frame(df)) {
    stop("`", arg, "` must be a data frame, got ", class(df)[1], ".",
         call. = FALSE)
  }
  missing_cols <- setdiff(.required_cols, names(df))
  if (length(missing_cols) > 0) {
    hints <- .col_hints[names(.col_hints) %in% names(df)]
    hint_txt <- if (length(hints) > 0) {
      paste0("\nIt looks like ", arg, " uses scoringutils/lopensemble-style ",
             "names; rename ",
             paste0("`", names(hints), "` -> `", hints, "`", collapse = ", "),
             ".")
    } else {
      ""
    }
    stop("`", arg, "` is missing required column(s): ",
         paste(missing_cols, collapse = ", "),
         ".\nRequired: model_id, output_type, output_type_id, value, ",
         "plus your task-id columns.", hint_txt,
         call. = FALSE)
  }
  invisible(df)
}

.validate_task_id_cols <- function(task_id_cols, df, arg = "model_out_tbl") {
  if (missing(task_id_cols) || is.null(task_id_cols)) {
    stop("`task_id_cols` is required: the column names that identify a ",
         "forecast task, e.g. c(\"location\", \"horizon\", \"target_date\").",
         call. = FALSE)
  }
  absent <- setdiff(task_id_cols, names(df))
  if (length(absent) > 0) {
    stop("task_id_cols not present in `", arg, "`: ",
         paste(absent, collapse = ", "), call. = FALSE)
  }
  invisible(task_id_cols)
}

.validate_weights <- function(weights) {
  if (is.null(weights)) return(invisible(NULL))
  if (!is.data.frame(weights)) {
    stop("`weights` must be a data frame with columns `model_id` and ",
         "`weight` (got ", class(weights)[1], ").", call. = FALSE)
  }
  needed <- setdiff(c("model_id", "weight"), names(weights))
  if (length(needed) > 0) {
    stop("`weights` is missing column(s): ", paste(needed, collapse = ", "),
         ". Expected `model_id` and `weight` (optionally `output_type_id` ",
         "for per-quantile weights).", call. = FALSE)
  }
  if (any(is.na(weights$weight)) || any(weights$weight < 0)) {
    stop("`weights$weight` must be non-negative and non-missing. Weights ",
         "do not need to sum to 1; they are normalised internally.",
         call. = FALSE)
  }
  invisible(weights)
}

.validate_observations <- function(observations, task_id_cols) {
  if (!is.data.frame(observations)) {
    stop("`observations` must be a data frame, got ",
         class(observations)[1], ".", call. = FALSE)
  }
  if (!"observed" %in% names(observations)) {
    stop("`observations` must contain a column named exactly `observed` ",
         "with the realised values.", call. = FALSE)
  }
  absent <- setdiff(task_id_cols, names(observations))
  if (length(absent) > 0) {
    stop("`observations` is missing task-id column(s): ",
         paste(absent, collapse = ", "), call. = FALSE)
  }
  invisible(observations)
}

# Tag output with the hubverse model_out_tbl class so downstream hubUtils
# tooling treats it as hub-format data.
.as_model_out_tbl <- function(df) {
  class(df) <- unique(c("model_out_tbl", class(df)))
  df
}
