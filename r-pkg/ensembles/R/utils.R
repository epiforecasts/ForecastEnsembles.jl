.julia_to_df <- function(jl_named_list) {
  cols <- JuliaConnectoR::juliaGet(jl_named_list)
  as.data.frame(cols, stringsAsFactors = FALSE)
}
