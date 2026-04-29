# Output types follow the hubverse `model_out_tbl` vocabulary.
# We keep them as Symbols at the data layer (matching the on-disk convention)
# and as Val-tagged singletons internally for dispatch.

const KNOWN_OUTPUT_TYPES = (:quantile, :sample, :mean, :median, :cdf, :pmf)

is_known_output_type(s::Symbol) = s in KNOWN_OUTPUT_TYPES

# Internal dispatch tag built from the on-disk symbol.
output_type_tag(s::Symbol) = Val(s)

# Output types whose `output_type_id` carries a numeric meaning
# (quantile level, CDF threshold, …) versus a sample/draw index.
is_numeric_id(s::Symbol) = s in (:quantile, :cdf, :pmf)
