# Bundled example data

`flu_forecasts.csv` is a slice of three models from the hubverse
[example-complex-forecast-hub](https://github.com/hubverse-org/example-complex-forecast-hub)
for 2022-12-17, restricted to the 23 quantile levels at horizon 1 for five
US locations (national + CA, FL, NY, TX). Original schema preserved
verbatim.

Columns: `model_id, reference_date, horizon, target_end_date, location,
target, output_type, output_type_id, value`.

Models:

- `Flusight-baseline` — naive baseline.
- `MOBS-GLEAM_FLUH`   — the MOBS-Lab metapopulation model.
- `PSI-DICE`          — PSI's Dynamic Influenza Causal Engine.

License: see the upstream repo's licence (CC0 in the model output files).
