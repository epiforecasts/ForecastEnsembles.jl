# API reference

## Forecast tables

```@docs
ForecastTable
output_type
```

## Methods

```@docs
QuantileEnsemble
MixtureEnsemble
QRA
CRPSStacking
```

## Verbs

```@docs
combine(::ForecastTable, ::QuantileEnsemble)
combine(::ForecastTable, ::MixtureEnsemble)
combine(::ForecastTable, ::ForecastEnsembles.FittedQRA)
combine(::ForecastTable, ::ForecastEnsembles.FittedCRPSStacking)
fit(::QRA, ::ForecastTable, ::AbstractDataFrame)
fit(::CRPSStacking, ::ForecastTable, ::AbstractDataFrame)
```

## Internals

```@docs
ForecastEnsembles.QuantileDistribution
ForecastEnsembles.FittedQRA
ForecastEnsembles.FittedCRPSStacking
```
