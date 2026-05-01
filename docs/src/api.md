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
combine(::ForecastTable, ::Ensembles.FittedQRA)
combine(::ForecastTable, ::Ensembles.FittedCRPSStacking)
fit(::QRA, ::ForecastTable, ::AbstractDataFrame)
fit(::CRPSStacking, ::ForecastTable, ::AbstractDataFrame)
```

## Internals

```@docs
Ensembles.QuantileDistribution
Ensembles.FittedQRA
Ensembles.FittedCRPSStacking
```
