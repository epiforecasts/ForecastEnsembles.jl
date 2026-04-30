# API reference

## Forecast tables

```@docs
ForecastTable
output_type
```

## Methods

```@docs
SimpleEnsemble
LinearPool
QRA
CRPSStacking
```

## Verbs

```@docs
combine(::ForecastTable, ::SimpleEnsemble)
combine(::ForecastTable, ::LinearPool)
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
