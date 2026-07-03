using Documenter
using ForecastEnsembles
using DataFrames    # `fit(::QRA, ::ForecastTable, ::AbstractDataFrame)` etc.

DocMeta.setdocmeta!(
    ForecastEnsembles,
    :DocTestSetup,
    :(using ForecastEnsembles, DataFrames);
    recursive = true,
)

makedocs(;
    modules = [ForecastEnsembles],
    authors = "Sebastian Funk",
    sitename = "ForecastEnsembles.jl",
    format = Documenter.HTML(;
        canonical = "https://sbfnk.github.io/ForecastEnsembles.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Methods" => "methods.md",
        "Worked example" => "example.md",
        "Extending" => "extending.md",
        "Roadmap" => "roadmap.md",
        "API" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(; repo = "github.com/sbfnk/ForecastEnsembles.jl", devbranch = "main")
