using Documenter
using Ensembles
using DataFrames    # `fit(::QRA, ::ForecastTable, ::AbstractDataFrame)` etc.

DocMeta.setdocmeta!(Ensembles, :DocTestSetup,
                    :(using Ensembles, DataFrames); recursive = true)

makedocs(;
    modules  = [Ensembles],
    authors  = "Sebastian Funk",
    sitename = "Ensembles.jl",
    format   = Documenter.HTML(;
        canonical = "https://sbfnk.github.io/ensembles.jl",
        edit_link = "main",
    ),
    pages    = [
        "Home"    => "index.md",
        "Methods" => "methods.md",
        "Worked example" => "example.md",
        "Extending" => "extending.md",
        "Roadmap" => "roadmap.md",
        "API"     => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(;
    repo      = "github.com/sbfnk/ensembles.jl",
    devbranch = "main",
)
