# Initialise the Julia bridge

Activates the bundled \`inst/julia/\` project (which pins the exact
versions of Ensembles.jl, DataFrames.jl, etc. that this package was
tested against), starts the JuliaConnectoR server with that project
active, and loads the bridge helper functions. Subsequent calls are
no-ops.

## Usage

``` r
julia_setup(julia_bindir = NULL, ensembles_jl_path = NULL)
```

## Arguments

- julia_bindir:

  Path to the directory containing the \`julia\` executable. If
  supplied, sets \`JULIACONNECTOR_JULIABIN\` so the bundled
  JuliaConnectoR uses that binary.

- ensembles_jl_path:

  Optional path to a checkout of Ensembles.jl. When supplied, the
  bundled project is reconfigured to develop that source instead of the
  version pinned in the manifest.

## Value

Invisible NULL.

## Details

Heavy lifting (binary detection, subprocess instantiate, lazy-init
guard) is delegated to \[juliaready::julia_ready()\].

## Examples

``` r
if (FALSE) { # \dontrun{
# Default: pick up the bundled bridge project and a Julia binary from
# the env or PATH.
julia_setup()

# Develop a local checkout of Ensembles.jl into the bundled project.
julia_setup(ensembles_jl_path = "~/code/ensembles.jl")
} # }
```
