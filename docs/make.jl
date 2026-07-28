using SLCEMonteCarlo
using SLCE   # the SLCE fitting core, for the executed `@example` model builds
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(SLCEMonteCarlo, :DocTestSetup, :(using SLCEMonteCarlo);
                    recursive = true)

makedocs(;
    sitename = "SLCEMonteCarlo.jl",
    modules = [SLCEMonteCarlo],
    repo = Remotes.GitHub("Tomonori-Tanaka", "SLCEMonteCarlo.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        canonical = "https://tomonori-tanaka.github.io/SLCEMonteCarlo.jl/dev",
        edit_link = "main",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "tutorials/cubic_heisenberg.md",
        ],
        "Guide" => [
            "guide/running.md",
            "guide/parallel_tempering.md",
            "guide/ground_states.md",
            "guide/parallelism.md",
            "guide/gpu.md",
            "guide/observables.md",
            "guide/checkpointing.md",
        ],
        "Theory" => [
            "theory/updates.md",
            "theory/binning.md",
        ],
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
    doctest = false,
)

# Publishes to https://tomonori-tanaka.github.io/SLCEMonteCarlo.jl/ from the `documentation build`
# CI job (which needs `permissions: contents: write`). Outside CI this is a no-op, so a
# local `julia --project=docs docs/make.jl` still just builds into `docs/build/`.
deploydocs(;
    repo = "github.com/Tomonori-Tanaka/SLCEMonteCarlo.jl",
    devbranch = "main",
    push_preview = false,
)
