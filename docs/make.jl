using SLCEMonteCarlo
using SLCE   # the SCE fitting core, for the executed `@example` model builds
using Documenter

DocMeta.setdocmeta!(SLCEMonteCarlo, :DocTestSetup, :(using SLCEMonteCarlo);
                    recursive = true)

makedocs(;
    sitename = "SLCEMonteCarlo.jl",
    modules = [SLCEMonteCarlo],
    # The SLCE dependency is a path-dev without a resolvable remote in this
    # build, so per-line source/edit links stay disabled; the navbar links to the
    # repository (private: github.com/Tomonori-Tanaka/SLCEMonteCarlo.jl).
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        edit_link = nothing,
        repolink = "https://github.com/Tomonori-Tanaka/SLCEMonteCarlo.jl",
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
