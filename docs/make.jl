using LR_BUG_Sylvester
using Documenter

makedocs(;
    modules = [LR_BUG_Sylvester],
    authors = "Giorgos Vretinaris",
    sitename = "LR_BUG_Sylvester.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://gvretina.github.io/LR_BUG_Sylvester.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Experiments" => "experiments.md",
        "Linear solves" => "linear_solves.md",
        "API" => "api.md",
    ],
)

deploydocs(; repo = "github.com/gvretina/LR_BUG_Sylvester.jl", devbranch = "main")
