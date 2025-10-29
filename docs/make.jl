using LR_BUG_Sylvester
using Documenter

DocMeta.setdocmeta!(LR_BUG_Sylvester, :DocTestSetup, :(using LR_BUG_Sylvester); recursive=true)

makedocs(;
    modules=[LR_BUG_Sylvester],
    authors="Giorgos Vretinaris",
    sitename="LR_BUG_Sylvester.jl",
    format=Documenter.HTML(;
        canonical="https://gvretina.github.io/LR_BUG_Sylvester.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gvretina/LR_BUG_Sylvester.jl",
    devbranch="main",
)
