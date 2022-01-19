

using Documenter, EikonalSolvers 

makedocs(modules = [EikonalSolvers],
         #repo = "../../{path}",
         sitename="EikonalSolvers.jl",
         authors = "Andrea Zunino",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "Home" => "index.md",
             "API" => "publicapi.md",
             "Private stuff" => "privatestuff.md"
         ]
         )

deploydocs(
    repo="https://gitlab.com/JuliaGeoph/EikonalSolvers.jl",
)
