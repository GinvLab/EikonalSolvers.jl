

#push!(LOAD_PATH,"../src/")

using Documenter, EikonalSolvers 

#modules = [EikonalSolvers],
makedocs(modules = [EikonalSolvers],
         repo = "../../{path}",
         sitename="EikonalSolvers.jl",
         authors = "Andrea Zunino",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "User guide" => "index.md",
             "API" => "publicapi.md"
         ]
         )

#deploydocs(repo="github.com/inverseproblem/EikonalSolvers.jl.git",target="build")
