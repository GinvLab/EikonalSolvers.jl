using Documenter, EikonalSolvers 

makedocs(modules = [EikonalSolvers],
         repo=Remotes.GitHub("GinvLab","EikonalSolvers.jl"), 
         sitename="EikonalSolvers.jl",
         authors = "Andrea Zunino",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "Home" => "index.md",
             "API" => "publicapi.md",
             "Private stuff" => "privatestuff.md"
         ],
         warnonly = [:missing_docs, :cross_references]
         )

deploydocs(
    repo="github.com/GinvLab/EikonalSolvers.jl.git",   
    devbranch = "main",
    deploy_config = Documenter.GitHubActions(),
    branch = "gl-pages",
)

