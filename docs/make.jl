using Documenter, EikonalSolvers 

makedocs(modules = [EikonalSolvers],
         repo="https://gitlab.com/JuliaGeoph/EikonalSolvers.jl/blob/{commit}{path}#{line}", 
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
    repo="gitlab.com/JuliaGeoph/EikonalSolvers.jl.git",   #EikonalSolvers.jl/blob/{commit}{path}#{line}",
    #repo="../../EikonalSolvers.jl",
    devbranch = "main",
    deploy_config = Documenter.GitLab(),
    branch = "gl-pages"
)


###########################################################
# GitLab <: DeployConfig

# GitLab implementation of DeployConfig.

# The following environment variables influence the build when using the GitLab configuration:

#     DOCUMENTER_KEY: must contain the Base64-encoded SSH private key for the repository. This variable should be set in the GitLab settings. Make sure this variable is marked NOT to be displayed in the build log.

#     CI_COMMIT_BRANCH: the name of the commit branch.

#     CI_EXTERNAL_PULL_REQUEST_IID: Pull Request ID from GitHub if the pipelines are for external pull requests.

#     CI_PROJECT_PATH_SLUG: The namespace with project name. All letters lowercased and non-alphanumeric characters replaced with -.

#     CI_COMMIT_TAG: The commit tag name. Present only when building tags.

#     CI_PIPELINE_SOURCE: Indicates how the pipeline was triggered.

# The CI_* variables are set automatically on GitLab. More information on how GitLab sets the CI_* variables can be found in the GitLab documentation.
