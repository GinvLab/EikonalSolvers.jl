

using Test
using EikonalSolvers
using Distributed
using LinearAlgebra

# Run tests

# get all the functions
include("test_cartesian.jl")
#include("test_spherical.jl")



nwor = nworkers()
println("Number of workers available: $nwor")

@testset "Eikonal vs. analytical solutions const. vel. [2D Cart. coord.]" begin
    test_fwdtt_2D_constvel()
end

@testset "Eikonal vs. analytical sol., lin. grad. vel. [2D Cart. coord.]" begin
    test_fwdtt_2D_lingrad()
end

@testset "Gradient w.r.t. velocity vs. finite differences [2D Cart. coord.]" begin
    test_gradvel_2D()
end

@testset "Gradient w.r.t. source loc. vs. finite differences [2D Cart. coord.]" begin
    test_gradsrc_2D()
end
