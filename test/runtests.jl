

using Test
using EikonalSolvers
using Distributed
using LinearAlgebra

# Run tests

# get all the functions
# include("test_utils.jl")
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

@testset "Gradients adjoint vs. finite differences [2D Cart. coord.]" begin
    test_gradvel_2D()
    test_gradsrc_2D()
end




# @testset "Tests eikonal vs. analytical solutions [Cartesian coodinates]" begin
#     println("Number of workers available: $nwor")
#     for fun in testfun
#         @test fun()
#     end
#     println()
# end

