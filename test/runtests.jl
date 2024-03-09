

using Test
using EikonalSolvers
using Distributed
using LinearAlgebra

# Run tests

# get all the functions
# include("test_utils.jl")
include("test_cartesian.jl")
#include("test_spherical.jl")



testfun  = [test_fwdtt_2D_constvel]


nwor = nworkers()

@testset "Eikonal vs. analytical solutions const. vel. [2D Cart. coord.]" begin
    println("Number of workers available: $nwor")
    test_fwdtt_2D_constvel()
end

@testset "Eikonal vs. analytical solutions, lin. grad. vel. [2D Cart. coord.]" begin
    println("Number of workers available: $nwor")
    test_fwdtt_2D_lingrad()
end



# @testset "Tests eikonal vs. analytical solutions [Cartesian coodinates]" begin
#     println("Number of workers available: $nwor")
#     for fun in testfun
#         @test fun()
#     end
#     println()
# end

