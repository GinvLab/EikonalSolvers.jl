

using Test
using EikonalSolvers


# Run tests

# get all the functions
include("test_suite.jl")


testname = ["forward traveltime 2D","forward traveltime 3D",
            "gradient traveltime 2D","gradient traveltime 3D"]
testfun  = [test_fwdtt_2D,          test_fwdtt_3D,
            test_gradtt_2D,          test_gradtt_3D]

nwor = nworkers()

@testset "Tests " begin
    println("\n Number of workers available: $nwor")
    for (tn,fun) in zip(testname,testfun)
        println()
        printstyled("Testing $tn \n", bold=true,color=:cyan)
        @test fun()
    end
    println()
end



