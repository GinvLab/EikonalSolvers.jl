

using Test
using EikonalSolvers


# Run tests

# get all the functions
include("test_suite.jl")


testname = ["forward traveltime 2D (Cartesian coord.)","forward traveltime 3D (Cartesian coord.)",
            "gradient traveltime 2D (Cartesian coord.)","gradient traveltime 3D (Cartesian coord.)",
            "forward traveltime 2D (spherical coord.)","forward traveltime 3D (spherical coord.)",
            "gradient traveltime 2D (spherical coord.)","gradient traveltime 3D (spherical coord.)" ]

testfun  = [test_fwdtt_2D,          test_fwdtt_3D,
            test_gradtt_2D,         test_gradtt_3D,
            test_fwdtt_2Dsphere,    test_fwdtt_3Dsphere,
            test_gradtt_2Dsphere,   test_gradtt_3Dsphere]

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



