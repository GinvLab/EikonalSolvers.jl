

using Test
using EikonalSolvers


# Run tests

# get all the functions
include("test_suite.jl")


testname = ["forward traveltime 2D FMM 2nd order (Cartesian coord.)", "forward traveltime 2D alternative (Cartesian coord.)",
            "forward traveltime 3D FMM 2nd order (Cartesian coord.)",  "forward traveltime 3D FMM alternative (Cartesian coord.)",
            "gradient traveltime 2D FMM 2nd order (Cartesian coord.)", "gradient traveltime 2D alternative (Cartesian coord.)",
            "gradient traveltime 3D FMM 2nd order (Cartesian coord.)", "gradient traveltime 3D alternative (Cartesian coord.)",
            "forward traveltime 2D FMM 2nd order  (spherical coord.)",
            "gradient traveltime 2D FMM 2nd order (spherical coord.)", "gradient traveltime 2D alternative (spherical coord.)",
            "gradient traveltime 3D FMM 2nd order (spherical coord.)", "gradient traveltime 3D alternative (spherical coord.)" ]

testfun  = [test_fwdtt_2D_FMM2ndord, test_fwdtt_2D_alternative,
            test_fwdtt_3D_FMM2ndord, test_fwdtt_3D_alternative,
            test_gradtt_2D_FMM2ndord, test_gradtt_2D_alternative,
            test_gradtt_3D_FMM2ndord, test_gradtt_3D_alternative,
            test_fwdtt_2Dspher_FMM2ndord,
            test_gradtt_2Dspher_FMM2ndord, test_gradtt_2Dspher_alternative,
            test_fwdtt_3Dspher_FMM2ndord,
            test_gradtt_3Dspher_FMM2ndord, test_gradtt_3Dspher_alternative ]
            
           

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



