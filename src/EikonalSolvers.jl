
#
# MIT License
# Copyright (c) 2019 Andrea Zunino
# 

##############################################
"""    
EikonalSolvers module, 2D and 3D traveltime and gradient computations in Cartesian 
 and spherical coordinates. 
Computes seismic traveltimes by solving the eikonal equation in two and three dimensions with the possibility of computing the gradient of the misfit function with respect to the velocity model.  
Both forward and gradient computations are parallelised using Julia's distributed computing functions and the parallelisation is "by source", distributing calculations for different sources to different processors
"""
module EikonalSolvers


export Grid2D,traveltime2D,gradttime2D
export Grid2DSphere,traveltime2Dsphere,gradttime2Dsphere

export Grid3D,traveltime3D,gradttime3D
export Grid3DSphere,traveltime3Dsphere,gradttime3Dsphere

export ttmisfitfunc
export EikonalProb

# using LinearAlgebra
## For parallelisation
using Distributed

## Binary heaps
include("BinHeap/BinHeaps.jl")
using .BinHeaps

## general utils
include("EikSolv/eikonalutils_spherical.jl")
include("EikSolv/eikonalutils.jl")

## 2D stuff
include("EikSolv/eikonalforward2D.jl")
include("EikSolv/eikonalgradient2D.jl")
# spherical/polar coodinates
include("EikSolv/eikonalforward2D_spherical.jl")
include("EikSolv/eikonalgradient2D_spherical.jl")

## 3D stuff
include("EikSolv/eikonalforward3D.jl")
include("EikSolv/eikonalgradient3D.jl")
# spherical/polar coodinates
include("EikSolv/eikonalforward3D_spherical.jl")
include("EikSolv/eikonalgradient3D_spherical.jl")

## Hamiltonian Monte Carlo setup
include("HMCtraveltimes.jl")
using .HMCtraveltimes


##--------------------------------------------------------------
## allowfixsqarg: if true, try to brute-force fix problems with
##                negative sqarg.
##    Use is discouraged...
Base.@kwdef mutable struct ExtraParams
    ## brute-force fix negative sqarg
    allowfixsqarg::Bool
    ## refine grid around source?
    refinearoundsrc::Bool
end

extrapars = ExtraParams(allowfixsqarg=false,
                        refinearoundsrc=true)

if extrapars.allowfixsqarg==true
    @warn("ExtraParams: allowfixsqarg==true, brute-force fixing of negative discriminant allowed.")
end
if extrapars.refinearoundsrc==false
    @warn("ExtraParams: refinearoundsrc==false, no grid refinement around source.")
end
##--------------------------------------------------------------



end
##############################################
