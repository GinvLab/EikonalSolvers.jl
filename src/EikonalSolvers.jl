
#
# MIT License
# Copyright (c) 2022 Andrea Zunino
# 

##############################################
"""    
EikonalSolvers

EikonalSolvers module, 2D and 3D traveltime and gradient computations in Cartesian 
 and spherical coordinates. 
Computes seismic traveltimes by solving the eikonal equation in two and three dimensions with the possibility of computing the gradient of the misfit function with respect to the velocity model.  
Both forward and gradient computations are parallelised using Julia's distributed computing functions and the parallelisation is "by source", distributing calculations for different sources to different processors.

# Exports

$(EXPORTS)
"""
module EikonalSolvers


export Grid2DCart,Grid2DSphere
export Grid3DCart,Grid3DSphere
export eiktraveltime,eikgradient
export eikttimemisfit
export tracerays

#export eiktraveltime2Dalt,eikgradient2Dalt
#export eiktraveltime3Dalt,eikgradient3Dalt

export EikonalProbVel,EikonalProbSrcLoc

export ExtraParams,GridRefinementPars

# from extensions
export savemodelvtk


# @warn "using TimerOutputs"
# using TimerOutputs
# const tiou = TimerOutput()

using LinearAlgebra
using OffsetArrays
using SparseArrays
using DocStringExtensions
using StaticArrays
using Interpolations


# using LinearAlgebra
## For parallelisation
using Distributed

## Binary heaps
include("BinHeap/BinHeaps.jl")
using .BinHeaps


## general utils
include("EikSolv/eikstructs.jl")
include("EikSolv/eikchecks.jl")
include("EikSolv/eikonalutils_spherical.jl")
include("EikSolv/eikutils.jl")
include("EikSolv/utils.jl")
include("EikSolv/eikmainfwd.jl")
include("EikSolv/eikmaingrad.jl")

## 2D stuff
include("EikSolv/eikforward2D.jl")
include("EikSolv/eikonalforward2D_alternative.jl")
include("EikSolv/eikonalgradient2D_alternative.jl")

## 3D stuff
include("EikSolv/eikforward3D.jl")
include("EikSolv/eikonalforward3D_alternative.jl")
include("EikSolv/eikonalgradient3D_alternative.jl")


## Hamiltonian Monte Carlo setup
include("HMCtraveltimes.jl")
using .HMCtraveltimes


# for extensions
"""
This function requires the package WriteVTK to be loaded first in order to work.
"""
function savemodelvtk end

 
end # module
##############################################
