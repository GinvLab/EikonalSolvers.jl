
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


export Grid2D,Grid2DSphere
export traveltime2D,gradttime2D
export traveltime2Dalt,gradttime2Dalt

export Grid3D,Grid3DSphere
export traveltime3D,gradttime3D
export traveltime3Dalt,gradttime3Dalt

export ttmisfitfunc
export EikonalProb

export ExtraParams


# println("Hello -> using TimerOutputs")
# using TimerOutputs


using LinearAlgebra
using OffsetArrays
using SparseArrays
using DocStringExtensions
using StaticArrays

# using LinearAlgebra
## For parallelisation
using Distributed

## Binary heaps
include("BinHeap/BinHeaps.jl")
using .BinHeaps


## general utils
include("EikSolv/eikstructs.jl")
include("EikSolv/eikonalutils_spherical.jl")
include("EikSolv/eikonalutils.jl")
include("EikSolv/utils.jl")

## 2D stuff
include("EikSolv/eikonalforwardND.jl")
include("EikSolv/eikonalgradientND.jl")
include("EikSolv/eikonalforward2D_alternative.jl")
include("EikSolv/eikonalgradient2D_alternative.jl")

## 3D stuff
include("EikSolv/eikonalforward3D.jl")
include("EikSolv/eikonalgradient3D.jl")
include("EikSolv/eikonalforward3D_alternative.jl")
include("EikSolv/eikonalgradient3D_alternative.jl")


## Hamiltonian Monte Carlo setup
include("HMCtraveltimes.jl")
using .HMCtraveltimes


# const extrapars = ExtraParams(allowfixsqarg=false,
#                               refinearoundsrc=true,
#                               manualGCtrigger=true)
# warningextrapar(extrapars)


end
##############################################
