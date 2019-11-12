
#
# MIT License
# Copyright (c) 2019 Andrea Zunino
# 

##############################################
"""
     EikonalSolvers module, 2D and 3D traveltime and gradient computations.
"""
module EikonalSolvers


export Grid2D,traveltime2D,gradttime2D
export Grid2DSphere,traveltime2Dsphere,gradttime2Dsphere

export Grid3D,traveltime3D,gradttime3D

export misfitfunc
export EikonalProb

## For parallelisation
using Distributed

## Binary heaps
include("BinHeap/BinHeaps.jl")
using .BinHeaps

## general utils
include("EikSolv/eikonalutils.jl")
include("EikSolv/eikonalutils_spherical.jl")

## 2D stuff
include("EikSolv/eikonalforward2D.jl")
include("EikSolv/eikonalgradient2D.jl")
# spherical/polar coodinates
include("EikSolv/eikonalforward2D_spherical.jl")
include("EikSolv/eikonalgradient2D_spherical.jl")

## 3D stuff
include("EikSolv/eikonalforward3D.jl")
include("EikSolv/eikonalgradient3D.jl")


## Hamiltonian Monte Carlo setup
include("HMCtraveltimes.jl")
using .HMCtraveltimes

end
##############################################
