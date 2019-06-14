

##############################################
"""
     EikonalSolvers module, 2D and 3D traveltime and gradient computations.
"""
module EikonalSolvers


export Grid2D
export traveltime2D
export gradttime2D

export Grid3D
export traveltime3D
export gradttime3D


## For parallelisation
using Distributed

## Binary heaps
include("BinHeap/BinHeaps.jl")
using .BinHeaps


## general utils
include("EikSolv/eikonalutils.jl")

## 2D stuff
include("EikSolv/eikonalforward2D.jl")
include("EikSolv/eikonalgradient2D.jl")

## 3D stuff
include("EikSolv/eikonalforward3D.jl")
include("EikSolv/eikonalgradient3D.jl")


end
##############################################
