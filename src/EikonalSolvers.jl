

##############################################
"""
     EikonalSolvers module
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
include("../BinHeap/BinHeaps.jl")
using .BinHeaps


## general utils
include("eikonalutils.jl")

## 2D stuff
include("eikonalforward2D.jl")
include("eikonalgradient2D.jl")

## 3D stuff
include("eikonalforward3D.jl")
include("eikonalgradient3D.jl")


end
##############################################
