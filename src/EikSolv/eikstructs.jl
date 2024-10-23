
######################################################

abstract type AbstractGridEik end
abstract type AbstractGridEik2D <: AbstractGridEik end
abstract type AbstractGridEik3D <: AbstractGridEik end


abstract type AbstractFMMVars end
abstract type AbstractAdjointVars end

abstract type AbstractSrcRefinVars end

######################################################
"""
$(TYPEDEF)

A structure holding the 2D grid parameters, geometry and size.

$(TYPEDFIELDS)


The constructor requires three arguments (see the example):
- `hgrid`: spacing of the grid nodes
- `cooinit`: origin of the coordinate system
- `grsize`: number of grid nodes along x and y

# Example
```julia-repl
julia> Grid2DCart(hgrid=5.0,cooinit=(0.0,0.0),grsize=(300,250))
```
"""
struct Grid2DCart <: AbstractGridEik2D
    "Spacing of the grid nodes"
    hgrid::Float64
    "Origin of the coodinates of the grid"
    cooinit::NTuple{2,Float64}
    "End of the coodinates of the grid"
    cooend::NTuple{2,Float64}
    "Number of grid nodes along x and y"
    grsize::NTuple{2,Int64}
    "x coordinates of the grid nodes"
    x::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}
    "y coordinates of the grid nodes"
    y::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}

    function Grid2DCart(; hgrid::Float64,cooinit::NTuple{2,Float64},
                        grsize::NTuple{2,Int64})
        @assert hgrid>0.0
        @assert all(grsize.>0)
        x = range(start=cooinit[1],step=hgrid,length=grsize[1]) 
        y = range(start=cooinit[2],step=hgrid,length=grsize[2]) 
        new(hgrid,cooinit,(x[end],y[end]),grsize,x,y)
    end
end

######################################################

"""
$(TYPEDEF)

A structure holding the 3D grid parameters, geometry and size.

# Fields 

$(TYPEDFIELDS)

The constructor requires three arguments (see the example):
- `hgrid`: spacing of the grid nodes
- `cooinit`: origin of the coordinate system
- `grsize`: number of grid nodes along x, y and z

# Example
```julia-repl
julia> Grid3DCart(hgrid=5.0,cooinit=(0.0,0.0,0.0),grsize=(60,60,40))
```
"""
struct Grid3DCart <: AbstractGridEik3D
    "Spacing of the grid nodes"
    hgrid::Float64
    "Origin of the coodinates of the grid"
    cooinit::NTuple{3,Float64}
    "End of the coodinates of the grid"
    cooend::NTuple{3,Float64}
    "Number of grid nodes along x, y and z"
    grsize::NTuple{3,Int64}
    "x coordinates of the grid nodes"
    x::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 
    "y coordinates of the grid nodes"
    y::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 
    "z coordinates of the grid nodes"
    z::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 
    
    function Grid3DCart(; hgrid::Float64,cooinit::NTuple{3,Float64},
                        grsize::NTuple{3,Int64})
        @assert hgrid>0.0
        @assert all(grsize.>0)
        x = range(start=cooinit[1],step=hgrid,length=grsize[1]) 
        y = range(start=cooinit[2],step=hgrid,length=grsize[2]) 
        z = range(start=cooinit[3],step=hgrid,length=grsize[3]) 
        new(hgrid,cooinit,(x[end],y[end],z[end]),grsize,x,y,z)
    end
end


######################################################
"""
$(TYPEDSIGNATURES)

A structure holding the 2D SPHERICAL grid parameters, geometry and size.

$(TYPEDFIELDS)

The fields are:    
- `Δr`: spacing of the grid nodes along the radial coordinate r
- `Δθ`: spacing of the grid nodes along the θ coordinate
- `cooinit`: origin of the coordinate system
- `grsize`: number of grid nodes along r and θ 


# Example
```julia-repl
julia> Grid2DSphere(Δr=15.0,Δθ=2.0,grsize=(10,15),cooinit=(500.0,0.0))
```
"""
struct Grid2DSphere <: AbstractGridEik2D
    "Spacing of the grid nodes along the radius (r)"
    Δr::Float64
    "Spacing of the grid nodes along the angle θ"
    Δθ::Float64
    "Origin of the coodinates of the grid"
    cooinit::NTuple{2,Float64}
    "End of the coodinates of the grid"
    cooend::NTuple{2,Float64}
    "Number of grid nodes along r and θ"
    grsize::NTuple{2,Int64}
    "r coordinates of the grid nodes"
    r::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 
    "θ coordinates of the grid nodes"
    θ::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 

    function Grid2DSphere(; Δr::Float64,Δθ::Float64,cooinit::NTuple{2,Float64},
                          grsize::NTuple{2,Int64} )
        @assert cooinit[1]>=0.0
        @assert Δr>0.0
        @assert Δθ>0.0
        @assert all(grsize.>0)
        r = range(start=cooinit[1],step=Δr,length=grsize[1]) 
        θ = range(start=cooinit[2],step=Δθ,length=grsize[2]) 
        ## limit to 180 degrees for now...
        @assert all(0.0.<=θ.<=180.0)
        # ## now convert everything to radians
        # println("Grid2DSphere(): converting θ to radians")
        # θ .= deg2rad(θ)
        # Δθ = deg2rad(Δθ)
        new(Δr,Δθ,cooinit,(r[end],θ[end]),grsize,r,θ)
    end
end


######################################################

"""
$(TYPEDSIGNATURES)

A structure holding the 3D SPHERICAL grid parameters, geometry and size.

$(TYPEDFIELDS)

The fields are:    
- `Δr`: spacing of the grid nodes along the radial coordinate r
- `Δθ`: spacing of the grid nodes along the θ coordinate
- `Δφ`: spacing of the grid nodes along the θ coordinate
- `cooinit`: origin of the coordinate system
- `grsize`: number of grid nodes along r, θ and φ 

# Example
```julia-repl
julia> Grid3DSphere(Δr=15.0,Δθ=2.0,Δφ=1.5,grsize=(10,15,12),cooinit=(500.0,20.0,0.0))
```
"""
struct Grid3DSphere <: AbstractGridEik3D
    "Spacing of the grid nodes along the radius (r)"
    Δr::Float64
    "Spacing of the grid nodes along the angle θ"
    Δθ::Float64
    "Spacing of the grid nodes along the angle φ"
    Δφ::Float64
    "Origin of the coodinates of the grid"
    cooinit::NTuple{3,Float64}
    "End of the coodinates of the grid"
    cooend::NTuple{3,Float64}
    "Number of grid nodes along r, θ and φ"
    grsize::NTuple{3,Int64}
    "r coordinates of the grid nodes"
    r::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 
    "θ coordinates of the grid nodes"
    θ::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 
    "φ coordinates of the grid nodes"
    φ::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} 

    function Grid3DSphere(; Δr::Float64,Δθ::Float64,Δφ::Float64,cooinit::NTuple{3,Float64},
                          grsize::NTuple{3,Int64})
        @assert cooinit[1]>0.0
        @assert all(grsize.>0)
        r = range(start=cooinit[1],step=Δr,length=grsize[1]) #[rinit+Δr*(i-1) for i =1:nr]
        θ = range(start=cooinit[2],step=Δθ,length=grsize[2]) #[θinit+Δθ*(i-1) for i =1:nθ]
        φ = range(start=cooinit[3],step=Δφ,length=grsize[3]) #[φinit+Δφ*(i-1) for i =1:nφ]
        ## exclude the poles... sin(θ) -> sin(0) or sin(180) = 0  -> leads to division by zero
        @assert all(5.0.<=θ.<=175.0)
        ## limit to 180 degrees for now...
        @assert all(0.0.<=φ.<=180.0)
        # ## now convert everything to radians
        # println("Grid2DSphere(): converting θ to radians")
        # θ .= deg2rad(θ)
        # Δθ = deg2rad(Δθ)
        new(Δr,Δθ,Δφ,cooinit,(r[end],θ[end],φ[end]),grsize,r,θ,φ)
    end
end

###############################################################

"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
Base.@kwdef struct GridRefinementPars
    "Downscaling factor for refined grid around the source"
    downscalefactor::Int64
    "Number of (coarse) grid nodes (radius) within which to refine the grid"
    noderadius::Int64
end

##################################################################

"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
struct ExtraParams
    "refine grid around source?"
    refinearoundsrc::Bool
    "trigger GC manually at selected points"
    manualGCtrigger::Bool
    "Serial, Threads or Distributed run? Either :serial, :sharedmem or :distribmem"
    parallelkind::Symbol
    "Radius for smoothing the gradient around the source. Zero means no smoothing."
    radiussmoothgradsrc::Int64
    "Smooth the gradient with a kernel of size (in pixels). Zero means no smoothing."
    smoothgradkern::Int64
    "Downscaling factor and node radius for refined grid creation"
    grdrefpars::GridRefinementPars

    function ExtraParams(; refinearoundsrc::Bool=true,
                         manualGCtrigger::Bool=false,
                         parallelkind::Symbol=:serial,
                         radiussmoothgradsrc::Integer=3,
                         smoothgradkern::Integer=0,
                         grdrefpars::GridRefinementPars=GridRefinementPars(downscalefactor=5,noderadius=3) )

        @assert radiussmoothgradsrc>= 0
        @assert smoothgradkern>=0

        if smoothgradkern>0 && iseven(smoothgradkern)
            @show smoothgradkern
            error("ExtraParams: smoothgradkern must be an odd number")
        end
        
        if !(parallelkind in [:serial,:sharedmem,:distribmem])
            error("ExtraParams(): 'parallelkind' must be one of :serial, :sharedmem or :distribmem")
        end
        return new(refinearoundsrc,manualGCtrigger,
                   parallelkind,radiussmoothgradsrc,smoothgradkern,
                   grdrefpars)
    end    
end

###############################################################

struct MapOrderGridFMM{N} 
    "Linear grid indices in FMM order (as visited by FMM)"
    lfmm2grid::Vector{Int64} # idx_fmmord
    "Linear FMM indices in grid order"
    lgrid2fmm::Vector{Int64} # idx_gridord
    grsize::NTuple{N,Int64}

    function MapOrderGridFMM(nxnynz::Tuple)
        nxyz = prod(nxnynz)
        lfmm2grid = zeros(Int64,nxyz)
        lgrid2fmm = zeros(Int64,nxyz)
        Ndim = length(nxnynz)
        return new{Ndim}(lfmm2grid,lgrid2fmm,nxnynz)
    end
end

########################################################

# structs for sparse matrices (CSR) to represent derivatives (discrete adjoint)
struct VecSPDerivMat
    "row pointer"
    iptr::Vector{Int64}
    "column index"
    j::Vector{Int64}
    "values"
    v::Vector{Float64}
    lastrowupdated::Base.RefValue{Int64}
    Nsize::MVector{2,Int64} ## will change, no tuples
    Nnnz::Base.RefValue{Int64}

    function VecSPDerivMat(; iptr,j,v,Nsize)
        lastrowupdated = Ref(0)
        # first row pointer must always be 1
        iptr[1] = 1
        Nnnz = Ref(0)
        @assert length(iptr)==Nsize[1]+1
        new(iptr,j,v,lastrowupdated,Nsize,Nnnz) 
    end
end

########################################################

struct VarsFMMOrder
    "Traveltime"
    ttime::Vector{Float64}
    "Last computed traveltime"
    lastcomputedtt::Base.RefValue{Int64}
    "Derivative (along x, y, z) matrix, row-deficient"
    Deriv::Vector{VecSPDerivMat}
    "keep track of source points in term of row index (ordered according to FMM)"
    onsrccols::Vector{Bool}
    "keep track of points marked as source points for the coarse grid while running the refinement of the grid around the source (ordered according to FMM)"
    onhpoints::Vector{Bool}
    
    function VarsFMMOrder(nijk::Tuple) 
        npts = prod(nijk)
        ndim = length(nijk)
        ttime = -1.0 .* ones(npts)
        ## nxy*3 because of the stencils in the second-order fast marching (max 3 points)
        ## Nsize=[npts,npts] will be changed while the algo is running...
        Deriv = Vector{VecSPDerivMat}(undef,ndim)
        for d=1:ndim
            Deriv[d] = VecSPDerivMat( iptr=zeros(Int64,npts+1), j=zeros(Int64,npts*3),
                                      v=zeros(npts*3), Nsize=[npts,npts] )
        end
        # vecDy = VecSPDerivMat( iptr=zeros(Int64,nxy+1), j=zeros(Int64,nxy*3),
        #                        v=zeros(nxy*3), Nsize=[nxy,nxy] )
        onsrccols = zeros(Bool,npts) # all false
        onhpoints = zeros(Bool,npts) # all false
        lastcomputedtt = Ref(0)
        return new(ttime,lastcomputedtt,Deriv,onsrccols,onhpoints)
    end
end

###################################################

abstract type CoeffDerivatives end

struct CoeffDerivCartesian <: CoeffDerivatives
    firstord::MVector{2,Float64}
    secondord::MVector{3,Float64}
end

struct CoeffDerivSpherical2D <: CoeffDerivatives
    firstord::AbstractArray #Union{MVector{2,Float64},MMatrix{Float64}}
    secondord::AbstractArray #Union{MVector{3,Float64},MMatrix{Float64}}
end

struct CoeffDerivSpherical3D <: CoeffDerivatives
    firstord::Matrix{Float64}
    secondord::Matrix{Float64}
end

####################################

struct AdjointVars2D <: AbstractAdjointVars
    idxconv::MapOrderGridFMM#2D
    fmmord::VarsFMMOrder
    codeDeriv::Array{Int64,2}

    function AdjointVars2D(n1,n2)
        idxconv = MapOrderGridFMM((n1,n2))
        fmmord = VarsFMMOrder((n1,n2))
        codeDeriv = zeros(Int64,n1*n2,2)
        new(idxconv,fmmord,codeDeriv)
    end
end


struct AdjointVars3D <: AbstractAdjointVars
    idxconv::MapOrderGridFMM#3D
    fmmord::VarsFMMOrder
    codeDeriv::Array{Int64,2}

    function AdjointVars3D(n1,n2,n3)
        idxconv = MapOrderGridFMM((n1,n2,n3))
        fmmord = VarsFMMOrder((n1,n2,n3))
        codeDeriv = zeros(Int64,n1*n2*n3,3)
        new(idxconv,fmmord,codeDeriv)
    end
end


####################################

mutable struct SourceBoxParams
    "(i,j) positions of the corners surrounding the source location"
    ijksrc::MArray
    "(x,y) position of the source"
    xyzsrc::MVector
    "coefficient matrix for the interpolation of the velocity"
    velcorn::MVector
    "distance from the corners to the source location"
    distcorn::MVector
end

####################################

mutable struct SourcePtsFromFineGrid
    ijksrc::Matrix{Int64} 
end

######################################################

struct FMMVars2D <: AbstractFMMVars
    ttime::Array{Float64,2}
    status::Array{UInt8,2}
    bheap::BinHeapMin
    "refine grid around source"
    refinearoundsrc::Bool
    srcboxpar::Union{SourceBoxParams,SourcePtsFromFineGrid}

    function FMMVars2D(n1::Integer,n2::Integer;
                       amIcoarsegrid::Bool,refinearoundsrc::Bool)
        ttime = zeros(Float64,n1,n2)
        status = zeros(UInt8,n1,n2)
        bheap = init_minheap(n1*n2)
        
        if  amIcoarsegrid && refinearoundsrc==true
            ## there is refinement
            srcboxpar = SourcePtsFromFineGrid(Array{Int64,2}(undef,0,2))
        else
            ## there is NO refinement
            Ncoe = 4
            srcboxpar = SourceBoxParams(MMatrix{Ncoe,2}(zeros(Int64,Ncoe,2)),
                                        MVector{2}(zeros(2)),
                                        MVector{Ncoe}(zeros(Ncoe)),
                                        MVector{Ncoe}(zeros(Ncoe)) )

        end
        new(ttime,status,bheap,refinearoundsrc,srcboxpar)
    end
end

######################################################

struct FMMVars3D <: AbstractFMMVars
    ttime::Array{Float64,3}
    status::Array{UInt8,3}
    bheap::BinHeapMin
    "refine grid around source"
    refinearoundsrc::Bool
    srcboxpar::Union{SourceBoxParams,SourcePtsFromFineGrid}
    
    function FMMVars3D(n1::Integer,n2::Integer,n3::Integer;
                       amIcoarsegrid::Bool,refinearoundsrc::Bool)
        ttime = zeros(Float64,n1,n2,n3)
        status = zeros(UInt8,n1,n2,n3)
        bheap = init_minheap(n1*n2*n3)

        if amIcoarsegrid && refinearoundsrc==true
            ## there is refinement
            srcboxpar = SourcePtsFromFineGrid(Array{Int64,2}(undef,0,3))
        else
            Ncoe = 8
            srcboxpar = SourceBoxParams(MMatrix{Ncoe,3}(zeros(Int64,Ncoe,3)),
                                        MVector{3}(zeros(3)),
                                        MVector{Ncoe}(zeros(Ncoe)),
                                        MVector{Ncoe}(zeros(Ncoe)) )                                          
        end
        new(ttime,status,bheap,refinearoundsrc,srcboxpar)
    end
end

###################################################

struct SrcRefinVars{M,N} <: AbstractSrcRefinVars
    downscalefactor::Int64
    ijkorigincoarse::NTuple{M,Int64}
    outijk_min::NTuple{N,Bool}
    outijk_max::NTuple{N,Bool}
    nearneigh_oper::SparseMatrixCSC{Float64,Int64}
    nearneigh_idxcoarse::Vector{Int64}
    velcart_fine::Array{Float64,N}
end


# struct SrcRefinVars3D <: AbstractSrcRefinVars
#     downscalefactor::Int64
#     ijkcoarse::NTuple{6,Int64}
#     #nxnynz_window_coarse::NTuple{3,Int64}
#     #outxyminmax::NTuple{6,Bool}
#     nearneigh_oper::SparseMatrixCSC{Float64,Int64}
#     nearneigh_idxcoarse::Vector{Int64}
#     velcart_fine::Array{Float64,3}
# end

###################################################
