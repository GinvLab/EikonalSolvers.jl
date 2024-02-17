
######################################################

abstract type GridEik end
abstract type GridEik2D <: GridEik end
abstract type GridEik3D <: GridEik end

######################################################
"""
$(TYPEDEF)

A structure holding the 2D grid parameters, geometry and size.

The fields are:    
- `hgrid`: spacing of the grid nodes (same for x and y)
- `xinit, yinit`: origin of the coordinates of the grid
- `nx, ny`: number of nodes along x and y for the velocity array
- `ntx, nty`: number of nodes along x and y for the time array when using a staggered grid, meaningful and used *only* for Podvin and Lecomte stencils


# Example
```julia-repl
julia> Grid2D(hgrid=5.0,xinit=0.0,yinit=0.0,nx=300,ny=250)
```
"""
struct Grid2D <: GridEik2D
    hgrid::Float64
    xinit::Float64
    yinit::Float64
    nx::Int64
    ny::Int64
    ntx::Int64
    nty::Int64
    x::Vector{Float64}
    y::Vector{Float64}

    function Grid2D(; hgrid::Float64,xinit::Float64,yinit::Float64,nx::Int64,ny::Int64)
        ntx::Int64 = nx+1
        nty::Int64 = ny+1
        x = [xinit+(i-1)*hgrid for i=1:nx]
        y = [yinit+(i-1)*hgrid for i=1:ny]
        new(hgrid,xinit,yinit,nx,ny,ntx,nty,x,y)
    end
end

######################################################

"""
$(TYPEDEF)

A structure holding the 3D grid parameters, geometry and size.

# Fields 

$(TYPEDFIELDS)

The fields are:    
- `hgrid`: spacing of the grid nodes (same for x, y and z)
- `xinit, yinit, zinit`: origin of the coordinates of the grid
- `nx, ny, nz`: number of nodes along x and y for the velocity array
- `ntx, nty, ntz`: number of nodes along x and y for the time array when using a staggered grid, meaningful and used *only* for Podvin and Lecomte stencils


# Example
```julia-repl
julia> Grid3D(hgrid=5.0,xinit=0.0,yinit=0.0,zinit=0.0,nx=60,ny=60,nz=40)
```
"""
struct Grid3D <: GridEik3D
    hgrid::Float64
    xinit::Float64
    yinit::Float64
    zinit::Float64
    nx::Int64
    ny::Int64
    nz::Int64
    ntx::Int64
    nty::Int64
    ntz::Int64
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    
    ## constructor function
    function Grid3D(; hgrid::Float64,xinit::Float64,yinit::Float64, zinit::Float64,
                    nx::Int64,ny::Int64,nz::Int64)
        ntx::Int64 = nx+1
        nty::Int64 = ny+1
        ntz::Int64 = nz+1
        x = [xinit+(i-1)*hgrid for i=1:nx]
        y = [yinit+(i-1)*hgrid for i=1:ny]
        z = [zinit+(i-1)*hgrid for i=1:nz]
        new(hgrid,xinit,yinit,zinit,nx,ny,nz,ntx,nty,ntz,x,y,z)
    end
end


######################################################
"""
$(TYPEDSIGNATURES)

A structure holding the 2D SPHERICAL grid parameters, geometry and size.

The fields are:    
- `Δr`: spacing of the grid nodes along the radial coordinate r
- `Δθ`: spacing of the grid nodes along the θ coordinate
- `rinit, θinit`: origin of the coordinates of the grid
- `nr, nθ`: number of nodes along r and θ for the velocity array (structured grid)


# Example
```julia-repl
julia> Grid2DSphere(Δr=15.0,Δθ=2.0,nr=10,nθ=15,rinit=500.0,θinit=0.0)
```
"""
struct Grid2DSphere <: GridEik2D
    Δr::Float64
    Δθ::Float64
    rinit::Float64
    θinit::Float64
    nr::Int64
    nθ::Int64
    r::Vector{Float64}
    θ::Vector{Float64}

    function Grid2DSphere(; Δr::Float64,Δθ::Float64,rinit::Float64,θinit::Float64,nr::Int64,nθ::Int64)
        r = [rinit+Δr*(i-1) for i =1:nr]
        θ = [θinit+Δθ*(i-1) for i =1:nθ]
        ## limit to 180 degrees for now...
        @assert all(0.0.<=θ.<=180.0)
        # ## now convert everything to radians
        # println("Grid2DSphere(): converting θ to radians")
        # θ .= deg2rad(θ)
        # Δθ = deg2rad(Δθ)
        new(Δr,Δθ,rinit,θinit,nr,nθ,r,θ)
    end
end


######################################################

"""
$(TYPEDSIGNATURES)

A structure holding the 3D SPHERICAL grid parameters, geometry and size.

The fields are:    
- `Δr`: spacing of the grid nodes along the radial coordinate r
- `Δθ`: spacing of the grid nodes along the θ coordinate
- `Δφ`: spacing of the grid nodes along the θ coordinate
- `rinit, θinit,φinit`: origin of the coordinates of the grid
- `nr, nθ, nφ`: number of nodes along r, θ and φ for the velocity array (structured grid)

# Example
```julia-repl
    julia> Grid3DSphere(Δr=15.0,Δθ=2.0,Δφ=1.5,nr=10,nθ=15,nφ=12,rinit=500.0,θinit=20.0,φinit=0.0)
```
"""
struct Grid3DSphere <: GridEik3D
    Δr::Float64
    Δθ::Float64
    Δφ::Float64
    rinit::Float64
    θinit::Float64
    φinit::Float64
    nr::Int64
    nθ::Int64
    nφ::Int64
    r::Vector{Float64}
    θ::Vector{Float64}
    φ::Vector{Float64}

    function Grid3DSphere(; Δr::Float64,Δθ::Float64,Δφ::Float64,rinit::Float64,θinit::Float64,φinit::Float64,
                          nr::Int64,nθ::Int64,nφ::Int64 )
        r = [rinit+Δr*(i-1) for i =1:nr]
        θ = [θinit+Δθ*(i-1) for i =1:nθ]
        φ = [φinit+Δφ*(i-1) for i =1:nφ]

        ## exclude the poles... sin(θ) -> sin(0) or sin(180) = 0  -> leads to division by zero
        @assert all(5.0.<=θ.<=175.0)
        ## limit to 180 degrees for now...
        @assert all(0.0.<=φ.<=180.0)
        # ## now convert everything to radians
        # println("Grid2DSphere(): converting θ to radians")
        # θ .= deg2rad(θ)
        # Δθ = deg2rad(Δθ)
        new(Δr,Δθ,Δφ,rinit,θinit,φinit,nr,nθ,nφ,r,θ,φ)
    end
end

###############################################################

Base.@kwdef struct GridRefinementPars
    downscalefactor::Int64
    noderadius::Int64
end

##################################################################

"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
struct ExtraParams
    "brute-force fix negative sqarg"
    allowfixsqarg::Bool
    "refine grid around source?"
    refinearoundsrc::Bool
    "trigger GC manually at selected points"
    manualGCtrigger::Bool
    "Serial, Threads or Distributed run? Either :serial, :sharedmem or :distribmem"
    parallelkind::Symbol
    "Radius for smoothing the gradient around the source. Zero means no smoothing."
    radiussmoothgradsrc::UInt64
    "Smooth the gradient with a kernel of size (in pixels). Zero means no smoothing."
    smoothgradkern::UInt64
    "Downscaling factor and node radius for refined grid creation"
    grdrefpars::GridRefinementPars

    function ExtraParams(; allowfixsqarg::Bool=false,
                         refinearoundsrc::Bool=true,
                         manualGCtrigger::Bool=false,
                         parallelkind::Symbol=:sharedmem,
                         radiussmoothgradsrc::Integer=3,
                         smoothgradkern::Integer=0,
                         grdrefpars::GridRefinementPars=GridRefinementPars(downscalefactor=3,noderadius=3) )
        
        if !(parallelkind in [:serial,:sharedmem,:distribmem])
            error("ExtraParams(): 'parallelkind' must be one of :serial, :sharedmem or :distribmem")
        end
        return new(allowfixsqarg,refinearoundsrc,manualGCtrigger,
                   parallelkind,radiussmoothgradsrc,smoothgradkern,
                   grdrefpars)
    end    
end

###############################################################

abstract type MapOrderGridFMM end

struct MapOrderGridFMM2D <: MapOrderGridFMM
    "Linear grid indices in FMM order (as visited by FMM)"
    lfmm2grid::Vector{Int64} # idx_fmmord
    "Linear FMM indices in grid order"
    lgrid2fmm::Vector{Int64} # idx_gridord
    nx::Int64
    ny::Int64

    function MapOrderGridFMM2D(nx,ny)
        nxy = nx*ny
        lfmm2grid = zeros(Int64,nxy)
        lgrid2fmm = zeros(Int64,nxy)
        return new(lfmm2grid,lgrid2fmm,nx,ny)
    end
end


struct MapOrderGridFMM3D <: MapOrderGridFMM
    "Linear grid indices in FMM order (as visited by FMM)"
    lfmm2grid::Vector{Int64} # idx_fmmord
    "Linear FMM indices in grid order"
    lgrid2fmm::Vector{Int64} # idx_gridord
    nx::Int64
    ny::Int64
    nz::Int64

    function MapOrderGridFMM3D(nx,ny,nz)
        nxyz = nx*ny*nz
        lfmm2grid = zeros(Int64,nxyz)
        lgrid2fmm = zeros(Int64,nxyz)
        return new(lfmm2grid,lgrid2fmm,nx,ny,nz)
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

struct VarsFMMOrder2D
    "Traveltime"
    ttime::Vector{Float64}
    "Last computed traveltime"
    lastcomputedtt::Base.RefValue{Int64}
    "Derivative (along x) matrix, row-deficient"
    vecDx::VecSPDerivMat
    "Derivative (along y) matrix, row-deficient"
    vecDy::VecSPDerivMat
    "keep track of source points in term of row index (ordered according to FMM)"
    onsrccols::Vector{Bool}
    "keep track of points marked as source points for the coarse grid while running the refinement of the grid around the source (ordered according to FMM)"
    onhpoints::Vector{Bool}
    
    function VarsFMMOrder2D(nx,ny)
        nxy =  nx*ny
        ttime = zeros(nxy)
        ## nxy*3 because of the stencils in the second-order fast marching (max 3 points)
        ## Nsize=[nxy,nxy] will be changed while the algo is running...
        vecDx = VecSPDerivMat( iptr=zeros(Int64,nxy+1), j=zeros(Int64,nxy*3),
                               v=zeros(nxy*3), Nsize=[nxy,nxy] )
        vecDy = VecSPDerivMat( iptr=zeros(Int64,nxy+1), j=zeros(Int64,nxy*3),
                               v=zeros(nxy*3), Nsize=[nxy,nxy] )
        onsrccols = zeros(Bool,nxy) # all false
        onhpoints = zeros(Bool,nxy) # all false
        lastcomputedtt = Ref(0)
        return new(ttime,lastcomputedtt,vecDx,vecDy,onsrccols,onhpoints)
    end
end

###################################################

struct VarsFMMOrder3D
    ttime::Vector{Float64}
     "Last computed traveltime"
    lastcomputedtt::Base.RefValue{Int64}
    vecDx::VecSPDerivMat
    vecDy::VecSPDerivMat
    vecDz::VecSPDerivMat

    function VarsFMMOrder3D(nx,ny,nz)
        nxyz =  nx*ny*nz
        ttime = zeros(nxyz)
        vecDx = VecSPDerivMat( iptr=zeros(Int64,nxyz+1), j=zeros(Int64,nxyz*3),
                               v=zeros(nxyz*3), Nsize=[nxyz,nxyz] )
        vecDy = VecSPDerivMat( iptr=zeros(Int64,nxyz+1), j=zeros(Int64,nxyz*3),
                               v=zeros(nxyz*3), Nsize=[nxyz,nxyz] )
        vecDz = VecSPDerivMat( iptr=zeros(Int64,nxyz+1), j=zeros(Int64,nxyz*3),
                               v=zeros(nxyz*3), Nsize=[nxyz,nxyz] )
        lastcomputedtt = Ref(0)
        return new(ttime,vecDx,vecDy,vecDz)
    end
end

###################################################

abstract type CoeffDerivatives end

struct CoeffDerivCartesian <: CoeffDerivatives
    firstord::MVector{2,Float64}
    secondord::MVector{3,Float64}
end

struct CoeffDerivSpherical2D <: CoeffDerivatives
    firstord::Matrix{Float64}
    secondord::Matrix{Float64}
end

struct CoeffDerivSpherical3D <: CoeffDerivatives
    firstord::Matrix{Float64}
    secondord::Matrix{Float64}
end

####################################

struct AdjointVars2D
    idxconv::MapOrderGridFMM2D
    fmmord::VarsFMMOrder2D
    codeDxy::Array{Int64,2}

    function AdjointVars2D(n1,n2)
        idxconv = MapOrderGridFMM2D(n1,n2)
        fmmord = VarsFMMOrder2D(n1,n2)
        codeDxy = zeros(Int64,n1*n2,2)
        new(idxconv,fmmord,codeDxy)
    end
end


struct AdjointVars3D
    idxconv::MapOrderGridFMM3D
    fmmord::VarsFMMOrder3D
    codeDxyz::Array{Int64,2}

    function AdjointVars3D(n1,n2,n3)
        idxconv = MapOrderGridFMM3D(n1,n2,n3)
        fmmord = VarsFMMOrder3D(n1,n2,n3)
        codeDxyz = zeros(Int64,n1*n2*n3,3)
        new(idxconv,fmmord,codeDxyz)
    end
end


####################################

struct SourceBoxParams2D
    "(i,j) positions of the corners surrounding the source location"
    ijsrc::MArray
    "(x,y) position of the source"
    xysrc::MVector
    "coefficient matrix for the interpolation of the velocity"
    velcorn::MVector
    "distance from the corners to the source location"
    distcorn::MVector
end

####################################

struct SourceBoxParams3D
    "(i,j,k) positions of the corners surrounding the source location"
    ijksrc::MArray
     "(x,y,z) position of the source"
    xyzsrc::MVector
    "coefficient matrix for the interpolation of the velocity"
    velcorn::MVector
    "distance from the corners to the source location"
    distcorn::MVector
end


####################################

mutable struct SourcePtsFromFineGrid
    ijsrc::Matrix{Int64} 
end

######################################################

struct FMMvars2D
    ttime::Array{Float64,2}
    status::Array{UInt8,2}
    bheap::BinHeapMin
    "refine grid around source"
    refinearoundsrc::Bool
    "brute-force fix negative saqarg"
    allowfixsqarg::Bool 
    srcboxpar::Union{SourceBoxParams2D,SourcePtsFromFineGrid}

    function FMMvars2D(n1,n2; amIcoarsegrid::Bool,refinearoundsrc,allowfixsqarg)
        ttime = zeros(Float64,n1,n2)
        status = zeros(UInt8,n1,n2)
        bheap = init_minheap(n1*n2)
        begin
            if allowfixsqarg==true
                @warn("ExtraParams: allowfixsqarg==true, brute-force fixing of negative discriminant allowed.")
            end
        end
        
        if  amIcoarsegrid && refinearoundsrc==true
            ## there is refinement
            srcboxpar = SourcePtsFromFineGrid(Array{Int64,2}(undef,0,2))
        else
            ## there is NO refinement
            Ncoe = 4
            srcboxpar = SourceBoxParams2D(MMatrix{Ncoe,2}(zeros(Int64,Ncoe,2)),
                                          MVector{2}(zeros(2)),
                                          MVector{Ncoe}(zeros(Ncoe)),
                                          MVector{Ncoe}(zeros(Ncoe)) )
        end
        new(ttime,status,bheap,refinearoundsrc,allowfixsqarg,srcboxpar)
    end
end

######################################################

struct FMMvars3D
    ttime::Array{Float64,3}
    status::Array{UInt8,3}
    bheap::BinHeapMin
    "refine grid around source"
    refinearoundsrc::Bool
    "brute-force fix negative saqarg"
    allowfixsqarg::Bool 
    srcboxpar::SourceBoxParams3D
    
    function FMMvars3D(n1,n2,n3; amIcoarsegrid::Bool,refinearoundsrc,allowfixsqarg)
        ttime = zeros(Float64,n1,n2,n3)
        status = zeros(UInt8,n1,n2,n3)
        bheap = init_minheap(n1*n2*n3)
        begin
            if allowfixsqarg==true
                @warn("ExtraParams: allowfixsqarg==true, brute-force fixing of negative discriminant allowed.")
            end
        end

        if amIcoarsegrid && refinearoundsrc==true
            srcboxpar = SourcePtsFromFineGrid(Array{Int64,2}(undef,0,3))
        else
            Ncoe = 8
            srcboxpar = SourceBoxParams3D(MMatrix{Ncoe,2}(zeros(Int64,Ncoe,3)),
                                          MVector{3}(zeros(3)),
                                          MVector{Ncoe}(zeros(Ncoe)),
                                          MVector{Ncoe}(zeros(Ncoe)) )                                          
        end
        new(ttime,status,bheap,refinearoundsrc,allowfixsqarg,srcboxpar)
    end
end

###################################################

struct SrcRefinVars2D
    downscalefactor::Int64
    ijcoarse::NTuple{4,Int64}
    nxny_window_coarse::NTuple{2,Int64}
    outxyminmax::NTuple{4,Bool}
    nearneigh_oper::SparseMatrixCSC{Float64,Int64}
    nearneigh_idxcoarse::Vector{Int64}
    vel2d_fine::Array{Float64,2}
end

###################################################
