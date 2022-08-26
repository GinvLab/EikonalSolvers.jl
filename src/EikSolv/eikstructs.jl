

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
struct Grid2D #Base.@kwdef 
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
struct Grid3D #Base.@kwdef
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
struct Grid2DSphere
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
struct Grid3DSphere
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

##############################################################################

struct MapOrderGridFMM2D
    "Linear grid indices in FMM order (as visited by FMM)"
    lfmm2grid::Vector{Int64} # idx_fmmord
    "Linear FMM indices in grid order"
    lgrid2fmm::Vector{Int64} # idx_gridord
    # cart2lin::LinearIndices
    # lin2cart::CartesianIndices
    nx::Int64
    ny::Int64

    function MapOrderGridFMM2D(nx,ny)
        nxy = nx*ny
        lfmm2grid = zeros(Int64,nxy)
        lgrid2fmm = zeros(Int64,nxy)
        # cart2lin = LinearIndices((nx,ny))
        # lin2cart = CartesianIndices((nx,ny))
        #return new(lfmm2grid,lgrid2fmm,cart2lin,lin2cart,nx,ny)
        return new(lfmm2grid,lgrid2fmm,nx,ny)
    end
end


struct MapOrderGridFMM3D
    "Linear grid indices in FMM order (as visited by FMM)"
    lfmm2grid::Vector{Int64} # idx_fmmord
    "Linear FMM indices in grid order"
    lgrid2fmm::Vector{Int64} # idx_gridord
    # cart2lin::LinearIndices
    # lin2cart::CartesianIndices
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

# structs for sparse matrices to represent derivatives (discrete adjoint)

struct VecSPDerivMat
    i::Vector{Int64}
    j::Vector{Int64}
    v::Vector{Float64}
    Nnnz::Base.RefValue{Int64}
    Nsize::Vector{Int64}

    function VecSPDerivMat(; i,j,v,Nsize)
        Nnnz = Ref(0)
        new(i,j,v,Nnnz,Nsize) 
    end
end

########################################################

struct VarsFMMOrder2D
    ttime::Vector{Float64}
    vecDx::VecSPDerivMat
    vecDy::VecSPDerivMat

    function VarsFMMOrder2D(nx,ny)
        nxy =  nx*ny
        ttime = zeros(nxy)
        vecDx = VecSPDerivMat( i=zeros(Int64,nxy*3), j=zeros(Int64,nxy*3),
                               v=zeros(nxy*3), Nsize=[nxy,nxy] )
        vecDy = VecSPDerivMat( i=zeros(Int64,nxy*3), j=zeros(Int64,nxy*3),
                               v=zeros(nxy*3), Nsize=[nxy,nxy] )
        return new(ttime,vecDx,vecDy)
    end
end

struct VarsFMMOrder3D
    ttime::Vector{Float64}
    vecDx::VecSPDerivMat
    vecDy::VecSPDerivMat
    vecDz::VecSPDerivMat

    function VarsFMMOrder3D(nx,ny,nz)
        nxyz =  nx*ny*nz
        ttime = zeros(nxyz)
        vecDx = VecSPDerivMat( i=zeros(Int64,nxyz*3), j=zeros(Int64,nxyz*3),
                               v=zeros(nxyz*3), Nsize=[nxyz,nxyz] )
        vecDy = VecSPDerivMat( i=zeros(Int64,nxyz*3), j=zeros(Int64,nxyz*3),
                               v=zeros(nxyz*3), Nsize=[nxyz,nxyz] )
        vecDz = VecSPDerivMat( i=zeros(Int64,nxyz*3), j=zeros(Int64,nxyz*3),
                               v=zeros(nxyz*3), Nsize=[nxyz,nxyz] )

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



###################################################
