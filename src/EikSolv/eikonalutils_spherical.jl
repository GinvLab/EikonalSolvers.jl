
######################################################
"""
A structure holding the 2D SPHERICAL grid parameters, geometry and size.

The constructor is given by:

     Grid2DSphere(; Δr::Float64,Δθ::Float64,rinit::Float64,θinit::Float64,nr::Int64,nθ::Int64)

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
#Base.@kwdef
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
     bilinear_interp_sph(f::Array{Float64,2},grdsph::Grid2DSphere,
                             xreq::Float64,yreq::Float64)

Bilinear interpolation (spherical coordinates).
"""
function bilinear_interp_sph(f::Array{Float64,2},grdsph::Grid2DSphere,
                             xreq::Float64,yreq::Float64)
    xinit = grdsph.rinit
    yinit = grdsph.θinit
    dx = grdsph.Δr
    dy = grdsph.Δθ
    nx = grdsph.nr
    nx = grdsph.nθ
    
    
    nx,ny = size(f)
    ## rearrange such that the coordinates of corners are (0,0), (0,1), (1,0), and (1,1)
    xh=(xreq-xinit)/dx
    yh=(yreq-yinit)/dy
    i=floor(Int64,xh+1) # indices starts from 1
    j=floor(Int64,yh+1) # indices starts from 1
  
    ## rearrange such that the coordinates of corners are (0,0), (0,1), (1,0), and (1,1)
    xh=(xreq-xinit)/dx
    yh=(yreq-yinit)/dy
    i=floor(Int64,xh+1) # indices starts from 1
    j=floor(Int64,yh+1) # indices starts from 1
    ## if at the edges of domain choose previous square...
    if i==nx
        i=i-1
    end
    if j==ny
        j=j-1
    end  ## if at the edges of domain choose previous square...
    if i==nx
        i=i-1
    end
    if j==ny
        j=j-1
    end
    xd=xh-(i-1) # indices starts from 1
    yd=yh-(j-1) # indices starts from 1
    intval = f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd +
        f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
    
    # println("------------")
    # @show xreq yreq xh yh 
    # @show i j xd yd
    # @show f[i:i+1,j:j+1], intval
    return intval
end

##########################################################################

"""
    findclosestnode_sph(x::Float64,y::Float64,z::Float64,xinit::Float64,
                        yinit::Float64,zinit::Float64,Δr::Float64,Δθ::Float64) 

Find closest node on a grid to a given point.
"""
function findclosestnode_sph(r::Float64,θ::Float64,rinit::Float64,θinit::Float64,
                             Δr::Float64,Δθ::Float64) 
    # xini ???
    # yini ???
    ir = floor((r-rinit)/Δr)
    iθ = floor((θ-θinit)/Δθ)

    rx = r-(ir*Δr+rinit)
    ry = θ-(iθ*Δθ+θinit)

    middler = Δr/2.0
    middleθ = Δθ/2.0

    if rx>=middler
        ir = ir+1
    end
    if ry>=middleθ  
        iθ = iθ+1
    end
    return Int(ir+1),Int(iθ+1) # julia
end


#############################################################
