
#######################################################
##        Eikonal utilities                          ## 
#######################################################

######################################################
"""
    distribsrcs(nsrc::Integer,nw::Integer)

Calculate how to subdivide the sources among workers for parallel jobs.
"""
function distribsrcs(nsrc::Integer,nw::Integer)
    ## calculate how to subdivide the srcs among the workers
    if nsrc>=nw
        dis = div(nsrc,nw)
        grpsizes = dis*ones(Int64,nw)        
        resto = mod(nsrc,dis)
        if resto>0
            ## add the reminder 
            grpsizes[1:resto] .+= 1
        end
    else
        ## if more workers than sources use only necessary workers
        grpsizes = ones(Int64,nsrc)        
    end
    ## now set the indices for groups of srcs
    grpsrc = zeros(Int64,length(grpsizes),2)
    grpsrc[:,1] = cumsum(grpsizes).-grpsizes.+1
    grpsrc[:,2] = cumsum(grpsizes)
    return grpsrc
end

######################################################
"""
A structure holding the 2D grid parameters, geometry and size.

The constructor is given by:

     Grid2D(hgrid::Float64,xinit::Float64,yinit::Float64,nx::Int64,ny::Int64)

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
Base.@kwdef struct Grid2D
    hgrid::Float64
    xinit::Float64
    yinit::Float64
    nx::Int64
    ny::Int64
    ntx::Int64
    nty::Int64

    function Grid2D(hgrid::Float64,xinit::Float64,yinit::Float64,nx::Int64,ny::Int64)
        ntx::Int64 = nx+1
        nty::Int64 = ny+1
        new(hgrid,xinit,yinit,nx,ny,ntx,nty)
    end
end

######################################################

"""
A structure holding the 3D grid parameters, geometry and size.

The constructor is given by:

     Grid2D(hgrid::Float64,xinit::Float64,yinit::Float64,zinit::Float64,
            nx::Int64,ny::Int64,nz::Int64)

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
Base.@kwdef struct Grid3D
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
    ## constructor function
    function Grid3D(hgrid::Float64,xinit::Float64,yinit::Float64, zinit::Float64,
                    nx::Int64,ny::Int64,nz::Int64)
        ntx::Int64 = nx+1
        nty::Int64 = ny+1
        ntz::Int64 = nz+1
        new(hgrid,xinit,yinit,zinit,nx,ny,nz,ntx,nty,ntz)
    end
end

########################################################

"""
    findclosestnode(x::Float64,y::Float64,xinit::Float64,
                    yinit::Float64,h::Float64) 

Find closest node on a grid to a given point.
"""
function findclosestnode(x::Float64,y::Float64,xinit::Float64,yinit::Float64,h::Float64) 
    # xini ???
    # yini ???
    ix = floor((x-xinit)/h)
    iy = floor((y-yinit)/h)
    rx = x-(ix*h+xinit)
    ry = y-(iy*h+yinit)
    middle = h/2.0
    if rx>=middle 
        ix = ix+1
    end
    if ry>=middle 
        iy = iy+1
    end
    return Int(ix+1),Int(iy+1) # julia
end
    
#####################################################################
"""
    findclosestnode(x::Float64,y::Float64,z::Float64,xinit::Float64,
                    yinit::Float64,zinit::Float64,h::Float64) 

Find closest node on a grid to a given point.
"""
function findclosestnode(x::Float64,y::Float64,z::Float64,xinit::Float64,yinit::Float64,zinit::Float64,h::Float64) 
    # xini ???
    # yini ???
    ix = floor((x-xinit)/h)
    iy = floor((y-yinit)/h)
    iz = floor((z-zinit)/h)
    rx = x-(ix*h+xinit)
    ry = y-(iy*h+yinit)
    rz = z-(iz*h+zinit)
    middle = h/2.0

    if rx>=middle 
        ix = ix+1
    end
    if ry>=middle 
        iy = iy+1
    end
    if rz>=middle 
        iz = iz+1
    end
    return Int(ix+1),Int(iy+1),Int(iz+1) # julia
end

#############################################################

"""
     bilinear_interp(f::Array{Float64,2},hgrid::Float64,
                         xinit::Float64,yinit::Float64, xreq::Float64,yreq::Float64)

Bilinear interpolation.
"""
function bilinear_interp(f::Array{Float64,2},hgrid::Float64,
                         xinit::Float64,yinit::Float64, xreq::Float64,yreq::Float64)
    nx,ny = size(f)
    ## rearrange such that the coordinates of corners are (0,0), (0,1), (1,0), and (1,1)
    xh=(xreq-xinit)/hgrid
    yh=(yreq-yinit)/hgrid
    i=floor(Int64,xh+1) # indices starts from 1
    j=floor(Int64,yh+1) # indices starts from 1
  
    ## rearrange such that the coordinates of corners are (0,0), (0,1), (1,0), and (1,1)
    xh=(xreq-xinit)/hgrid
    yh=(yreq-yinit)/hgrid
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

#############################################################
"""
    trilinear_interp(ttime::Array{Float64,3},ttgrdspacing::Float64,
                      xinit::Float64,yinit::Float64,zinit::Float64,
                      x::Float64,y::Float64,z::Float64)

Trinilear interpolation.
"""
function trilinear_interp(ttime::Array{Float64,3},ttgrdspacing::Float64,
                      xinit::Float64,yinit::Float64,zinit::Float64,
                      x::Float64,y::Float64,z::Float64)
    
    xh = (x-xinit)/ttgrdspacing
    yh = (y-yinit)/ttgrdspacing
    zh = (z-zinit)/ttgrdspacing
    i = floor(Int64,xh)
    j = floor(Int64,yh)
    k = floor(Int64,zh)

    ## if at the edges of domain choose previous square...
    nx,ny,nz=size(ttime)
    if i==nx
        i=i-1
    end
    if j==ny
        j=j-1
    end
    if k==nz
        k=k-1
    end
    
    if ((i>xh) | (j>yh) | (k>zh)) 
        println("Interpolation failed...")
        println("$i,$xh,$j,$yh,$k,$zh")
        return
    elseif ((i+1<xh)|(j+1<yh)|(k+1<zh))
        println("Interpolation failed...")
        println("$i,$xh,$j,$yh,$k,$zh")
        return
    end 
    # print*,x,y,z

    x0=i*ttgrdspacing
    y0=j*ttgrdspacing
    z0=k*ttgrdspacing
    x1=(i+1)*ttgrdspacing
    y1=(j+1)*ttgrdspacing
    z1=(k+1)*ttgrdspacing

    ## Fortran indices start from 1 while i,j,k from 0
    ii=i+1
    jj=j+1
    kk=k+1
    f000 = ttime[ii,jj,  kk] 
    f010 = ttime[ii,jj+1,kk]
    f001 = ttime[ii,jj,  kk+1]
    f011 = ttime[ii,jj+1,kk+1]

    f100 = ttime[ii+1,jj,  kk] 
    f110 = ttime[ii+1,jj+1,kk]
    f101 = ttime[ii+1,jj,  kk+1]
    f111 = ttime[ii+1,jj+1,kk+1]

    ## On a periodic and cubic lattice, let x_d, y_d, and z_d be the differences between each of x, y, z and the smaller coordinate related, that is:
    xd = (x - x0)/(x1 - x0)
    yd = (y - y0)/(y1 - y0)
    zd = (z - z0)/(z1 - z0)
    
    ## where x_0 indicates the lattice point below x , and x_1 indicates the lattice point above x and similarly for y_0, y_1, z_0 and z_1.
    ## First we interpolate along x (imagine we are pushing the front face of the cube to the back), giving:
    c00 = f000 * (1 - xd) + f100 * xd 
    c10 = f010 * (1 - xd) + f110 * xd 
    c01 = f001 * (1 - xd) + f101 * xd 
    c11 = f011 * (1 - xd) + f111 * xd 

    ## Where V[x_0,y_0, z_0] means the function value of (x_0,y_0,z_0). Then we interpolate these values (along y, as we were pushing the top edge to the bottom), giving:
    c0 = c00 * (1 - yd) + c10 *yd
    c1 = c01 * (1 - yd) + c11 *yd

    ## Finally we interpolate these values along z(walking through a line):
    interpval = c0 * (1 - zd) + c1 * zd 
      
    return interpval
end

###################################################################

@doc raw"""
     misfitfunc(velmod::Array{Float64},ttpicksobs::Array{Float64},stdobs::Vector{Float64},grd::Union{Grid2D,Grid3D})

Calculate the misfit functional 
```math
    S = \dfrac{1}{2} \sum_i \dfrac{\left( \mathbf{\tau}_i^{\rm{calc}}(\mathbf{v})-\mathbf{\tau}_i^{\rm{obs}} \right)^2}{\sigma_i^2} \, .
```
# Arguments
    * `velmod`: velocity model, either a 2D or 3D array.
    * `ttpicksobs`: the traveltimes at the receivers.
    * `stdobs`: a vector of standard deviations representing the error on the measured traveltimes.

# Returns
    The value of the misfit functional (L2-norm).

"""
function misfitfunc(velmod::Array{Float64},ttpicksobs::Array{Float64},
                    stdobs::Vector{Float64},coordsrc::Array{Float64,2},
                    coordred::Array{Float64,2},grd::Union{Grid2D,Grid3D})

    if ndims(velmod)==2
        # compute the forward response
        ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
    elseif ndims(velmod)==3
        # compute the forward response
        ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)
    else
        error("Input velocity model has wrong dimensions.")
    end

    # flatten traveltime array
    dcalc = ttpicks[:]
    dobs = ttpicksobs[:]

    ## L2 norm
    diffcalobs = dcalc .- dobs
    misf = 0.5 .* sum( diffcalobs.^2 ./ stdobs  )
    
    return misf
end

###################################################
#end # module EikUtils                           ##
###################################################


