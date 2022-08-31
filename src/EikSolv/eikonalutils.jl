
#######################################################
##        Eikonal utilities                          ## 
#######################################################

######################################################
"""
$(TYPEDSIGNATURES)

Calculate how to subdivide the sources among workers for parallel jobs.
"""
function distribsrcs(nsrc::Integer,nw::Integer)
    ## calculate how to subdivide the srcs among the workers
    if nsrc>=nw
        dis = div(nsrc,nw)
        grpsizes = dis*ones(Int64,nw)        
        resto = mod(nsrc,nw)
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


"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Bilinear interpolation.
"""
function bilinear_interp(f::AbstractArray{Float64,2},grd::Union{Grid2D,Grid2DSphere}, xreq::Float64,yreq::Float64;
                         return_coeffonly::Bool=false)

    if typeof(grd)==Grid2D
        dx = grd.hgrid
        dy = grd.hgrid
        xinit = grd.xinit
        yinit = grd.yinit

    elseif typeof(grd)==Grid2DSphere
        dx = grd.Δr
        dy = grd.Δθ
        xinit = grd.rinit
        yinit = grd.θinit

    end
    
    nx,ny = size(f)
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
    end 

    xd=xh-(i-1) # indices starts from 1
    yd=yh-(j-1) # indices starts from 1

    if return_coeffonly
        coeff = @SVector[(1.0-xd)*(1.0-yd),
                         (1.0-yd)*xd, 
                         (1.0-xd)*yd,
                         xd*yd]
        ijs = @SMatrix[i   j;
                      i+1 j;
                      i j+1;
                      i+1 j+1]
        return coeff,ijs

    else
        intval = f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd +
            f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
        return intval

    end

    return intval
end

#############################################################

"""
$(TYPEDSIGNATURES)

Trinilear interpolation.
"""
function trilinear_interp(ttime::AbstractArray{Float64,3},grd::Union{Grid3D,Grid3DSphere},
                          xyzpt::AbstractVector{Float64}; return_coeffonly::Bool=false)
 
    xin,yin,zin = xyzpt[1],xyzpt[2],xyzpt[3]

    if typeof(grd)==Grid3D
        dx = grd.hgrid
        dy = grd.hgrid
        dz = grd.hgrid
        xinit = grd.xinit
        yinit = grd.yinit
        zinit = grd.zinit

    elseif typeof(grd)==Grid3DSphere
        dx = grd.Δr
        dy = grd.Δθ
        dz = grd.Δφ
        xinit = grd.rinit
        yinit = grd.θinit
        zinit = grd.φinit

    end
    
    xh = (xin-xinit)/dx
    yh = (yin-yinit)/dy
    zh = (zin-zinit)/dz
    i = floor(Int64,xh)
    j = floor(Int64,yh)
    k = floor(Int64,zh)
    x = xin-xinit  
    y = yin-yinit
    z = zin-zinit

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
        println("trilinear_interp(): Interpolation failed...")
        println("$i,$xh,$j,$yh,$k,$zh")
        return
    elseif ((i+1<xh)|(j+1<yh)|(k+1<zh))
        println("trilinear_interp(): Interpolation failed...")
        println("$i,$xh,$j,$yh,$k,$zh")
        return
    end 

    # @show i,j,k
    # @show x,y,z

    x0=i*dx
    y0=j*dy
    z0=k*dz
    x1=(i+1)*dx
    y1=(j+1)*dy
    z1=(k+1)*dz

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
    
    # ## where x_0 indicates the lattice point below x , and x_1 indicates the lattice point above x and similarly for y_0, y_1, z_0 and z_1.
    # ## First we interpolate along x (imagine we are pushing the front face of the cube to the back), giving:
    # c00 = f000 * (1 - xd) + f100 * xd 
    # c10 = f010 * (1 - xd) + f110 * xd 
    # c01 = f001 * (1 - xd) + f101 * xd 
    # c11 = f011 * (1 - xd) + f111 * xd 

    # ## Where V[x_0,y_0, z_0] means the function value of (x_0,y_0,z_0). Then we interpolate these values (along y, as we were pushing the top edge to the bottom), giving:
    # c0 = c00 * (1 - yd) + c10 *yd
    # c1 = c01 * (1 - yd) + c11 *yd

    # ## Finally we interpolate these values along z(walking through a line):
    # interpval = c0 * (1 - zd) + c1 * zd


    # ## Finally we interpolate these values along z(walking through a line):
    # interpval = c0 * (1 - zd) + c1 * zd 

    ##########################################################33

    if return_coeffonly

        coeff = @SVector [(1-xd)*(1-yd)*(1-zd) ,
                          xd*(1-yd)*(1-zd) ,
                          (1-xd)*yd*(1-zd) ,
                          (1-xd)*(1-yd)*zd ,
                          xd*(1-yd)*(1-zd) ,
                          (1-xd)*yd*zd ,
                          xd*yd*(1-zd) ,
                          xd*yd*zd ]
        
        ijs = @SMatrix [ii   jj    kk;
                        ii  jj+1   kk;
                        ii   jj    kk+1;
                        ii  jj+1   kk+1;
                        ii+1  jj    kk;
                        ii+1  jj+1  kk;
                        ii+1  jj    kk+1;
                        ii+1  jj+1  kk+1 ]

        return coeff,ijs

    else

        interpval = f000 * (1-xd)*(1-yd)*(1-zd) +
            f100 * xd*(1-yd)*(1-zd) +
            f010 * (1-xd)*yd*(1-zd) +
            f001 * (1-xd)*(1-yd)*zd +
            f101 * xd*(1-yd)*(1-zd) +
            f011 * (1-xd)*yd*zd +
            f110 * xd*yd*(1-zd) +
            f111 * xd*yd*zd

        return interpval
    end
    # @show ii,jj,kk
    # @show ttime[ii,jj,kk],interpval

    return interpval
end

###################################################################

"""
$(TYPEDSIGNATURES)

Calculate the misfit functional 
```math
    S = \\dfrac{1}{2} \\sum_i \\dfrac{\\left( \\mathbf{\\tau}_i^{\\rm{calc}}(\\mathbf{v})-\\mathbf{\\tau}_i^{\\rm{obs}} \\right)^2}{\\sigma_i^2} \\, .
```
# Arguments
    - `velmod`: velocity model, either a 2D or 3D array.
    - `ttpicksobs`: a vector of vectors of the traveltimes at the receivers.
    - `stdobs`: a vector of vectors standard deviations representing the error on the measured traveltimes.
    - `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
    - `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
    - `grd`: the struct holding the information about the grid, one of `Grid2D`,`Grid3D`,`Grid2Dsphere`,`Grid3Dsphere`

# Returns
    The value of the misfit functional (L2-norm), the same used to compute the gradient with adjoint methods.

"""
function ttmisfitfunc(velmod::Union{Array{Float64,2},Array{Float64,3}},ttpicksobs,
                      stdobs,coordsrc,
                      coordrec,grd::Union{Grid2D,Grid3D,Grid2DSphere,Grid3DSphere} )
                      
# function ttmisfitfunc(velmod::Union{Array{Float64,2},Array{Float64,3}},ttpicksobs::Vector{Vector{Float64}},
#                       stdobs::Vector{Vector{Float64}},coordsrc::Array{Float64,2},
#                       coordrec::Vector{Array{Float64,2}},grd::Union{Grid2D,Grid3D,Grid2DSphere,Grid3DSphere};
#                       ttalgo::String="ttFMM_hiord")

    if typeof(grd)==Grid2D 
        # compute the forward response
        ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)

    elseif typeof(grd)==Grid2DSphere
        # compute the forward response
        ttpicks = traveltime2Dsphere(velmod,grd,coordsrc,coordrec)

    elseif typeof(grd)==Grid3D 
        # compute the forward response
        ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

    elseif typeof(grd)==Gride3DSphere
        # compute the forward response
        ttpicks = traveltime3Dsphere(velmod,grd,coordsrc,coordrec)

    else
        error("Input velocity model has wrong dimensions.")
    end

    nsrc = size(coordsrc,1)
    #nrecs1 = size.(coordrec,1)
    #totlen = sum(nrec1)
    misf = 0.0
    for s=1:nsrc
        misf += sum( (ttpicks[s].-ttpicksobs[s]).^2 ./ stdobs[s].^2)
    end
    misf *= 0.5

    # flatten traveltime array
    # dcalc = vec(ttpicks)
    # dobs = vec(ttpicksobs)
    # stdobsv = vec(stdobs)
    # ## L2 norm^2
    # diffcalobs = dcalc .- dobs 
    # misf = 0.5 * sum( diffcalobs.^2 ./ stdobsv.^2 )

    return misf
end

###################################################

function addentry!(D::VecSPDerivMat,i::Integer,j::Integer,v::Float64)
    p = D.Nnnz[]+1
    D.i[p] = i
    D.j[p] = j
    D.v[p] = v
    D.Nnnz[] = p 
    return
end

###################################################

function cart2lin2D(i::Integer,j::Integer,ni::Integer)
    l = i + (j-1)*ni
    return l
end

function cart2lin3D(i::Integer,j::Integer,k::Integer,ni::Integer,nj::Integer)
    l = i + ni*( (j-1)+nj*(k-1) )
    return l
end

###################################################

function lin2cart2D!(l::Integer,ni::Integer,point::AbstractVector) 
    point[2] = div(l-1,ni) + 1
    point[1] = l-(point[2]-1)*ni
    return 
end

function lin2cart3D!(l::Integer,ni::Integer,nj::Integer,point::AbstractVector) 
    point[3] = div(l-1,ni*nj) + 1
    point[2] = div(l-1-(point[3]-1)*ni*nj, ni) + 1
    point[1] = l-(point[2]-1)*ni - (point[3]-1)*ni*nj  
    return 
end

############################################################

function isacoarsegridnode(i::Int,j::Int,downscalefactor::Int,i1coarse::Int,j1coarse::Int)
    a = rem(i, downscalefactor)
    b = rem(j, downscalefactor)
    if a==b==0
        icoa = div(i,downscalefactor)+i1coarse
        jcoa = div(j,downscalefactor)+j1coarse
        return true,icoa,jcoa
    else
        return false,nothing,nothing
    end
    return
end


function isacoarsegridnode(i::Int,j::Int,k::Int,downscalefactor::Int,i1coarse::Int,j1coarse::Int,k1coarse::Int)
    a = rem(i, downscalefactor)
    b = rem(j, downscalefactor)
    c = rem(k, downscalefactor)
    if a==b==c==0
        icoa = div(i,downscalefactor)+i1coarse
        jcoa = div(j,downscalefactor)+j1coarse
        kcoa = div(k,downscalefactor)+k1coarse
        return true,icoa,jcoa,kcoa
    else
        return false,nothing,nothing,nothing
    end
    return
end

##############################################################################
