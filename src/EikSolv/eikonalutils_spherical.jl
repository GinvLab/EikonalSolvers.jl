
######################################################

"""
$(TYPEDSIGNATURES)

Bilinear interpolation (spherical coordinates).
"""
function bilinear_interp_sph(f::Array{Float64,2},grdsph::Grid2DSphere,
                             xreq::Float64,yreq::Float64)
    xmin = grdsph.rinit
    ymin = grdsph.θinit
    dx = grdsph.Δr
    dy = grdsph.Δθ
    nx = grdsph.nr
    nx = grdsph.nθ    
    
    nx,ny = size(f)
    ## rearrange such that the coordinates of corners are (0,0), (0,1), (1,0), and (1,1)
    xh=(xreq-xmin)/dx
    yh=(yreq-ymin)/dy
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
    intval = f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd +
        f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
    
    return intval
end


#############################################################
"""
$(TYPEDSIGNATURES)

Trinilear interpolation in spherical coordinates.
"""
function trilinear_interp_sph(ttime::Array{Float64,3},grdsph::Grid3DSphere,xin::Float64,yin::Float64,zin::Float64)

    ## All this function needs a check!!!  <<<<<=========####
    ## Is it ok to do this interpolation in spherical coordinates??
    
    xh = (xin-grdsph.rinit)/grdsph.Δr
    yh = (yin-grdsph.θinit)/grdsph.Δθ
    zh = (zin-grdsph.φinit)/grdsph.Δφ
    i = floor(Int64,xh)
    j = floor(Int64,yh)
    k = floor(Int64,zh)

    x = xin-grdsph.rinit
    y = yin-grdsph.θinit
    z = zin-grdsph.φinit

    ## if at the edges of domain choose previous square...
    nr,nθ,nφ=grdsph.nr,grdsph.nθ,grdsph.nφ
    if i==nr
        i=i-1
    end
    if j==nθ
        j=j-1
    end
    if k==nφ
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

    ##
    x0=i*grdsph.Δr
    y0=j*grdsph.Δθ
    z0=k*grdsph.Δφ
    x1=(i+1)*grdsph.Δr
    y1=(j+1)*grdsph.Δθ
    z1=(k+1)*grdsph.Δφ
    
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


#########################################################################

"""
$(TYPEDSIGNATURES)

Find closest node on a 2D grid to a given point in spherical/polar coordinates.
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


#####################################################################

"""
$(TYPEDSIGNATURES)

Find closest node on a 3D grid to a given point in spherical coordinates.
"""
function findclosestnode_sph(r::Float64,θ::Float64,φ::Float64,rinit::Float64,θinit::Float64,
                             φinit::Float64,Δr::Float64,Δθ::Float64,Δφ::Float64)
    # rini ???
    # θini ???
    ir = floor((r-rinit)/Δr)
    iθ = floor((θ-θinit)/Δθ)
    iφ = floor((φ-φinit)/Δφ)
    rr = r-(ir*Δr+rinit)
    rθ = θ-(iθ*Δθ+θinit)
    rφ = φ-(iφ*Δφ+φinit)
 
    middler = Δr/2.0
    middleθ = Δθ/2.0
    middleφ = Δφ/2.0

    if rr>=middler 
        ir = ir+1
    end
    if rθ>=middleθ
        iθ = iθ+1
    end
    if rφ>=middleφ
        iφ = iφ+1
    end
    return Int(ir+1),Int(iθ+1),Int(iφ+1) # julia
end

#############################################################
