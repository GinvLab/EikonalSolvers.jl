

###################################################

## 2D
function cart2lin(ij::Union{SVector{2,<:Integer},MVector{2,<:Integer}},
                  nij::NTuple{2,Integer})::Integer
    #l = i + (j-1)*ni
    l = ij[1] + (ij[2]-1)*nij[1]
    return l
end

## 3D 
function cart2lin(ijk::Union{SVector{3,<:Integer},MVector{3,<:Integer}},
                  nijk::NTuple{3,Integer})::Integer
    #l = i + ni*( (j-1)+nj*(k-1) )
    l = ijk[1] + nijk[1]*( (ijk[2]-1)+nijk[2]*(ijk[3]-1) )
    return l
end

###################################################

## 2D 
function lin2cart!(l::Integer,nij::NTuple{2,Integer},point::MVector{2,<:Integer})
    # point[2] = div(l-1,nij) + 1
    # point[1] = l-(point[2]-1)*ni
    point[2] = div(l-1,nij[1]) + 1
    point[1] = l-(point[2]-1)*nij[1]
    return 
end

## 3D 
function lin2cart!(l::Integer,nijk::NTuple{3,Integer},point::MVector{3,<:Integer})
    # point[3] = div(l-1,ni*nj) + 1
    # point[2] = div(l-1-(point[3]-1)*ni*nj, ni) + 1
    # point[1] = l-(point[2]-1)*ni - (point[3]-1)*ni*nj
    ninj = nijk[1]*nijk[2]
    ni = nijk[1]
    point[3] = div(l-1,ninj) + 1
    point[2] = div(l-1-(point[3]-1)*ninj, ni) + 1
    point[1] = l-(point[2]-1)*ni - (point[3]-1)*ninj
    return 
end

#############################################################

@inline function distance2points(::Union{Grid2DCart,Grid3DCart},
                                 xyz1::AbstractVector,xyz2::AbstractVector)
    dist = sqrt( sum((xyz1 .- xyz2 ).^2) )
    return dist
end


@inline function distance2points(::Grid2DSphere,
                                 rθ1::AbstractVector,rθ2::AbstractVector;
                                 degrees::Bool=true)
    r1,θ1 = rθ1
    r2,θ2 = rθ2
    if degrees
        dist = sqrt( r1^2 + r2^2 - 2*r1*r2*cosd(θ1-θ2) )
    else
        dist = sqrt( r1^2 + r2^2 - 2*r1*r2*cos(θ1-θ2) )
    end
    return dist
end


@inline function distance2points(::Grid3DSphere,
                                 rθφ1::AbstractVector,rθφ2::AbstractVector;
                                 degrees::Bool=true)

    error("sphericaldistance(): not yet implemented.")
end

############################################################

## 2D
function isacoarsegridnode(ij::MVector{2,<:Integer},downscalefactor::Int,
                           ijcoarse::NTuple{2,Int64})
    i1coarse = ijcoarse[1]
    j1coarse = ijcoarse[2]
    i = ij[1]
    j = ij[2]
    a = rem(i-1, downscalefactor) 
    b = rem(j-1, downscalefactor) 
    if a==b==0
        icoa = div(i-1,downscalefactor)+i1coarse
        jcoa = div(j-1,downscalefactor)+j1coarse
        # @show i,j,downscalefactor,i1coarse,j1coarse
        # @show icoa,jcoa
        return true,MVector(icoa,jcoa)
    else
        return false,nothing
    end
    return
end

## 3D
function isacoarsegridnode(ijk::MVector{3,<:Integer},downscalefactor::Int,
                           ijkcoarse::NTuple{3,Int64})
    i1coarse = ijkcoarse[1]
    j1coarse = ijkcoarse[2]
    k1coarse = ijkcoarse[3]
    i = ijk[1]
    j = ijk[2]
    k = ijk[3]
    a = rem(i-1, downscalefactor)
    b = rem(j-1, downscalefactor)
    c = rem(k-1, downscalefactor)
    if a==b==c==0
        icoa = div(i-1,downscalefactor)+i1coarse
        jcoa = div(j-1,downscalefactor)+j1coarse
        kcoa = div(k-1,downscalefactor)+k1coarse
        return true,MVector(icoa,jcoa,kcoa)
    else
        return false,nothing
    end
    return
end


#################################################################################

"""
$(TYPEDSIGNATURES)

 Test if point is on borders of 2D domain.
"""
function isonbord(ib::Int64,jb::Int64,n1::Int64,n2::Int64)
    isonb1st = false
    isonb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>n1 || jb<1 || jb>n2
        isonb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>n1-1 || jb<2 || jb>n2-1
        isonb2nd =  true
    end
    return isonb1st,isonb2nd
end

############################################################

"""
$(TYPEDSIGNATURES)

 Test if point is on borders of a 3D domain.
"""
function isonbord(ib::Int64,jb::Int64,kb::Int64,nx::Int64,ny::Int64,nz::Int64)
    isonb1 = false
    isonb2 = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>nx || jb<1 || jb>ny || kb<1 || kb>nz
        isonb1 =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>nx-1 || jb<2 || jb>ny-1 || kb<2 || kb>nz-1 
        isonb2 =  true
    end
    return isonb1,isonb2
end

##############################################################################

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


##############################################################################

"""

$(TYPEDSIGNATURES)

Ray tracing utility. Given a traveltime grid and source and receiver positions, trace the rays.
"""
function tracerays(grd::AbstractGridEik,ttime::Vector{Array{Float64}},
                   coordsrc::Array{Float64,2},coordrec::Vector{Array{Float64,2}};
                   steplen::Real=0.01)
    
    nsrc = size(coordsrc,1)
    rays_srcs = Vector{Vector{Matrix{Float64}}}(undef,nsrc)
    for s=1:nsrc
        rays_srcs[s] = tracerays_singlesrc(grd,ttime[s],coordsrc[s,:],coordrec[s],steplen=steplen)
    end
    
    return rays_srcs
end


##############################################################################

"""

$(TYPEDSIGNATURES)

Ray tracing utility. Given a traveltime grid and source and receiver positions, trace the rays.
"""
function tracerays_singlesrc(grd::AbstractGridEik,ttime::Array{Float64,N},
                             srccoo::AbstractVector,coordrec::Array{Float64,2};
                             steplen::Real) where N

    # setup interpolation
    if typeof(grd)==Grid2DCart
        Ndim = 2
        itp = scale(interpolate(ttime,BSpline(Cubic())),grd.x,grd.y)
        rstep = steplen * grd.hgrid
        thrdist = grd.hgrid*sqrt(2)

    elseif typeof(grd)==Grid3DCart
        Ndim = 3
        itp = scale(interpolate(ttime,BSpline(Cubic())),grd.x,grd.y,grd.z)
        rstep = steplen * grd.hgrid
        thrdist = grd.hgrid*sqrt(2)

    elseif typeof(grd)==Grid2DSphere
        Ndim = 2
        itp = scale(interpolate(ttime,BSpline(Cubic())),grd.r,grd.θ)
        rstep = steplen 
        thrdist = grd.Δr*sqrt(2)

    else
        error("tracerays_singlesrc(): ray tracing for spherical grids not yet implemented.")
    end

    Nrec = size(coordrec,1)
    rays = Vector{Matrix{Float64}}(undef,Nrec)


    Nseg::Int64 = 1e6
    raypath = zeros(Nseg,Ndim)
    
    for r=1:Nrec

        rec = coordrec[r,:]
        # distance for polar coord. has input in degrees
        dist2src = distance2points(grd,rec,srccoo) 
        raypath[1,:] .= rec

        s=1
        while dist2src >= thrdist
            
            #x,y = raypath[s,:]
            if Ndim == 2
                gradT = gradient(itp,raypath[s,1],raypath[s,2])
            elseif Ndim == 3
                gradT = gradient(itp,raypath[s,1],raypath[s,2],raypath[s,3])
            end

     
            if typeof(grd)==Grid2DCart ||  typeof(grd)==Grid3DCart 
                ## normalize the gradient vector to make the step-length more meaningful
                normgradT = gradT ./ sqrt(sum((gradT).^2))
                ## compute the new point
                newpt = raypath[s,:] .- rstep .* normgradT

            elseif typeof(grd)==Grid2DSphere
                rcur = raypath[s,1]
                # need radiants for the conversion
                θcur = deg2rad(raypath[s,2])

                # transform the gradient into Cartesian coordinates considering the
                #   unit vectors in polar coordinates
                dfdx = cos(θcur) * gradT[1] - 1/rcur * sin(θcur) * rad2deg(gradT[2]) # weird but it's rad2deg...
                dfdy = sin(θcur) * gradT[1] + 1/rcur * cos(θcur) * rad2deg(gradT[2])
                gradTcart = [dfdx, dfdy] 
                # normalise the gradient
                normgradTcart = gradTcart ./ sqrt(sum((gradTcart).^2))
                # convert to Cartesian for vector subtraction
                point_cart = polardeg2cartesian(raypath[s,1],raypath[s,2])
                ## compute the new pointn
                newpt_cart = point_cart .- rstep .* normgradTcart
                # convert back to polar coordinates
                newpt = collect(cartesian2polardeg(newpt_cart[1],newpt_cart[2]))
            end

      
            ## If the newpt is outside the domain...
            for d=1:Ndim
                if newpt[d] < grd.cooinit[d]
                    newpt[d] = grd.cooinit[d]
                elseif newpt[d] > grd.cooend[d]
                    newpt[d] = grd.cooend[d]
                end
            end
                        
            ## if the array is not big enough, re-allocate
            if s>=size(raypath,1)            
                raypath = [raypath; zeros(Nseg,Ndim)]
            end
            raypath[s+1,:] .= newpt
                  
            dist2src = distance2points(grd,raypath[s+1,:],srccoo)
            s += 1
        end
        ## last point, i.e., the source
        raypath[s,:] .= srccoo
         
        ## output array
        rays[r] = raypath[1:s,:]
    end

    return rays
end

##############################################################################

