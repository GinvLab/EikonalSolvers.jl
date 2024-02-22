

#############################################
using OffsetArrays

# Function to generate an N-dimensional kernel of uniform values that sum to 1
# The center of the hypercube will have indices (0, 0, ..., 0)
"""
Uniform kernel for smoothing
"""
function uniform_kernel(dim::Integer, n::Integer)
    @assert isodd(n)
    @assert n>0
    OffsetArray(fill(1/n^dim, fill(n, dim)...), fill(-n÷2-1, dim)...)
end

##########################################################

"""
Gaussian kernel for smoothing
"""
function gaussian_kernel(dim::Integer, l::Integer)
    @assert l>0
    @assert isodd(l)
    gker(s,r) = exp(-r^2/(2.0*s))
    # pos = LinRange(-(l-1)/2.0, (l-1)/2.0, l)
    ar = zeros([l for i=1:dim]...)
    s = (l/2)-sqrt(l/2)
    for I in CartesianIndices(ar)
        pos = [j-(size(ar,N)÷2+1) for (N,j) in enumerate(I.I)]
        r = sqrt(sum(pos.^2))
        ar[I] = gker(s,r)
    end
    ar /= sum(ar)
    return OffsetArray(ar,fill(-l÷2-1, dim)...) 
end

##########################################################

"""
    Smooth an array of any dimension
"""
function smoothimg(kernel,img)
    @assert typeof(kernel)<:OffsetArray
    @assert all( size(kernel).==size(kernel,1) )
    l = size(kernel,1)÷2+1
    filtered = zero(img)
    szimg = size(img)
    singleton = (szimg.==1)
    for I in CartesianIndices(img)
        idxs = I.I
        if all( idxs .>= l ) && all( idxs .<= (szimg.-l.+1) )
            for J in CartesianIndices(kernel)
                if I+J in CartesianIndices(img)
                    filtered[I] += img[I+J] * kernel[J]
                end
            end
        else
            # borders
            filtered[I] = img[I]
        end
    end
    return filtered
end

##########################################################
"""
    Smooth gradient for arrays of any dimensions.
"""
function smoothgradient(l,inpimg)
    
    sizinp = size(inpimg)
    singldim = findall(sizinp.==1)
    # drop dimension(s) if == 1
    img = dropdims(inpimg,dims=tuple(singldim...))

    dim = ndims(img)
    dimpad = size(img).+l
    mea = sum(img)/length(img)
    imgpadded = fill(mea,dimpad) #zeros(dimpad)
    # "internal" indices
    idxset = [l÷2+1:i for i in size(img).+l÷2]
    setindex!(imgpadded, img, idxset...  )
    kernel = gaussian_kernel(dim, l)
    out1d = smoothimg(kernel,imgpadded)


    out2d = reshape(out1d,dimpad)
    out = getindex(out2d,idxset...)

    # @show dim,l
    # @show size(inpimg)
    # @show size(kernel)
    # @show size(out)

    return out
end



##################################################

# function test()

#     gr = rand(27,37,12)
#     nx,ny,nz = size(gr)
#     gr[nx÷2,:,:] .= 1.3
#     gr2 = smoothgradient(5,gr)

#     fig  = Figure(resolution=(700,1200))
#     ax1,hm1 = heatmap(fig[1,1],gr[:,:,5])
#     Colorbar(fig[1,2],hm1)
#     ax2,hm2 = heatmap(fig[2,1],gr2[:,:,5])
#     Colorbar(fig[2,2],hm2)
#     display(fig)

#     return gr,gr2
# end

########################################################################

function smoothgradaroundsrc2D!(grad::AbstractArray,xysrc::AbstractVector{<:Real},
                                grd::Union{Grid2DCart,Grid2DSphere} ; radiuspx::Integer)

    ## no smoothing
    if radiuspx==0
        return
    elseif radiuspx<0
        error("smoothgradaroundsrc!(): 'radius'<0 ")
    end
    
    if typeof(grd)==Grid2DCart
        # Cartesian

        xsrc,ysrc = xysrc[1],xysrc[2]
        isr,jsr = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid)
        nx,ny = grd.nx,grd.ny


        rmax = radiuspx*grd.hgrid
        imin = isr-radiuspx
        imax = isr+radiuspx
        jmin = jsr-radiuspx
        jmax = jsr+radiuspx

        for j=jmin:jmax
            for i=imin:imax
                # deal with the borders
                if i<1 || i>nx || j<1 || j>ny
                    continue

                else
                    xcur = grd.x[i]
                    ycur = grd.y[j]
                    # inverse of geometrical spreading
                    r = sqrt((xcur-xsrc)^2+(ysrc-ycur)^2)
                    if r<=rmax
                        # normalized inverse of geometrical spreading
                        att = r/rmax
                        grad[i,j] *= att
                    end
                end
            end
        end

    elseif typeof(grd)==Grid2DSphere
        # Spherical

        rsrc,θsrc = xysrc[1],xysrc[2]
        isr,jsr = findclosestnode_sph(rsrc,θsrc,grd.rinit,grd.θinit,
                                      grd.Δr,grd.Δθ)
        nr,nθ = grd.nr,grd.nθ
        xsrc,ysrc = sphericaldeg2cartesian(rsrc,θsrc)


        ## for simplicity rmax is defined in terms of the radius only
        rmax = radiuspx*grd.Δr
        imin = isr-radiuspx
        imax = isr+radiuspx
        jmin = jsr-radiuspx
        jmax = jsr+radiuspx

        for j=jmin:jmax
            for i=imin:imax
                # deal with the borders
                if i<1 || i>nr || j<1 || j>nθ
                    continue

                else
                    xcur,ycur = sphericaldeg2cartesian(grd.r[i],grd.θ[j])                    
                    # inverse of geometrical spreading
                    ra = sqrt((xcur-xsrc)^2+(ysrc-ycur)^2)
                    if ra<=rmax
                        # normalized inverse of geometrical spreading
                        att = ra/rmax
                        grad[i,j] *= att
                    end
                end
            end
        end

    else
        error("smoothgradaroundsrc2D!(): Wrong grid type.")

    end # if
    
    return 
end

################################################################3

function smoothgradaroundsrc3D!(grad::AbstractArray,xyzsrc::AbstractVector{<:Real},
                                grd::Union{Grid3DCart,Grid3DSphere} ;
                                radiuspx::Integer)

    ## no smoothing
    if radiuspx==0
        return
    elseif radiuspx<0
        error("smoothgradaroundsrc!(): 'radius'<0 ")
    end

    if typeof(grd)==Grid3D
        # Cartesian

        xsrc,ysrc,zsrc = xyzsrc[1],xyzsrc[2],xyzsrc[3]
        isr,jsr,ksr = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid)
        nx,ny,nz = grd.nx,grd.ny,grd.nz

        rmax = radiuspx*grd.hgrid
        imin = isr-radiuspx
        imax = isr+radiuspx
        jmin = jsr-radiuspx
        jmax = jsr+radiuspx
        kmin = ksr-radiuspx
        kmax = ksr+radiuspx

        for k=kmin:kmax
            for j=jmin:jmax
                for i=imin:imax
                    # deal with the borders
                    if i<1 || i>nx || j<1 || j>ny || k<1 || k>nz
                        continue

                    else
                        xcur = grd.x[i]
                        ycur = grd.y[j]
                        zcur = grd.z[k]
                        # inverse of geometrical spreading
                        r = sqrt((xcur-xsrc)^2+(ysrc-ycur)^2+(zsrc-zcur)^2)
                        if r<=rmax
                            # normalized inverse of geometrical spreading
                            att = r/rmax
                            grad[i,j,k] *= att
                        end
                    end
                end
            end
        end

        
    elseif typeof(grd)==Grid3DSphere
        # Spherical
        rsrc,θsrc,φsrc = xyzsrc[1],xyzsrc[2],xyzsrc[3]
        isr,jsr,ksr = findclosestnode_sph(rsrc,θsrc,φsrc,grd.rinit,grd.θinit,grd.φinit,
                                          grd.Δr,grd.Δθ,grd.Δφ)
        nr,nθ,nφ = grd.nr,grd.nθ,grd.nφ
        xsrc,ysrc,zsrc = sphericaldeg2cartesian(rsrc,θsrc,φsrc)


        ## for simplicity rmax is defined in terms of the radius only
        rmax = radiuspx*grd.Δr
        imin = isr-radiuspx
        imax = isr+radiuspx
        jmin = jsr-radiuspx
        jmax = jsr+radiuspx
        kmin = ksr-radiuspx
        kmax = ksr+radiuspx

        for k=kmin:kmax
            for j=jmin:jmax
                for i=imin:imax
                    # deal with the borders
                    if i<1 || i>nr || j<1 || j>nθ || k<1 || k>nφ
                        continue

                    else
                        xcur,ycur,zcur = sphericaldeg2cartesian(grd.r[i],grd.θ[j],grd.φ[k])
                        # inverse of geometrical spreading
                        ra = sqrt((xcur-xsrc)^2+(ysrc-ycur)^2+(zsrc-zcur)^2)
                        if ra<=rmax
                            # normalized inverse of geometrical spreading
                            att = ra/rmax
                            grad[i,j,k] *= att
                        end
                    end
                end
            end
        end

    else
        error("smoothgradaroundsrc3D!(): Wrong grid type.")

    end # if

    return 
end



########################################################################

@inline function sphericaldeg2cartesian(r,θ,φ)
    x = r * sind(θ) * cosd(φ) 
    y = r * sind(θ) * sind(φ)
    z = r * cosd(θ)
    return x,y,z
end

@inline function sphericaldeg2cartesian(r,θ)
    x = r * cosd(θ) 
    y = r * sind(θ) 
    return x,y
end

########################################################################


########################################################################

# function raytrace2doneshot(ttime::Array{Float64,2},grd::Grid2D,coordrec::Array{Float64,2},
#                            coordsrc::Vector{Float64})

#     nx,ny = grd.nx,grd.ny
#     nrec = size(coordrec,1)
#     HUGE = 1e30
#     tmprayij = MVector(0,0)

#     # indices for exploring neighboring points
#     neigh = SA[1  0;
#                0  1;
#               -1  0;
#                0 -1;
#                1  1;
#               -1  1;
#                1 -1;
#               -1 -1]

#     raycoo = zeros(nx*ny,2)
#     rays = Vector{Array{Float64,2}}(undef,nrec)
#     tmpi,tmpj = 0,0
#     nelemray = 0

#     for r=1:nrec

#         xrec,yrec = coordrec[r,1],coordrec[r,2]
#         raycoo[:,:] .= 0.0
        
#         # initial coordinates, i.e., receiver location
#         raycoo[1,:] .= (xrec,yrec)

#         # find the closest node to the source point
#         icur,jcur = findclosestnode(xrec,yrec,grd.xinit,grd.yinit,grd.hgrid)

#         @show icur,jcur
#         ## single ray
#         smallesttt = ttime[icur,jcur]
#         l=2
#         notonsource = true
#         while notonsource


#             display(ttime[icur-1:icur+1,jcur-1:jcur+1])

#             foundsmallertt = false
#             for ne=1:8 ## eight neighbors
#                 i = icur + neigh[ne,1]
#                 j = jcur + neigh[ne,2]
#                 ## if the point is out of bounds skip this iteration
#                 if (i>nx) || (i<1) || (j>ny) || (j<1)
#                     continue
#                 end

#                 if ttime[i,j]<smallesttt
#                     smallesttt = ttime[i,j]
#                     tmprayij[:] .= (i,j)
#                     foundsmallertt = true
#                     #@show i,j,smallesttt
#                 end
#             end

#             if foundsmallertt
#                 #@show l,tmprayij
#                 icur,jcur = tmprayij[1],tmprayij[2]
#                 raycoo[l,:] .= (grd.hgrid*(tmprayij[1]-1)+grd.xinit,grd.hgrid*(tmprayij[2]-1)+grd.yinit)
#                 l+=1

#                 #@show icur,jcur,smallesttt
#             else
#                 raycoo[l,:] .= coordsrc[:]
#                 nelemray = l
#                 notonsource = false
#             end

#         end # while

#         #@show nelemray
#         rays[r] = raycoo[1:nelemray,:]

#     end # r=1:nrec

#     return rays
# end
