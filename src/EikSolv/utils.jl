

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
    if iseven(l)
        error("gaussian_kernel(): l must be odd")
    end
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
    # convert l to Int otherwise it underflows
    return OffsetArray(ar,fill(-Int(l)÷2-1, dim)...) 
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

function smoothgradaroundsrc!(grad::AbstractArray,xysrc::AbstractVector{<:Real},
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
        ijsrccorn = findenclosingbox(grd,xysrc)
        nx,ny = grd.grsize

        rmax = radiuspx*grd.hgrid
        imin = ijsrccorn[1,1]-radiuspx
        imax = ijsrccorn[2,1]+radiuspx
        jmin = ijsrccorn[1,2]-radiuspx
        jmax = ijsrccorn[3,2]+radiuspx

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
        ijsrccorn = findenclosingbox(grd,xysrc)
       
        nr,nθ = grd.nr,grd.nθ
        xsrc,ysrc = polardeg2cartesian(rsrc,θsrc)

        ## for simplicity rmax is defined in terms of the radius only
        rmax = radiuspx*grd.Δr
        imin = ijsrccorn[1,1]-radiuspx
        imax = ijsrccorn[2,1]+radiuspx
        jmin = ijsrccorn[1,2]-radiuspx
        jmax = ijsrccorn[3,2]+radiuspx


        for j=jmin:jmax
            for i=imin:imax
                # deal with the borders
                if i<1 || i>nr || j<1 || j>nθ
                    continue

                else
                    xcur,ycur = polardeg2cartesian(grd.r[i],grd.θ[j])                    
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
        error("smoothgradaroundsrc!(): Wrong grid type.")

    end # if
    
    return 
end

################################################################3

function smoothgradaroundsrc!(grad::AbstractArray,xyzsrc::AbstractVector{<:Real},
                              grd::Union{Grid3DCart,Grid3DSphere} ;
                              radiuspx::Integer)

    ## no smoothing
    if radiuspx==0
        return
    elseif radiuspx<0
        error("smoothgradaroundsrc!(): 'radius'<0 ")
    end

    if typeof(grd)==Grid3DCart
        # Cartesian

        xsrc,ysrc,zsrc = xyzsrc[1],xyzsrc[2],xyzsrc[3]
        ijksrccorn = findenclosingbox(grd,xyzsrc)
        nx,ny,nz = grd.grsize

        rmax = radiuspx*grd.hgrid
        imin = ijksrccorn[1,1]-radiuspx
        imax = ijksrccorn[2,1]+radiuspx
        jmin = ijksrccorn[1,2]-radiuspx
        jmax = ijksrccorn[3,2]+radiuspx
        kmin = ijksrccorn[1,1]-radiuspx
        kmax = ijksrccorn[4,1]+radiuspx

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
        ijksrccorn = findenclosingbox(grd,xysrc)
     
        nr,nθ,nφ = grd.nr,grd.nθ,grd.nφ
        xsrc,ysrc,zsrc = sphericaldeg2cartesian(rsrc,θsrc,φsrc)

        ## for simplicity rmax is defined in terms of the radius only
        rmax = radiuspx*grd.Δr
        imin = ijksrccorn[1,1]-radiuspx
        imax = ijksrccorn[2,1]+radiuspx
        jmin = ijksrccorn[1,2]-radiuspx
        jmax = ijksrccorn[3,2]+radiuspx
        kmin = ijksrccorn[1,1]-radiuspx
        kmax = ijksrccorn[4,1]+radiuspx

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
        error("smoothgradaroundsrc!(): Wrong grid type.")

    end # if

    return 
end



########################################################################

@inline function sphericaldeg2cartesian(r::Real,θ::Real,φ::Real)
    x = r * sind(θ) * cosd(φ) 
    y = r * sind(θ) * sind(φ)
    z = r * cosd(θ)
    return x,y,z
end

@inline function polardeg2cartesian(r::Real,θ::Real)
    x = r * cosd(θ) 
    y = r * sind(θ) # input in degrees
    return x,y
end

@inline function polarrad2cartesian(r::Real,θ::Real)
    x = r * cos(θ) 
    y = r * sin(θ) # input in radiants
    return x,y
end

@inline function cartesian2polardeg(x::Real,y::Real)
    r = sqrt(x^2+y^2)
    θ = atand(y,x) # output in degrees
    return r,θ
end

@inline function cartesian2polarrad(x::Real,y::Real)
    r = sqrt(x^2+y^2)
    θ = atan(y,x) # output in radiants
    return r,θ
end

########################################################################

