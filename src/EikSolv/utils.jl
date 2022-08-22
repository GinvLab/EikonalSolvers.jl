
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

# function smoothgradatsource2D!(grad::AbstractArray,ijsrcs::Array{Int64,2} ;
#                              radiuspx::Integer=5)

#     isr = ijsrcs[1]
#     jsr = ijsrcs[2]
#     nx,ny = size(grad)

#     rmax = radiuspx
#     imin = isr-radiuspx
#     imax = isr+radiuspx
#     jmin = jsr-radiuspx
#     jmax = jsr+radiuspx

#     for j=jmin:jmax
#         for i=imin:imax
#             # deal with the borders
#             if i<1 || i>nx || j<1 || j>ny
#                 continue
#             else
#                 # inverse of geometrical spreading
#                 r = sqrt(float(i-isr)^2+float(j-jsr)^2)
#                 if r<=rmax
#                     # normalized inverse of geometrical spreading
#                     att = r/rmax
#                     grad[i,j] *= att
#                 end
#             end
#         end
#     end

#     return 
# end

#############################################



########################################################################

function smoothgradaroundsrc!(grad::AbstractArray,xsrc::Real,ysrc::Real,grd::Grid2D ;
                              radiuspx::Integer=5)

    isr,jsr = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid)
    nx,ny = grd.nx,grd.ny

    rmax = (radiuspx-1)*grd.hgrid
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
                xcur = grd.xinit + (i-1)*grd.hgrid
                ycur = grd.yinit + (j-1)*grd.hgrid
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

    return 
end

########################################################################
