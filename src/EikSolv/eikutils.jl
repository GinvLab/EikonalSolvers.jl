

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

###################################################################

"""
$(TYPEDSIGNATURES)

Calculate the Gaussian misfit functional 
```math
    S = \\dfrac{1}{2} \\sum_i \\dfrac{\\left( \\mathbf{u}_i^{\\rm{calc}}(\\mathbf{v})-\\mathbf{u}_i^{\\rm{obs}} \\right)^2}{\\sigma_i^2} \\, .
```

# Arguments
- `velmod`: velocity model, either a 2D or 3D array.
- `ttpicksobs`: a vector of vectors of the traveltimes at the receivers.
- `stdobs`: a vector of vectors standard deviations representing the error on the measured traveltimes.
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
- `grd`: the struct holding the information about the grid, one of `Grid2D`,`Grid3D`,`Grid2Dsphere`,`Grid3Dsphere`
- `extraparams` (optional): a struct containing some "extra" parameters, namely
    * `parallelkind`: serial, Threads or Distributed run? (:serial, :sharedmem, :distribmem)
    * `refinearoundsrc`: whether to perform a refinement of the grid around the source location
    * `grdrefpars`: refined grid around the source parameters (`downscalefactor` and `noderadius`)
    * `allowfixsqarg`: brute-force fix negative saqarg. Don't use this!
    * `manualGCtrigger`: trigger garbage collector (GC) manually at selected points.

# Returns
The value of the misfit functional (L2-norm), the same used to compute the gradient with adjoint methods.

"""
function eikttimemisfit(velmod::Array{Float64,N},ttpicksobs::AbstractArray,
                        stdobs::AbstractArray,coordsrc::AbstractArray,
                        coordrec,grd::AbstractGridEik ;
                        extraparams::Union{ExtraParams,Nothing}=nothing)::Float64 where N
    
    if extraparams==nothing
        extraparams = ExtraParams()
    end

    # compute the forward response
    ttpicks = eiktraveltime(velmod,grd,coordsrc,coordrec,extraparams=extraparams)

    nsrc = size(coordsrc,1)
    ##nrecs1 = size.(coordrec,1)
    ##totlen = sum(nrec1)
    misf::Float64 = 0.0
    for s=1:nsrc
        misf += sum( (ttpicks[s].-ttpicksobs[s]).^2 ./ stdobs[s].^2)
    end
    misf *= 0.5

    return misf
end

##############################################################################


"""

$(TYPEDSIGNATURES)

Ray tracing utility. Given a traveltime grid and source and receiver positions, trace the rays.
"""
function tracerays_singlesrc(grd::AbstractGridEik,ttime::Array{Float64,N},
                             srccoo::AbstractVector,coordrec::Array{Float64,2};
                             steplen::Real=0.1) where N

    # setup interpolation
    if typeof(grd)==Grid2DCart
        Ndim = 2
        itp = scale(interpolate(ttime,BSpline(Cubic())),grd.x,grd.y)
    elseif typeof(grd)==Grid3DCart
        Ndim = 3
        itp = scale(interpolate(ttime,BSpline(Cubic())),grd.x,grd.y,grd.z)
    end

    Nrec = size(coordrec,1)
    rays = Vector{Matrix{Float64}}(undef,Nrec)

    thrdist = grd.hgrid*sqrt(2)
    rstep = steplen * grd.hgrid

    Nseg::Int64 = 1e6
    raypath = zeros(Nseg,Ndim)
    
    for r=1:Nrec

        rec = coordrec[r,:]
        raypath[1,:] .= rec
        dist2src = sqrt(sum((raypath[1,:] .- srccoo).^2))

        s=1
        while dist2src >= thrdist
            
            x,y = raypath[s,:]
            if Ndim == 2
                gradT = gradient(itp,raypath[s,1],raypath[s,2])
            elseif Ndim == 3
                gradT = gradient(itp,raypath[s,1],raypath[s,2],raypath[s,3])
            end
            ## normalize the gradient vector to make the step-length more meaningful
            normgradT = gradT ./ sqrt(sum((gradT).^2))
            ## compute the new point
            newpt = raypath[s,:] .- rstep .* normgradT

            # if any(grd.xyxinit .> newpt) ||  any(grd.xyzend .< newpt)
            #     error("Point outside the domain")
            # end
            
            raypath[s+1,:] = newpt
            
            dist2src = sqrt(sum((raypath[s+1,:] .- srccoo).^2))

            s += 1
            ## if the array is not big enough, re-allocate
            if s>size(raypath,1)            
                raypath = [raypath; zeros(Nseg)]
            end
        end
        ## last point, i.e., the source
        raypath[s,:] .= srccoo
        ## output array
        rays[r] = raypath[1:s,:]
    end

    return rays
end

##############################################################################
