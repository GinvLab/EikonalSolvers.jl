

#######################################################
##     Misfit of gradient using adjoint              ## 
#######################################################


###############################################################################
########################################################################################
## @doc raw because of some backslashes in the string...
@doc raw""" 
    gradttime3Dsphere(vel::Array{Float64,3},grd::Grid3DSphere,coordsrc::Array{Float64,2},
                coordrec::Array{Float64,2},pickobs::Array{Float64,2},
                pickcalc::Array{Float64,2}, stdobs::Array{Float64,2} )                

Calculate the gradient using the adjoint state method for 3D velocity models in spherical coordinates. 
Returns the gradient of the misfit function with respect to velocity calculated at the given point (velocity model).
The gradient is calculated using the adjoint state method. 
The computations are run in parallel depending on the number of workers (nworkers()) available.
The algorithm used is "gradFMM\_hiord", a second order fast marching method for forward, high order fast marching method for adjoint.  

# Arguments
- `vel`: the 3D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y,z), a 3-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y,z), a 3-column array 
- `pickobs`: observed traveltime picks
- `stdobs`: standard deviation of error on observed traveltime picks, an array with same shape than `pickobs`

# Returns
- `grad`: the gradient as a 3D array

"""
function gradttime3Dsphere(vel::Array{Float64,3},grd::Grid3DSphere,coordsrc::Array{Float64,2},coordrec::Array{Float64,2},
                           pickobs::Array{Float64,2},stdobs::Array{Float64,2})
   
    @assert size(coordsrc,2)==3
    @assert size(coordrec,2)==3
    @assert all(vel.>0.0)
    @assert all(grd.rinit.<=coordsrc[:,1].<=((grd.nr-1)*grd.Δr+grd.rinit))
    @assert all(grd.θinit.<=coordsrc[:,2].<=((grd.nθ-1)*grd.Δθ+grd.θinit))
    @assert all(grd.φinit.<=coordsrc[:,3].<=((grd.nφ-1)*grd.Δφ+grd.φinit))

    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(grd.nr,grd.nθ,grd.nφ,nchu)
    ## do the calculations
    @sync for s=1:nchu
        igrs = grpsrc[s,1]:grpsrc[s,2]
        @async tmpgrad[:,:,:,s] = remotecall_fetch(calcgradsomesrc3D,wks[s],vel,
                                                   coordsrc[igrs,:],coordrec,
                                                   grd,stdobs[:,igrs],pickobs[:,igrs] )
    end
    grad = sum(tmpgrad,dims=4)
    return grad
end

################################################################3

"""
Calculate the gradient for some requested sources 
"""
function calcgradsomesrc3D(vel::Array{Float64,3},xθsrc::Array{Float64,2},coordrec::Array{Float64,2},
                           grd::Grid3DSphere,stdobs::Array{Float64,2},pickobs1::Array{Float64,2} )

    nr,nθ,nφ=size(vel)
    ttpicks1 = zeros(size(coordrec,1))
    nsrc = size(xθsrc,1)
    grad1 = zeros(nr,nθ,nφ)


    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        ###########################################
        ## calc ttime
 
        ## if adjalgo=="gradFMM_hiord"
        ttonesrc = ttFMM_hiord(vel,xθsrc[s,:],grd)

        ## ttime at receivers
        @inbounds for i=1:size(coordrec,1)
            ttpicks1[i] = trilinear_interp_sph( ttonesrc,grd,coordrec[i,1],
                                                coordrec[i,2],coordrec[i,3] )
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        ## if adjalgo=="gradFMM_hiord"
        grad1 += eikgrad_FMM_hiord_SINGLESRC(ttonesrc,vel,xθsrc[s,:],coordrec,
                                             grd,pickobs1[:,s],ttpicks1,stdobs[:,s])
    end
    
    return grad1
end

############################################################################

function sourceboxlocgrad_sph!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Vector{Float64},
                               grd::Grid3DSphere )

    ##########################
    ##   Init source
    ##########################
    mindistsrc = 1e-5
    rsrc,θsrc,φsrc = srcpos[1],srcpos[2],srcpos[3]

    ## regular grid
    onsrc = zeros(Bool,grd.nr,grd.nθ,grd.nφ)
    onsrc[:,:,:] .= false
    ir,iθ,iφ = findclosestnode_sph(rsrc,θsrc,φsrc,grd.rinit,grd.θinit,grd.φinit,grd.Δr,grd.Δθ,grd.Δφ)
    rr = rsrc-grd.r[ir] #rsrc-((ir-1)*grd.Δr+grd.rinit)
    rθ = θsrc-grd.θ[iθ] #θsrc-((iθ-1)*grd.Δθ+grd.θinit)
    rφ = φsrc-grd.φ[iφ] #φsrc-((iφ-1)*grd.Δφ+grd.φinit)
    max_r = grd.r[end] #(grd.nr-1)*grd.Δr+grd.rinit
    max_θ = grd.θ[end] #(grd.nθ-1)*grd.Δθ+grd.θinit
    max_φ = grd.φ[end] #(grd.nφ-1)*grd.Δφ+grd.φinit
    
    halfg = 0.0
    src_on_nodeedge = false

    ## loop to make sure we don't accidentally move the src to another node/edge
    ## distance in POLAR coordinates
    ## sqrt(r1^2 +r2^2 -2*r1*r2*(sin(θ1)*sin(θ2)*cos(φ1-φ2)+cos(θ1)*cos(θ2)))
    r1 = rsrc
    r2 = grd.r[ir]
    θ1 = θsrc
    θ2 = grd.θ[iθ]
    φ1 = φsrc
    φ2 = grd.φ[iφ]
    dist = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))
    while  (dist<=mindistsrc) || (abs(rr)<=mindistsrc) || (abs(rθ)<=mindistsrc) || (abs(rφ)<=mindistsrc)
        src_on_nodeedge = true
        
        ## shift the source 
        if rsrc < max_r-0.002*grd.Δr
            rsrc = rsrc+0.001*grd.Δr
        else #(make sure it's not already at the bottom y)
            rsrc = rsrc-0.001*grd.Δr
        end
        if θsrc < max_θ-0.002*grd.Δθ
            θsrc = θsrc+0.001*grd.Δθ
        else #(make sure it's not already at the bottom y)
            θsrc = θsrc-0.001*grd.Δθ
        end
        if φsrc < max_φ-0.002*grd.Δφ
            φsrc = φsrc+0.001*grd.Δφ
        else #(make sure it's not already at the bottom z)
            φsrc = φsrc-0.001*grd.Δφ
        end

        # print("new time at src:  $(tt[ir,iθ]) ")
        ## recompute parameters related to position of source
        ## regular grid
        ir,iθ,iφ = findclosestnode_sph(rsrc,θsrc,φsrc,grd.rinit,grd.θinit,grd.φinit,grd.Δr,grd.Δθ,grd.Δφ)
        rr = rsrc-grd.r[ir] #((ir-1)*grd.Δr+grd.rinit)
        rθ = θsrc-grd.θ[iθ] #((iθ-1)*grd.Δθ+grd.θinit)
        rφ = φsrc-grd.φ[iφ] #((iφ-1)*grd.Δφ+grd.φinit)
        
        r1 = rsrc
        r2 = grd.r[ir]
        θ1 = θsrc
        θ2 = grd.θ[iθ]
        φ1 = φsrc
        φ2 = grd.φ[iφ]
        dist = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))       
    end

    ## To avoid singularities, the src can only be inside a box,
    ##  not on a grid node or edge... see above. So we need to have
    ##  always 4 nodes where onsrc is true
    if (rr>=halfg) & (rθ>=halfg) & (rφ>=halfg)
        onsrc[ir:ir+1,iθ:iθ+1,iφ:iφ+1] .= true
    elseif (rr<halfg) & (rθ>=halfg) & (rφ>=halfg)
        onsrc[ir-1:ir,iθ:iθ+1,iφ:iφ+1] .= true
    elseif (rr<halfg) & (rθ<halfg) & (rφ>=halfg)
        onsrc[ir-1:ir,iθ-1:iθ,iφ:iφ+1] .= true
    elseif (rr>=halfg) & (rθ<halfg) & (rφ>=halfg)
        onsrc[ir:ir+1,iθ-1:iθ,iφ:iφ+1] .= true

    elseif (rr>=halfg) & (rθ>=halfg) & (rφ<halfg)
        onsrc[ir:ir+1,iθ:iθ+1,iφ-1:iφ] .= true
    elseif (rr<halfg) & (rθ>=halfg) & (rφ<halfg)
        onsrc[ir-1:ir,iθ:iθ+1,iφ-1:iφ] .= true
    elseif (rr<halfg) & (rθ<halfg) & (rφ<halfg)
        onsrc[ir-1:ir,iθ-1:iθ,iφ-1:iφ] .= true
    elseif (rr>=halfg) & (rθ<halfg) & (rφ<halfg)
        onsrc[ir:ir+1,iθ-1:iθ,iφ-1:iφ] .= true
    end


    ## if src on node or edge, recalculate traveltimes in box
    if src_on_nodeedge==true
        ## RE-set a new ttime around source 
        #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
        ijksrc = findall(onsrc)
        # println(" set time around src $isrc  $jsrc $ksrc") 
        #for (k,j,i) in zip(ksrc,jsrc,isrc)
        for lcart in ijksrc
            # for j in jsrc
            #     for i in isrc
            i = lcart[1]
            j = lcart[2]
            k = lcart[3]
            ## regular grid
            # rp = (i-1)*grd.hgrid+grd.rinit
            # θp = (j-1)*grd.hgrid+grd.θinit
            # φp = (k-1)*grd.hgrid+grd.φinit
            # ii = Int(floor((rsrc-grd.rinit)/grd.Δr) +1)
            # jj = Int(floor((θsrc-grd.θinit)/grd.Δθ) +1)
            # kk = Int(floor((φsrc-grd.φinit)/grd.Δφ) +1)             

            r1 = rsrc
            r2 = grd.r[i]
            θ1 = θsrc
            θ2 = grd.θ[j]
            φ1 = φsrc
            φ2 = grd.φ[k]
            distp = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))
            ttime[i,j,k] = distp / vel[i,j,k]ttime[i,j,k] 
        end
    end

    return onsrc,rsrc,θsrc,φsrc
end

############################################################################

function recboxlocgrad_sph!(ttime::Array{Float64,3},lambda::Array{Float64,3},ttpicks::Vector{Float64},grd::Grid3DSphere,
                        rec::Array{Float64,2},pickobs::Vector{Float64},stdobs::Vector{Float64} )

    ##########################
    ## Init receivers
    ##########################

    ## regular grid
    onarec = zeros(Bool,grd.nr,grd.nθ,grd.nφ)
    nrmax = grd.nr
    nθmax = grd.nθ
    nφmax = grd.nφ

    onarec[:,:,:] .= false
    nrec=size(rec,1)

    deltar = grd.Δr
    deltaθ = deg2rad(grd.Δθ)  ## DEG to RAD !!!!
    deltaφ = deg2rad(grd.Δφ)  ## DEG to RAD !!!!

    ## init receivers
    nrec=size(rec,1)
    for r=1:nrec
        i,j,k = findclosestnode_sph(rec[r,1],rec[r,2],rec[r,3],grd.rinit,grd.θinit,grd.φinit,grd.Δr,grd.Δθ,grd.Δφ)

        if (i==1) || (i==nrmax) || (j==1) || (j==nθmax) || (k==1) || (k==nφmax)
            ## Receivers on the boundary of model
            ## (n⋅∇T)λ = Tcalc - Tobs
            if i==1
                # forward diff
                n∇T = - (ttime[2,j,k]-ttime[1,j,k])/deltar 
            elseif i==nrmax
                # backward diff
                n∇T = (ttime[end,j,k]-ttime[end-1,j,k])/deltar 
            elseif j==1
                # forward diff
                n∇T = - (ttime[i,2,k]-ttime[i,1,k])/(grd.r[i]*deltaθ)
            elseif j==nθmax
                # backward diff
                n∇T = (ttime[i,end,k]-ttime[i,end-1,k])/(grd.r[i]*deltaθ) 
            elseif k==1
                # forward diff
                n∇T = - (ttime[i,j,2]-ttime[i,j,1])/(grd.r[i]*sind(grd.θ[j])*deltaφ)  
            elseif k==nφmax
                # backward diff
                n∇T = (ttime[i,j,end]-ttime[i,j,end-1])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
            end
            onarec[i,j,k] = true
            lambda[i,j,k] = (ttpicks[r]-pickobs[r])/(n∇T * stdobs[r]^2)
            # println(" Receiver on border of model (i==1)||(i==ntx)||(j==1)||(j==nty || (k==1) || (k==ntz))")
            # println(" Not yet implemented...")
            # return nothing
        else
            ## Receivers within the model
            onarec[i,j,k] = true
            lambda[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs[r]^2
        end
    end
  
    return onarec
end

###################################################################################################

function adjderivonsource_sph(tt::Array{Float64,3},onsrc::Array{Bool,3},i::Int64,j::Int64,k::Int64,
                              grd::Grid3DSphere,rsrc::Float64,θsrc::Float64,φsrc::Float64)

    ##          thx
    ##     o-----.------> x
    ##     |\    |
    ##     | \d  | HPy
    ##     |  \  |
    ##     |   \ |
    ##     |    \|
    ## thy .-----.P
    ##     |HPx 
    ##     |
    ##     \/
    ##     y
    ##
    ##                 
    ##      thx  .....x.........
    ##        ...     |         ...
    ##     o..        |            ..o
    ##     \         .P             /
    ##  thy \    .... |            / 
    ##       \x..     |           /
    ##        \       |          /
    ##         \    ..|.....    / 
    ##          o...        ...o
    ## 
    ##

    ## If we are in the square containing the source,
    ## use real position of source for the derivatives.
    ## Calculate tt on x and y on the edges of the square
    ##  H is the projection of the point src onto X, Y and Z edges
    xp = grd.r[i] #Float64(i-1)*dh+xinit
    yp = grd.θ[j] #Float64(j-1)*dh+yinit
    zp = grd.φ[k] #Float64(k-1)*dh+zinit

    # distances to the sides

    ## distance to the side "x", along radius
    distHPx = abs(xp-rsrc) ## abs needed for later...

    ## distance to the side "y", along θ
    r1 = rsrc
    r2 = rsrc
    θ1 = θsrc
    θ2 = grd.θ[k]
    φ1 = φsrc
    φ2 = φsrc
    distHPy = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))

    ## distance to the side "z", along φ
    r1 = rsrc
    r2 = rsrc
    θ1 = θsrc
    θ2 = θsrc
    φ1 = φsrc
    φ2 = grd.φ[k]
    distHPz = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))

    # distance from current point to source
    r1 = rsrc
    r2 = grd.r[i]
    θ1 = θsrc
    θ2 = grd.θ[j]
    φ1 = φsrc
    φ2 = grd.φ[k]
    dist2src = sqrt(r1^2+r2^2-2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))

    # assert dist2src>0.0 otherwise a singularity will occur
    @assert dist2src>0.0

    ##>>>>> CHECK the stuff below!!! <<<<<<<<<<<<##
    
    deltar = grd.Δr
    deltaθ = deg2rad(grd.Δθ)  ## DEG to RAD !!!!
    deltaφ = deg2rad(grd.Δφ)  ## DEG to RAD !!!!
    arcHPy = abs(rsrc*deg2rad(yp-θsrc)) ## arc distance 
    arcHPz = abs(rsrc*sind(θsrc)*deg2rad(zp-φsrc)) ## arc distance 
    
    ## Calculate the traveltime to hit the x edge
    ## time at H along x
    thx = tt[i,j,k]*distHPx/dist2src
    ## Calculate the traveltime to hit the y edge
    ## time at H along y
    thy = tt[i,j,k]*distHPy/dist2src
    ## Calculate the traveltime to hit the z edge
    ## time at H along z
    thz = tt[i,j,k]*distHPz/dist2src
    
    if onsrc[i+1,j,k]==true              
        aback = -(tt[i,j,k]-tt[i-1,j,k])/deltar
        if distHPx==0.0
            aforw = 0.0 # point exactly on source
        else
            aforw = -(thx-tt[i,j,k])/distHPx # dist along r
        end
    elseif onsrc[i-1,j,k]==true
        if distHPx==0.0
            aback = 0.0 # point exactly on source
        else
            aback = -(tt[i,j,k]-thx)/distHPx # dist along r
        end
        aforw = -(tt[i+1,j,k]-tt[i,j,k])/deltar
    end

    if onsrc[i,j+1,k]==true
        bback = -(tt[i,j,k]-tt[i,j-1,k])/(grd.r[i]*deltaθ)
        if distHPy==0.0
            bforw = 0.0 # point exactly on source
        else
            bforw = -(thy-tt[i,j,k])/arcHPy # dist along θ arc
        end
    else onsrc[i,j-1,k]==true
        if distHPy==0.0
            bback = 0.0 # point exactly on source
        else        
            bback = -(tt[i,j,k]-thy)/arcHPy # dist along θ arc
        end
        bforw = -(tt[i,j+1,k]-tt[i,j,k])/(grd.r[i]*deltaθ)
    end
    
    if onsrc[i,j,k+1]==true     
        cback = -(tt[i,j,k]-tt[i,j,k-1])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
        if distHPz==0.0
            cforw = 0.0 # point exactly on source
        else        
            cforw = -(thz-tt[i,j,k])/arcHPz # dist along φ arc
        end
    elseif onsrc[i,j,k-1]==true
        if distHPz==0.0
            cback = 0.0 # point exactly on source
        else        
            cback = -(tt[i,j,k]-thz)/arcHPz # dist along φ arc
        end
        cforw = -(tt[i,j,k+1]-tt[i,j,k])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
    end

    return  aback,aforw,bback,bforw,cback,cforw
end

#############################################################################

function eikgrad_FMM_hiord_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                                     src::Vector{Float64},rec::Array{Float64,2},
                                     grd::Grid3DSphere,pickobs::Vector{Float64},
                                     ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(3,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel)) ## SAME SIZE!!
    nr,nθ,nφ = grd.nr,grd.nθ,grd.nφ  ##size(ttime)

    lambda = zeros((grd.nr,grd.nθ,grd.nφ)) 
 
    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    #dh = grd.hgrid
    # ntx = nr ##grd.ntx
    # nty = nθ ##grd.nty
    # ntz = nφ ##grd.nty
    
    ## Grid position
    rinit = grd.rinit
    θinit = grd.θinit
    φinit = grd.φinit
    
    ## source
    ## regular grid
    onsrc,rsrc,θsrc,φsrc = sourceboxlocgrad_sph!(tt,vel,src,grd)
    ## receivers
    onarec = recboxlocgrad_sph!(tt,lambda,ttpicks,grd,rec,pickobs,stdobs)

    ######################################################
    #-------------------------------
    ## init FMM 
    neigh = [1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]
 
    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,nr,nθ,nφ)
    status[:,:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :, : ] .= 2 ## set to accepted on boundary
    status[nr,:, : ] .= 2 ## set to accepted on boundary
    status[:, 1, : ] .= 2 ## set to accepted on boundary
    status[:, nθ,: ] .= 2 ## set to accepted on boundary
    status[:, :, 1 ] .= 2 ## set to accepted on boundary
    status[:, :, nφ] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nr,nθ),find(status.==2))
    ijkrec = findall(status.==2)
    irec = [x[1] for x in ijkrec]
    jrec = [x[2] for x in ijkrec]
    krec = [x[3] for x in ijkrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=nr*nθ*nφ
 
    bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## conversion cart to lin indices, old sub2ind
    linid_nrnθnφ = LinearIndices((nr,nθ,nφ))
    ## conversion lin to cart indices, old sub2ind
    cartid_nrnθnφ = CartesianIndices((nr,nθ,nφ))

    ## construct initial narrow band
    @inbounds for l=1:naccinit 
        @inbounds for ne=1:6 ## six potential neighbors
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            k = jrec[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nr-1) || (i<2) || (j>nθ-1) || (j<2) || (k>nφ-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far

                # get handle
                #han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθnφ[i,j,k]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = nr*nθ*nφ
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 3D indices from handle
        cijka = cartid_nrnθnφ[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted     
        status[ia,ja,ka] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA_hiord!(tt,status,onsrc,onarec,grd,rsrc,θsrc,φsrc,lambda,ia,ja,ka)
        
        ## try all neighbors of newly accepted point
        @inbounds for ne=1:6 ## six potential neighbors
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nr-1) || (i<2) || (j>nθ-1) || (j<2) || (k>nφ-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far, active

                # get handle
                #han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθnφ[i,j,k] #i+nr*(j-1)
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1
                
                ## NO NEED to update in narrow band because
                ##   we already know the time tt everywhere...
            end
        end
    end
                               
    ## *NOT* a staggered grid, so no interpolation...
    gradadj = -lambda./vel.^3
        
    return gradadj
end # eikadj_FMM_hiord_SINGLESRC  


##====================================================================##

function isoutrange_sph(ib::Int64,jb::Int64,kb::Int64,nr::Int64,nθ::Int64,nφ::Int64)
    isoutb1st = false
    isoutb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<=1 || ib>=nr || jb<=1 || jb>=nθ || kb<=1 || kb>=nφ
        isoutb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<=2 || ib>=nr-1 || jb<=2 || jb>=nθ-1 || kb<=2 || kb>=nφ-1
        isoutb2nd = true
    end
    return isoutb1st,isoutb2nd
end

##====================================================================##

function calcLAMBDA_hiord!(tt::Array{Float64,3},status::Array{Int64},onsrc::Array{Bool,3},onarec::Array{Bool,3},
                           grd::Grid3DSphere,rsrc::Float64,θsrc::Float64,φsrc::Float64,lambda::Array{Float64,3},
                           i::Int64,j::Int64,k::Int64)

    deltar = grd.Δr
    deltaθ = deg2rad(grd.Δθ)  ## DEG to RAD !!!!
    deltaφ = deg2rad(grd.Δφ)  ## DEG to RAD !!!!
    nr,nθ,nφ = size(lambda)
    isout1st,isout2nd = isoutrange_sph(i,j,k,nr,nθ,nφ)

    if onsrc[i,j,k]==true # in the box containing the src

        aback,aforw,bback,bforw,cback,cforw = adjderivonsource_sph(tt,onsrc,i,j,k,grd,rsrc,θsrc,φsrc)

    else # not on src

        ##
        ## Central differences in Leung & Qian, 2006 scheme!
        ##  
        if !isout2nd
            ##
            ## Fourth order central diff O(h^4); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ## 12*h => 12*dh/2 = 6*dh
            aback = -( -tt[i+1,j,k]+8.0*tt[i,j,k]-8.0*tt[i-1,j,k]+tt[i-2,j,k] )/(6.0*deltar)
            aforw = -( -tt[i+2,j,k]+8.0*tt[i+1,j,k]-8.0*tt[i,j,k]+tt[i-1,j,k] )/(6.0*deltar)
            bback = -( -tt[i,j+1,k]+8.0*tt[i,j,k]-8.0*tt[i,j-1,k]+tt[i,j-2,k] )/(6.0*(grd.r[i]*deltaθ))
            bforw = -( -tt[i,j+2,k]+8.0*tt[i,j+1,k]-8.0*tt[i,j,k]+tt[i,j-1,k] )/(6.0*(grd.r[i]*deltaθ))
            cback = -( -tt[i,j,k+1]+8.0*tt[i,j,k]-8.0*tt[i,j,k-1]+tt[i,j,k-2] )/(6.0*(grd.r[i]*sind(grd.θ[j])*deltaφ))
            cforw = -( -tt[i,j,k+2]+8.0*tt[i,j,k+1]-8.0*tt[i,j,k]+tt[i,j,k-1] )/(6.0*(grd.r[i]*sind(grd.θ[j])*deltaφ))
            
        else
            ##
            ## Central diff (Leung & Qian, 2006); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ##  2*h => 2*dh/2 = dh
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/deltar
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/deltar
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/(grd.r[i]*deltaθ)
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/(grd.r[i]*deltaθ)
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/(grd.r[i]*sind(grd.θ[j])*deltaφ)

        end                     

    end                    

    ##-----------------------------------
    aforwplus  = ( aforw+abs(aforw) )/2.0
    aforwminus = ( aforw-abs(aforw) )/2.0
    abackplus  = ( aback+abs(aback) )/2.0
    abackminus = ( aback-abs(aback) )/2.0

    bforwplus  = ( bforw+abs(bforw) )/2.0
    bforwminus = ( bforw-abs(bforw) )/2.0
    bbackplus  = ( bback+abs(bback) )/2.0
    bbackminus = ( bback-abs(bback) )/2.0

    cforwplus  = ( cforw+abs(cforw) )/2.0
    cforwminus = ( cforw-abs(cforw) )/2.0
    cbackplus  = ( cback+abs(cback) )/2.0
    cbackminus = ( cback-abs(cback) )/2.0
    
    ##==============================================================
    ### Fix problems with higher order derivatives...
    ### If the denominator is zero, try using shorter stencil (see above)
    if !isout2nd
        if aforwplus==0.0 && abackminus==0.0 && bforwplus==0.0 && bbackminus==0.0 && cforwplus==0.0 && cbackminus==0.0
            ## revert to smaller stencil
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/deltar
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/deltar
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/(grd.r[i]*deltaθ)
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/(grd.r[i]*deltaθ)
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
            # recompute stuff
            aforwplus  = ( aforw+abs(aforw) )/2.0
            aforwminus = ( aforw-abs(aforw) )/2.0
            abackplus  = ( aback+abs(aback) )/2.0
            abackminus = ( aback-abs(aback) )/2.0
            bforwplus  = ( bforw+abs(bforw) )/2.0
            bforwminus = ( bforw-abs(bforw) )/2.0
            bbackplus  = ( bback+abs(bback) )/2.0
            bbackminus = ( bback-abs(bback) )/2.0
            cforwplus  = ( cforw+abs(cforw) )/2.0
            cforwminus = ( cforw-abs(cforw) )/2.0
            cbackplus  = ( cback+abs(cback) )/2.0
            cbackminus = ( cback-abs(cback) )/2.0
        end
    end
    ##==============================================================

    ##-------------------------------------------
    ## make SURE lambda was INITIALIZED TO ZERO
    ## Leung & Qian, 2006
    numer =
        (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / deltar +
        (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / (grd.r[i]*deltaθ) +
        (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / (grd.r[i]*sind(grd.θ[j])*deltaφ)
    
    denom = (aforwplus - abackminus)/deltar + (bforwplus - bbackminus)/(grd.r[i]*deltaθ) +
        (cforwplus - cbackminus)/(grd.r[i]*sind(grd.θ[j])*deltaφ)

    lambda[i,j,k] = numer/denom    

    if denom==0.0
        # @show onsrc[i,j]
        # @show aforwplus,abackminus
        # @show bforwplus,bbackminus
        error("calcLAMBDA_hiord!(): denom==0")
    end

    # @show aforwplus,aforwminus,abackplus,abackminus
    # @show bforwplus,bforwminus,bbackplus,bbackminus
    # @show cforwplus,cforwminus,cbackplus,cbackminus
    return nothing #lambda
end



##################################################
#end # end module
##################################################
