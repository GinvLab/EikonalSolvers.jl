

#######################################################
##     Misfit of gradient using adjoint              ## 
#######################################################


###############################################################################

## @doc raw because of some backslashes in the string...
@doc raw"""
    gradttime2Dsphere(vel::Array{Float64,2},grd::Grid2DSphere,coordsrc::Array{Float64,2},coordrec::Array{Float64,2},
                pickobs::Array{Float64,2},stdobs::Array{Float64,2})

Calculate the gradient using the adjoint state method for 2D velocity models. 
Returns the gradient of the misfit function with respect to velocity calculated at the given point (velocity model). 
The gradient is calculated using the adjoint state method.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 2D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y), a 2-column array 
- `pickobs`: observed traveltime picks
- `stdobs`: standard deviation of error on observed traveltime picks, an array with same shape than `pickobs`
The algorithm to use to compute the forward and gradient is "gradFMM\_hiord", second order fast marching method for forward, high order fast marching method for adjoint calculations.

# Returns
- `grad`: the gradient as a 2D array

"""
function gradttime2Dsphere(vel::Array{Float64,2}, grd::Grid2DSphere,coordsrc::Array{Float64,2},coordrec::Array{Float64,2},
                    pickobs::Array{Float64,2},stdobs::Array{Float64,2})

    @assert size(coordsrc,2)==2
    @assert size(coordrec,2)==2
    @assert all(vel.>0.0)
    @assert all(grd.rinit.<=coordsrc[:,1].<=((grd.nr-1)*grd.Δr+grd.rinit))
    @assert all(grd.θinit.<=coordsrc[:,2].<=((grd.nθ-1)*grd.Δθ+grd.θinit))

    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(grd.nr,grd.nθ,nchu)
    ## do the calculations
    @sync begin 
        for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @async tmpgrad[:,:,s] = remotecall_fetch(calcgradsomesrc2D,wks[s],vel,
                                                     coordsrc[igrs,:],coordrec,
                                                     grd,stdobs[:,igrs],pickobs[:,igrs])
        end
    end
    grad = sum(tmpgrad,dims=3)[:,:]
    return grad
end

###############################################################################

"""
Calculate the gradient for some requested sources 
"""
function calcgradsomesrc2D(vel::Array{Float64,2},xθsrc::Array{Float64,2},coordrec::Array{Float64,2},
                           grd::Grid2DSphere,stdobs::Array{Float64,2},pickobs1::Array{Float64,2})
                          
    nx,ny=size(vel)
    nsrc = size(xθsrc,1)
    nrec = size(coordrec,1)                
    ttpicks1 = zeros(nrec)
    grad1 = zeros(nx,ny)
  
    ## adjalgo=="ttFMM_hiord"
    # in this case velocity and time arrays have the same shape
    ttgrdonesrc = zeros(grd.nr,grd.nθ,nsrc)
    
    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        ###########################################
        ## calc ttime
        ## adjalgo=="gradFMM_hiord"
        ttgrdonesrc = ttFMM_hiord(vel,xθsrc[s,:],grd)

        ## ttime at receivers
        for i=1:size(coordrec,1)
            ttpicks1[i] = bilinear_interp_sph( ttgrdonesrc,grd,coordrec[i,1],coordrec[i,2] )
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        ## adjalgo=="gradFMM_hiord"
        grad1 += eikgrad_FMM_hiord_SINGLESRC(ttgrdonesrc,vel,xθsrc[s,:],coordrec,
                                             grd,pickobs1[:,s],ttpicks1,stdobs[:,s])

    end

    return grad1
end 

#############################################################################

function sourceboxlocgrad_sph!(ttime::Array{Float64,2},vel::Array{Float64,2},srcpos::Vector{Float64},
                               grd::Grid2DSphere )
    
    ##########################
    ##   Init source
    ##########################
    mindistsrc = 1e-5
    rsrc,θsrc = srcpos[1],srcpos[2]

    ## regular grid
    onsrc = zeros(Bool,grd.nr,grd.nθ)
    onsrc[:,:] .= false
    ir,iθ = findclosestnode_sph(rsrc,θsrc,grd.rinit,grd.θinit,grd.Δr,grd.Δθ) 
    rr = rsrc-grd.r[ir] #rsrc-((ir-1)*grd.Δr+grd.rinit)
    rθ = θsrc-grd.θ[iθ] #θsrc-((iθ-1)*grd.Δθ+grd.θinit)
    max_r = grd.r[end] #(grd.nr-1)*grd.Δr+grd.rinit
    max_θ = grd.θ[end] #(grd.nθ-1)*grd.Δθ+grd.θinit
        
    halfg = 0.0    
    src_on_nodeedge = false   

    ## loop to make sure we don't accidentally move the src to another node/edge
    ## sqrt(r1^+r2^2 - 2*r1*r2*cos(θ1-θ2))
    dist = sqrt(rsrc^2+grd.r[ir]^2-2.0*rsrc*grd.r[ir]*cosd(θsrc-grd.θ[iθ]))
    while (dist<=mindistsrc) || (abs(rr)<=mindistsrc) || (abs(rθ)<=mindistsrc)
        src_on_nodeedge = true
        
        ## shift the source 
        if rsrc < max_r-0.002*grd.Δr
            rsrc = rsrc+0.001*grd.Δr
        else #(make sure it's not already at the bottom θ)
            rsrc = rsrc-0.001*grd.Δr
        end
        if θsrc < max_θ-0.002*grd.Δθ
            θsrc = θsrc+0.001*grd.Δθ
        else #(make sure it's not alreadθ at the bottom θ)
            θsrc = θsrc-0.001*grd.Δθ
        end

        # print("new time at src:  $(tt[ir,iθ]) ")
        ## recompute parameters related to position of source
        ## regular grid
        ir,iθ = findclosestnode_sph(rsrc,θsrc,grd.rinit,grd.θinit,grd.Δr,grd.Δθ) 
        rr = rsrc-grd.r[ir] #rsrc-((ir-1)*grd.Δr+grd.rinit)
        rθ = θsrc-grd.θ[iθ] #θsrc-((iθ-1)*grd.Δθ+grd.θinit)
        dist = sqrt(rsrc^2+grd.r[ir]^2-2.0*rsrc*grd.r[ir]*cosd(θsrc-grd.θ[iθ]))
    end

    ## To avoid singularities, the src can only be inside a box,
    ##  not on a grid node or edge... see above. So we need to have
    ##  always 4 nodes where onsrc is true
    if (rr>=halfg) & (rθ>=halfg)
        onsrc[ir:ir+1,iθ:iθ+1] .= true
    elseif (rr<halfg) & (rθ>=halfg)
        onsrc[ir-1:ir,iθ:iθ+1] .= true
    elseif (rr<halfg) & (rθ<halfg)
        onsrc[ir-1:ir,iθ-1:iθ] .= true
    elseif (rr>=halfg) & (rθ<halfg)
        onsrc[ir:ir+1,iθ-1:iθ] .= true
    end

    ## if src on node or edge, recalculate traveltimes in box
    if src_on_nodeedge==true
        ## RE-set a new ttime around source 
        #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
        ijsrc = findall(onsrc)
        #for (j,i) in zip(jsrc,isrc)
        for lcart in ijsrc
            #    for i in isrc
            i = lcart[1]
            j = lcart[2]
            ## regular grid
            # rp = (i-1)*grd.hgrid+grd.rinit
            # θp = (j-1)*grd.hgrid+grd.θinit
            ii = Int(floor((rsrc-grd.rinit)/grd.Δr) +1)
            jj = Int(floor((θsrc-grd.θinit)/grd.Δθ) +1)
            #### vel[isrc[1,1],jsrc[1,1]]
            r1=rsrc
            r2=grd.r[ii]
            distp = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[jj]))  
            ttime[i,j] = distp / vel[ii,jj]
        end 
    end
    # returning also rsrc,θsrc because the src may have been repositioned
    #   to avoid singularities
    return onsrc,rsrc,θsrc
end

##############################################################################

function recboxlocgrad_sph!(ttime::Array{Float64,2},lambda::Array{Float64,2},ttpicks::Vector{Float64},grd::Grid2DSphere,
                            rec::Array{Float64,2},pickobs::Vector{Float64},stdobs::Vector{Float64} )

    ##########################
    ## Init receivers
    ##########################

    ## regular grid
    onarec = zeros(Bool,grd.nr,grd.nθ)
    nxmax = grd.nr
    nymax = grd.nθ

    onarec[:,:] .= false
    nrec=size(rec,1)

    deltar = grd.Δr
    deltaθ = deg2rad(grd.Δθ)  ## DEG to RAD !!!!
    
    for r=1:nrec
        i,j = findclosestnode_sph(rec[r,1],rec[r,2],grd.rinit,grd.θinit,grd.Δr,grd.Δθ)

        if (i==1) || (i==nxmax) || (j==1) || (j==nymax)
            ## Receivers on the boundary of model
            ## (n⋅∇T)γ = Tcalc - Tobs
            if i==1
                # forward diff
                n∇T = - (ttime[2,j]-ttime[1,j])/deltar
            elseif i==nxmax
                # backward diff
                n∇T = (ttime[end,j]-ttime[end-1,j])/deltar
            elseif j==1
                # forward diff
                n∇T = - (ttime[i,2]-ttime[i,1])/(grd.r[i]*deltaθ) 
            elseif j==nymax
                # backward diff
                n∇T = (ttime[i,end]-ttime[i,end-1])/(grd.r[i]*deltaθ)
            end
            onarec[i,j] = true
            lambda[i,j] = (ttpicks[r]-pickobs[r])/(n∇T * stdobs[r]^2)
            # println(" Receiver on border of model (i==1)||(i==nx)||(j==1)||(j==ny)")
            # println(" Not yet implemented...")
            # return []
        else
            ## Receivers within the model
            onarec[i,j] = true
            lambda[i,j] = (ttpicks[r]-pickobs[r])/stdobs[r]^2
        end
    end
    
    return onarec
end

#############################################################################

function adjderivonsource_sph(tt::Array{Float64,2},onsrc::Array{Bool,2},i::Int64,j::Int64,
                          grd::Grid2DSphere,rsrc::Float64,θsrc::Float64)        

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

    ## If we are in the box(?) containing the source,
    ## use real position of source for the derivatives.
    ## Calculate tt on x and y on the edges of the square
    ##  H is the projection of the point src onto X and Y
    xp = grd.r[i] #Float64(i-1)*dh+xinit
    yp = grd.θ[j] #Float64(j-1)*dh+yinit

    ## distance to the side "x", along radius
    distHPx = abs(xp-rsrc)

    ## distance to the side "y", along θ
    r1=rsrc
    r2=rsrc
    distHPy = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-yp))  

    ## distance P to src
    r1=rsrc
    r2=grd.r[i]
    dist2src = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[j]))  
    # assert dist2src>0.0 otherwise a singularity will occur
    @assert dist2src>0.0

    deltar = grd.Δr
    deltaθ = deg2rad(grd.Δθ)  ## DEG to RAD !!!!
    arcHPy = abs(rsrc*deg2rad(yp-θsrc)) ## arc distance use RSRC...
    
    ## Calculate the traveltime to hit the x side
    ## time at H along x
    thx = tt[i,j]*distHPx/dist2src
    ## Calculate the traveltime to hit the y side
    ## time at H along y
    thy = tt[i,j]*distHPy/dist2src
    
    if onsrc[i+1,j]==true                            
        aback = -(tt[i,j]-tt[i-1,j])/deltar
        if distHPx==0.0
            aforw = 0.0 # point exactly on source
        else
            aforw = -(thx-tt[i,j])/distHPx # dist along r
        end
    elseif onsrc[i-1,j]==true
        if distHPx==0.0
            aback = 0.0 # point exactly on source
        else
            aback = -(tt[i,j]-thx)/distHPx # dist along r
        end
        aforw = -(tt[i+1,j]-tt[i,j])/deltar
    end
    
    if onsrc[i,j+1]==true
        bback = -(tt[i,j]-tt[i,j-1])/(grd.r[i]*deltaθ)
        if distHPy==0.0
            bforw = 0.0 # point exactly on source
        else
            bforw = -(thy-tt[i,j])/arcHPy # dist along θ arc
        end        
    else onsrc[i,j-1]==true
        if distHPy==0.0
            bback = 0.0 # point exactly on source
        else
            bback = -(tt[i,j]-thy)/arcHPy # dist along θ arc
        end
        bforw = -(tt[i,j+1]-tt[i,j])/(grd.r[i]*deltaθ)
    end

    return aback,aforw,bback,bforw
end


###################################################################################
###################################################################################

function eikgrad_FMM_hiord_SINGLESRC(ttime::Array{Float64,2},vel::Array{Float64,2},
                         src::Vector{Float64},rec::Array{Float64,2},
                         grd::Grid2DSphere,pickobs::Vector{Float64},
                         ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(2,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel)) ## SAME SIZE!!
    nr,nθ = grd.nr,grd.nθ

    lambda = zeros((nr,nθ))    
    onarec = zeros(Bool,nr,nθ)
    onarec[:,:] .= false

    ##########
    ## Copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    #dh = grd.hgrid
    
    ## source
    ## regular grid
    onsrc,rsrc,θsrc = sourceboxlocgrad_sph!(tt,vel,src,grd)
    ## receivers
    onarec = recboxlocgrad_sph!(tt,lambda,ttpicks,grd,rec,pickobs,stdobs)

    
    ######################################################
    neigh = [1  0;
             0  1;
            -1  0;
             0 -1]

    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,nr,nθ)
    status[:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :] .= 2 ## set to accepted on boundary
    status[nr,:] .= 2 ## set to accepted on boundary
    status[:, 1] .= 2 ## set to accepted on boundary
    status[:,nθ] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nr,nθ),find(status.==2))
    ijrec = findall(status.==2)
    irec = [x[1] for x in ijrec]
    jrec = [x[2] for x in ijrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=nr*nθ
    bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## conversion cart to lin indices, old sub2ind
    linid_nrnθ = LinearIndices((nr,nθ))
    ## conversion lin to cart indices, old sub2ind
    cartid_nrnθ = CartesianIndices((nr,nθ))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:4 ## four potential neighbors
            
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nr-1) || (i<2) || (j>nθ-1) || (j<2)
                continue
            end
            
            if status[i,j]==0 ## far

                # get handle
                #han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθ[i,j]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j],han)
                # change status, add to narrow band
                status[i,j]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = nr*nθ
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 2D indices from handle
        #ia,ja = ind2sub((nr,nθ),han)
        ja = div(han,nr) +1
        ia = han - nr*(ja-1)
        # set status to accepted     
        status[ia,ja] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA_hiord!(tt,status,onsrc,onarec,grd,rsrc,θsrc,lambda,ia,ja)
                          

        ## try all neighbors of newly accepted point
        for ne=1:4 
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
 
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nr-1) || (i<2) || (j>nθ-1) || (j<2)
                continue
            end
            
            if status[i,j]==0 ## far, active

                # get handle
                #han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθ[i,j] #i+nr*(j-1)
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j],han)
                # change status, add to narrow band
                status[i,j]=1
                
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

function isoutrange_sph(ib::Int64,jb::Int64,nr::Int64,nθ::Int64)
    isoutb1st = false
    isoutb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<=1 || ib>=nr || jb<=1 || jb>=nθ
        isoutb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<=2 || ib>=nr-1 || jb<=2 || jb>=nθ-1
        isoutb2nd = true
    end
    return isoutb1st,isoutb2nd
end

########################################################################################

function calcLAMBDA_hiord!(tt::Array{Float64,2},status::Array{Int64},onsrc::Array{Bool,2},onarec::Array{Bool,2},
                           grd::Grid2DSphere,rsrc::Float64,θsrc::Float64,lambda::Array{Float64,2},i::Int64,j::Int64)

    deltar = grd.Δr
    deltaθ = deg2rad(grd.Δθ)  ## DEG to RAD !!!!
    nr,nθ = size(lambda)
    isout1st,isout2nd = isoutrange(i,j,nr,nθ)

    if onsrc[i,j]==true #  in the box containing the src

        aback,aforw,bback,bforw = adjderivonsource_sph(tt,onsrc,i,j,grd,rsrc,θsrc)

    else # not on src

        ##
        ## Central differences in Leung & Qian, 2006 scheme!
        ##  
        if !isout2nd
            ##
            ## Fourth order central diff O(h^4); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ## 12*h => 12*dh/2 = 6*dh
            aback = -( -tt[i+1,j]+8.0*tt[i,j]-8.0*tt[i-1,j]+tt[i-2,j] )/(6.0*deltar)
            aforw = -( -tt[i+2,j]+8.0*tt[i+1,j]-8.0*tt[i,j]+tt[i-1,j] )/(6.0*deltar)
            bback = -( -tt[i,j+1]+8.0*tt[i,j]-8.0*tt[i,j-1]+tt[i,j-2] )/(6.0*grd.r[i]*deltaθ)
            bforw = -( -tt[i,j+2]+8.0*tt[i,j+1]-8.0*tt[i,j]+tt[i,j-1] )/(6.0*grd.r[i]*deltaθ)
        else
            ##
            ## Central diff (Leung & Qian, 2006); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ##  2*h => 2*dh/2 = dh
            aback = -(tt[i,j]  -tt[i-1,j])/deltar
            aforw = -(tt[i+1,j]-tt[i,j])/deltar
            bback = -(tt[i,j]  -tt[i,j-1])/(grd.r[i]*deltaθ)
            bforw = -(tt[i,j+1]-tt[i,j])/(grd.r[i]*deltaθ)
        end  
    end                    
    
    ##================================================

    aforwplus  = ( aforw+abs(aforw) )/2.0
    aforwminus = ( aforw-abs(aforw) )/2.0
    abackplus  = ( aback+abs(aback) )/2.0
    abackminus = ( aback-abs(aback) )/2.0

    bforwplus  = ( bforw+abs(bforw) )/2.0
    bforwminus = ( bforw-abs(bforw) )/2.0
    bbackplus  = ( bback+abs(bback) )/2.0
    bbackminus = ( bback-abs(bback) )/2.0

    ##==============================================================
    ### Fix problems with higher order derivatives...
    ### If the denominator is zero, try using shorter stencil (see above)
    if !isout2nd
        if aforwplus==0.0 && abackminus==0.0 && bforwplus==0.0 && bbackminus==0.0
            ## revert to smaller stencil
            aback = -(tt[i,j]  -tt[i-1,j])/deltar
            aforw = -(tt[i+1,j]-tt[i,j])/deltar
            bback = -(tt[i,j]  -tt[i,j-1])/(grd.r[i]*deltaθ)
            bforw = -(tt[i,j+1]-tt[i,j])/(grd.r[i]*deltaθ)
            # recompute stuff
            aforwplus  = ( aforw+abs(aforw) )/2.0
            aforwminus = ( aforw-abs(aforw) )/2.0
            abackplus  = ( aback+abs(aback) )/2.0
            abackminus = ( aback-abs(aback) )/2.0
            bforwplus  = ( bforw+abs(bforw) )/2.0
            bforwminus = ( bforw-abs(bforw) )/2.0
            bbackplus  = ( bback+abs(bback) )/2.0
            bbackminus = ( bback-abs(bback) )/2.0
        end
    end
    ##==============================================================
    
    ## make SURE lambda was INITIALIZED TO ZERO
    ## Leung & Qian, 2006 
    numer = (abackplus * lambda[i-1,j] - aforwminus * lambda[i+1,j] ) / deltar +
        (bbackplus * lambda[i,j-1] - bforwminus * lambda[i,j+1]) / (grd.r[i]*deltaθ)
    
    denom = (aforwplus - abackminus)/deltar + (bforwplus - bbackminus)/(grd.r[i]*deltaθ)
    
    lambda[i,j] = numer/denom

    ##================================================
    # try once more to fix denom==0.0
    if denom==0.0

        # set ttime on central pixel as the mean of neighbors
        ttmp = (tt[i+1,j]+tt[i-1,j]+tt[i,j+1]+tt[i,j-1])/4.0

        ## revert to smaller stencil
        aback = -(ttmp  -tt[i-1,j])/deltar
        aforw = -(tt[i+1,j]-ttmp)/deltar
        bback = -(ttmp  -tt[i,j-1])/(grd.r[i]*deltaθ)
        bforw = -(tt[i,j+1]-ttmp)/(grd.r[i]*deltaθ)
        # recompute stuff
        aforwplus  = ( aforw+abs(aforw) )/2.0
        aforwminus = ( aforw-abs(aforw) )/2.0
        abackplus  = ( aback+abs(aback) )/2.0
        abackminus = ( aback-abs(aback) )/2.0
        bforwplus  = ( bforw+abs(bforw) )/2.0
        bforwminus = ( bforw-abs(bforw) )/2.0
        bbackplus  = ( bback+abs(bback) )/2.0
        bbackminus = ( bback-abs(bback) )/2.0

        ## recompute lambda
        numer = (abackplus * lambda[i-1,j] - aforwminus * lambda[i+1,j] ) / dh +
            (bbackplus * lambda[i,j-1] - bforwminus * lambda[i,j+1]) / dh
        denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh
        lambda[i,j] = numer/denom
        
    end
    
    # if the second fix didn't work exit...
    if denom==0.0
        # @show onsrc[i,j]
        # @show aforwplus,abackminus
        # @show bforwplus,bbackminus
        error("calcLAMBDA_hiord!(): denom==0, (i,j)=($i,$j), 2nd ord.: $(!isout2nd)")
    end

    return #lambda
end

##################################################

##################################################
#end # module EikonalGrad2D                      ##
##################################################
