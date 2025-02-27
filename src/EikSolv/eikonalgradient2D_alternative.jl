
#############################################################################


"""
$(TYPEDSIGNATURES)

Calculate the gradient using the adjoint state method for 2D velocity models. 
Returns the gradient of the misfit function with respect to velocity calculated at the given point (velocity model). 
The gradient is calculated using the adjoint state method.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 2D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays 
- `pickobs`: observed traveltime picks
- `stdobs`: standard deviation of error on observed traveltime picks, an array with same shape than `pickobs`
- `gradttalgo`: the algorithm to use to compute the forward and gradient, one amongst the following
    * "gradFS\\_podlec", fast sweeping method using Podvin-Lecomte stencils for forward, fast sweeping method for adjoint   
    * "gradFMM\\_podlec," fast marching method using Podvin-Lecomte stencils for forward, fast marching method for adjoint 
    * "gradFMM\\_hiord", second order fast marching method for forward, high order fast marching method for adjoint 
- `smoothgrad`: smooth the gradient? true or false

# Returns
- `grad`: the gradient as a 2D array

"""
function eikgradient2Dalt(vel::Array{Float64,2}, grd::AbstractGridEik2D,coordsrc::Array{Float64,2},
                          coordrec::Vector{Array{Float64,2}},pickobs::Vector{Vector{Float64}},
                          stdobs::Vector{Vector{Float64}} ;    gradttalgo::String,
                          smoothgradsourceradius::Integer=0,smoothgrad::Bool=false )

    if typeof(grd)==Grid2DCart
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
        if gradttalgo!="gradFMM_hiord"
            error("gradttime2Dalt(): For spherical coordinates the only available algorithm is 'gradFMM_hiord'. ")
        end
    end
    if simtype==:cartesian
        n1,n2 = grd.nx,grd.ny #size(vel)  ## NOT A STAGGERED GRID!!!
        ax1min,ax1max = grd.x[1],grd.x[end]
        ax2min,ax2max = grd.y[1],grd.y[end]
    elseif simtype==:spherical
        n1,n2 = grd.nr,grd.nθ
        ax1min,ax1max = grd.r[1],grd.r[end]
        ax2min,ax2max = grd.θ[1],grd.θ[end]
    end

    # some checks
    @assert size(coordsrc,2)==2
    @assert all(vel.>0.0)
    @assert size(coordsrc,1)==length(coordrec)
    @assert all(ax1min.<=coordsrc[:,1].<=ax1max)
    @assert all(ax2min.<=coordsrc[:,2].<=ax2max)
    
    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(n1,n2,nchu)
    ## do the calculations
    @sync begin 
        for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @async tmpgrad[:,:,s] = remotecall_fetch(calcgradsomesrc2Dalt,wks[s],vel,
                                                     coordsrc[igrs,:],coordrec[igrs],
                                                     grd,stdobs[igrs],pickobs[igrs],
                                                     gradttalgo,smoothgradsourceradius )
        end
    end
    grad = dropdims(sum(tmpgrad,dims=3),dims=3)
    #@assert !any(isnan.(grad))

    ## smooth gradient
    if smoothgrad
        l = 5  # 5 pixels kernel
        grad = smoothgradient(l,grad)
    end
 
    return grad
end

###############################################################################

"""
$(TYPEDSIGNATURES)


Calculate the gradient for some requested sources 
"""
function calcgradsomesrc2Dalt(vel::Array{Float64,2},xysrc::Array{Float64,2},
                           coordrec::Vector{Array{Float64,2}},grd::AbstractGridEik2D,
                           stdobs::Vector{Vector{Float64}},pickobs1::Vector{Vector{Float64}},
                           gradalgo::String,smoothgradsourceradius::Integer )
                           
    nx,ny=size(vel)
    nsrc = size(xysrc,1)
    nrec = size(coordrec,1)                
    #ttpicks1 = zeros(nrec)
    ttpicks1 = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicks1[i] = zeros(curnrec)
    end

    grad1 = zeros(nx,ny)
    
    if gradalgo=="gradFMM_hiord"
        # in this case velocity and time arrays have the same shape
        ttgrdonesrc = zeros(nx,ny,nsrc)
        ## pre-allocate ttime and status arrays plus the binary heap
        fmmvars = FMMvars2D(nx,ny,refinearoundsrc=true,
                        allowfixsqarg=false)
        ##  No discrete adjoint calculations (continuous adjoint in this case)
        adjvars = nothing
    end

    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        curnrec = size(coordrec[s],1) 
        ttpicks1 = zeros(curnrec)

        ###########################################
        ## calc ttime

        if gradalgo=="gradFMM_podlec"
            ttgrdonesrc = ttFMM_podlec(vel,xysrc[s,:],grd)
            
        elseif gradalgo=="gradFMM_hiord"
            ttFMM_hiord!(fmmvars,vel,xysrc[s,:],grd,adjvars)
            ttgrdonesrc = fmmvars.ttime

        elseif gradalgo=="gradFS_podlec"
            ttgrdonesrc = ttFS_podlec(vel,xysrc[s,:],grd)
            
        else
            println("\ncalcgradsomesrc(): Wrong gradalgo name: $(gradalgo)... \n")
            return nothing
        end
        
        ## ttime at receivers
        for i=1:curnrec
            ttpicks1[i] = bilinear_interp( ttgrdonesrc,grd,coordrec[s][i,:] )
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        if gradalgo=="gradFS_podlec"

            grad1 .+= eikgrad_FS_SINGLESRC(ttgrdonesrc,vel,xysrc[s,:],coordrec[s],grd,
                                           pickobs1[s],ttpicks1,stdobs[s])

        elseif gradalgo=="gradFMM_podlec"

            grad1 .+= eikgrad_FMM_SINGLESRC(ttgrdonesrc,vel,xysrc[s,:],coordrec[s],
                                            grd,pickobs1[s],ttpicks1,stdobs[s])
            
        elseif gradalgo=="gradFMM_hiord"
            grad1 .+= eikgrad_FMM_hiord_SINGLESRC(ttgrdonesrc,vel,xysrc[s,:],coordrec[s],
                                                  grd,pickobs1[s],ttpicks1,stdobs[s])

        else
            println("Wrong gradalgo algo name: $(gradalgo)... ")
            return
        end
        
        ###########################################
        ## smooth gradient around the source
        smoothgradaroundsrc2D!(grad1,xysrc[s,:],grd,radiuspx=smoothgradsourceradius)

    end

    return grad1
end 

##############################################################################

"""
$(TYPEDSIGNATURES)
"""
function recboxlocgrad!(ttime::Array{Float64,2},lambda::Array{Float64,2},ttpicks::Vector{Float64},grd::Grid2DCart,
                        rec::Array{Float64,2},pickobs::Vector{Float64},stdobs::Vector{Float64}; staggeredgrid::Bool)

    ##########################
    ## Init receivers
    ##########################

    if staggeredgrid==false
        ## regular grid
        onarec = zeros(Bool,grd.nx,grd.ny)
        nxmax = grd.nx
        nymax = grd.ny
    elseif staggeredgrid==true
        ## STAGGERED grid
        onarec = zeros(Bool,grd.ntx,grd.nty)
        nxmax = grd.ntx
        nymax = grd.nty
        hgr = grd.hgrid/2.0
    end
    onarec[:,:] .= false
    nrec=size(rec,1)

    fisttimereconbord = true
    for r=1:nrec
        if staggeredgrid==false
            i,j = findclosestnode(rec[r,1],rec[r,2],grd.xinit,grd.yinit,grd.hgrid)
        elseif staggeredgrid==true
            i,j = findclosestnode(rec[r,1],rec[r,2],grd.xinit-hgr,grd.yinit-hgr,grd.hgrid)
        end
        if (i==1) || (i==nxmax) || (j==1) || (j==nymax)
            ## Receivers on the boundary of model
            ## (n⋅∇T)γ = Tcalc - Tobs
            if i==1
                # forward diff
                n∇T = - (ttime[2,j]-ttime[1,j])/grd.hgrid 
            elseif i==nxmax
                # backward diff
                n∇T = (ttime[end,j]-ttime[end-1,j])/grd.hgrid 
            elseif j==1
                # forward diff
                n∇T = - (ttime[i,2]-ttime[i,1])/grd.hgrid 
            elseif j==nymax
                # backward diff
                n∇T = (ttime[i,end]-ttime[i,end-1])/grd.hgrid 
            end
            onarec[i,j] = true
            #############################################
            ##  FIX ME: receivers on the border...     ## <<<<===================#####
            #############################################
            if fisttimereconbord
                @warn(" Receiver(s) on border of model, \n still untested, spurious results may be encountered.")
                fisttimereconbord=false
            end
            ## Taking into account Neumann boundary condition
            ## lambda[i,j] = (ttpicks[r]-pickobs[r])/(n∇T * stdobs[r]^2)
            ## NOT taking into account Neumann boundary condition
            lambda[i,j] =  (ttpicks[r]-pickobs[r])/stdobs[r]^2

        else
            ## Receivers within the model
            onarec[i,j] = true
            lambda[i,j] = (ttpicks[r]-pickobs[r])/stdobs[r]^2
        end
    end
    
    return onarec
end

#############################################################################

"""
$(TYPEDSIGNATURES)
"""
function adjderivonsource(tt::Array{Float64,2},onsrc::Array{Bool,2},i::Int64,j::Int64,
                          xinit::Float64,yinit::Float64,dh::Float64,xsrc::Float64,ysrc::Float64;
                          staggeredgrid::Bool)        

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

    ## If we are in the square containing the source,
    ## use real position of source for the derivatives.
    ## Calculate tt on x and y on the edges of the square
    ##  H is the projection of the point src onto X and Y
    if staggeredgrid==false
        xp = Float64(i-1)*dh+xinit
        yp = Float64(j-1)*dh+yinit
    else
        hgr = dh/2.0
        xp = Float64(i-1)*dh+xinit-hgr
        yp = Float64(j-1)*dh+yinit-hgr
    end
    ## distances P to src
    distHPx = abs(xp-xsrc)
    distHPy = abs(yp-ysrc)
    dist2src = sqrt( distHPx^2+distHPy^2 )
    # assert dist2src>0.0 otherwise a singularity will occur
    @assert dist2src>0.0

    ## Calculate the traveltime to hit the x side
    ## time at H along x
    thx = tt[i,j]*distHPy/dist2src
    ## Calculate the traveltime to hit the y side
    ## time at H along y
    thy = tt[i,j]*distHPx/dist2src
    
    if onsrc[i+1,j]==true                            
        aback = -(tt[i,j]-tt[i-1,j])/dh
        if distHPx==0.0
            aforw = 0.0 # point exactly on source
        else
            aforw = -(thx-tt[i,j])/distHPx # dist along x
        end
    elseif onsrc[i-1,j]==true
        if distHPx==0.0
            aback = 0.0 # point exactly on source
        else
            aback = -(tt[i,j]-thx)/distHPx # dist along x
        end
        aforw = -(tt[i+1,j]-tt[i,j])/dh
    end
    if onsrc[i,j+1]==true
        bback = -(tt[i,j]-tt[i,j-1])/dh
        if distHPy==0.0
            bforw = 0.0 # point exactly on source
        else
            bforw = -(thy-tt[i,j])/distHPy # dist along y
        end
    else onsrc[i,j-1]==true
        if distHPy==0.0
            bback = 0.0 # point exactly on source
        else
            bback = -(tt[i,j]-thy)/distHPy # dist along y
        end
        bforw = -(tt[i,j+1]-tt[i,j])/dh
    end

    return aback,aforw,bback,bforw
end

#############################################################################

"""
$(TYPEDSIGNATURES)
"""
function eikgrad_FS_SINGLESRC(ttime::Array{Float64,2},vel::Array{Float64,2},
                         src::Vector{Float64},rec::Array{Float64,2},
                         grd::Grid2DCart,pickobs::Vector{Float64},
                         ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(2,)
    mindistsrc = 1e-5
    epsilon = 1e-5
   
    #@assert size(src)=
    @assert size(ttime)==(size(vel).+1)
    ## ntx,nty = size(ttime)
 
    
    #println("ntx,nty ",ntx," ",nty)
    lambdaold = zeros((grd.ntx,grd.nty)) .+ 1e32
    lambdaold[:,1]   .= 0.0
    lambdaold[:,end] .= 0.0
    lambdaold[1,:]   .= 0.0
    lambdaold[end,:] .= 0.0   

    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    ntx = grd.ntx
    nty = grd.nty
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    
    ## source
    ## STAGGERED grid
    onsrc,xsrc,ysrc = sourceboxlocgrad!(tt,vel,src,grd; staggeredgrid=true )
    # receivers
    # lambdaold!!!!
    onarec = recboxlocgrad!(tt,lambdaold,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=true)

    ######################################################

    iswe = [2 ntx-1 1; ntx-1 2 -1; ntx-1 2 -1; 2 ntx-1 1]
    jswe = [2 nty-1 1; 2 nty-1  1; nty-1 2 -1; nty-1 2 -1]
    
    ###################################################
    lambdanew=copy(lambdaold)
    #pa=0
    swedifference = 100*epsilon
    while swedifference>epsilon 
        #pa +=1
        lambdaold[:,:] =  lambdanew

        for swe=1:4
      
            for j=jswe[swe,1]:jswe[swe,3]:jswe[swe,2]
                for i=iswe[swe,1]:iswe[swe,3]:iswe[swe,2]
                                     
                    if onsrc[i,j]==true # in the box containing the src

                        aback,aforw,bback,bforw = adjderivonsource(tt,onsrc,i,j,xinit,yinit,dh,xsrc,ysrc,staggeredgrid=true)
                        
                    else # not on src
                        ## Leung & Qian, 2006 ( -deriv...)
                        aback = -(tt[i,j]  -tt[i-1,j])/dh
                        aforw = -(tt[i+1,j]-tt[i,j])/dh
                        bback = -(tt[i,j]  -tt[i,j-1])/dh
                        bforw = -(tt[i,j+1]-tt[i,j])/dh
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

                    ## Leung & Qian, 2006 
                    numer = (abackplus * lambdanew[i-1,j] - aforwminus * lambdanew[i+1,j] ) / dh +
                        (bbackplus * lambdanew[i,j-1] - bforwminus * lambdanew[i,j+1]) / dh
                    
                    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh
                    
                    ###################################################################
                    ###################################################################
                    
                    if onarec[i,j]==true 
                        #gradT = sqrt( (tt[i+1,j]-tt[i,j])^2 + (tt[i,j+1]-tt[i,j])^2 )
                        # numer2 = numer + dtonarec[i,j]
                        # lambdaprop = numer2/denom 
                        # lambdanew[i,j] = min(lambdaold[i,j],lambdaprop)
                        lambdanew[i,j] = lambdaold[i,j]
                    else 
                        lambdaprop = numer/denom
                        lambdanew[i,j] = min(lambdaold[i,j],lambdaprop)
                    end                
                    
                end
            end
        end
   
        ##################################################
        # println(">>>>err=',NP.sum(abs(lambdanew-lambdaold)),'\n'
        # print 'max old new',lambdanew.max(),lambdaold.max()
        swedifference = sum(abs.(lambdanew .- lambdaold))
        ##################################################
    end

    ## STAGGERED GRID, so lambdanew[1:end-1,1:end-1]... any better idea? Interpolation?
    # gradadj = -lambdanew[1:end-1,1:end-1]./vel.^3
    # Average grad at the center of cells (velocity cells -STAGGERED GRID)
    averlambda = ( lambdanew[1:end-1,1:end-1].+lambdanew[2:end,1:end-1].+
                   lambdanew[1:end-1,2:end].+lambdanew[2:end,2:end]) ./ 4.0
    gradadj = -averlambda./vel.^3
        
    return gradadj
end

###############################################################################

"""
$(TYPEDSIGNATURES)
"""
function eikgrad_FMM_SINGLESRC(ttime::Array{Float64,2},vel::Array{Float64,2},
                               src::Vector{Float64},rec::Array{Float64,2},
                               grd::Grid2DCart,pickobs::Vector{Float64},
                               ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(2,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    @assert size(ttime)==(size(vel).+1)
    ##nx,ny = size(ttime)
    
    #println("nx,ny ",nx," ",ny)
    lambda = zeros((grd.ntx,grd.nty)) # + 1e32
 
    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    ntx = grd.ntx
    nty = grd.nty
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    
    ## source
    ## STAGGERED grid
    onsrc,xsrc,ysrc = sourceboxlocgrad!(tt,vel,src,grd, staggeredgrid=true )
    ## receivers
    onarec = recboxlocgrad!(tt,lambda,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=true)
    
    ######################################################
    neigh = SA[1  0;
             0  1;
            -1  0;
             0 -1]

    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,ntx,nty)
    status[:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :] .= 2 ## set to accepted on boundary
    status[ntx,:] .= 2 ## set to accepted on boundary
    status[:, 1] .= 2 ## set to accepted on boundary
    status[:,nty] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((ntx,nty),find(status.==2))
    ijrec = findall(status.==2)
    irec = [x[1] for x in ijrec]
    jrec = [x[2] for x in ijrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=ntx*nty
    #bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))
    bheap = init_maxheap(Nmax)

    ## conversion cart to lin indices, old sub2ind
    linid_ntxnty = LinearIndices((ntx,nty))
    ## conversion lin to cart indices, old sub2ind
    cartid_ntxnty = CartesianIndices((ntx,nty))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:4 ## four potential neighbors
            
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            ## For the adjoint SKIP BORDERS!!!
            if (i>ntx-1) || (i<2) || (j>nty-1) || (j<2)
                continue
            end
            
            if status[i,j]==0 ## far

                # get handle
                #han = sub2ind((ntx,nty),i,j)
                han = linid_ntxnty[i,j]
                ## add tt of point to binary heap
                insert_maxheap!(bheap,tt[i,j],han)
                # change status, add to narrow band
                status[i,j]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = ntx*nty
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh[]<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 2D indices from handle
        #ia,ja = ind2sub((ntx,nty),han)
        ja = div(han,ntx) +1
        ia = han - ntx*(ja-1)
        # set status to accepted     
        status[ia,ja] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA!(tt,status,onsrc,onarec,dh,
                    xinit,yinit,xsrc,ysrc,lambda,ia,ja)
        
        ## try all neighbors of newly accepted point
        for ne=1:4 
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
 
            ## if the point is out of bounds skip this iteration
            ## For the adjoint SKIP BORDERS!!!
            if (i>ntx-1) || (i<2) || (j>nty-1) || (j<2)
                continue
            end
            
            if status[i,j]==0 ## far, active

                # get handle
                #han = sub2ind((ntx,nty),i,j)
                han = linid_ntxnty[i,j] #i+ntx*(j-1)
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j],han)
                # change status, add to narrow band
                status[i,j]=1
                
                ## NO NEED to update in narrow band because
                ##   we already know the time tt everywhere...
            end
        end
    end

    ## STAGGERED GRID, so lambdanew[1:end-1,1:end-1]
    # gradadj = -lambdanew[1:end-1,1:end-1]./vel.^3
    # Average grad at the center of cells (velocity cells -STAGGERED GRID)
    averlambda = ( lambda[1:end-1,1:end-1].+lambda[2:end,1:end-1].+
                   lambda[1:end-1,2:end].+lambda[2:end,2:end]) ./ 4.0
    gradadj = -averlambda./vel.^3
        
    return gradadj
end # eikadj_FMM_SINGLESRC  


##--------------------------------------------

"""
$(TYPEDSIGNATURES)
"""
function calcLAMBDA!(tt::Array{Float64,2},status::Array{Int64,2},onsrc::Array{Bool,2},
                     onarec::Array{Bool,2},dh::Float64,xinit::Float64,yinit::Float64,
                     xsrc::Float64,ysrc::Float64,lambda::Array{Float64,2},i::Int64,j::Int64)
    
    if onsrc[i,j]==true # in the box containing the src

        aback,aforw,bback,bforw = adjderivonsource(tt,onsrc,i,j,xinit,yinit,dh,xsrc,ysrc,staggeredgrid=true)
        
    else # not on src
        aback = -(tt[i,j]  -tt[i-1,j])/dh
        aforw = -(tt[i+1,j]-tt[i,j])/dh
        bback = -(tt[i,j]  -tt[i,j-1])/dh
        bforw = -(tt[i,j+1]-tt[i,j])/dh
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

    ## make SURE lambda was INITIALIZED TO ZERO
    ## Leung & Qian, 2006 
    numer = (abackplus * lambda[i-1,j] - aforwminus * lambda[i+1,j] ) / dh +
        (bbackplus * lambda[i,j-1] - bforwminus * lambda[i,j+1]) / dh
    
    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh
    
    ###################################################################
    
    # if onarec[i,j]==true 
    #     #gradT = sqrt( (tt[i+1,j]-tt[i,j])^2 + (tt[i,j+1]-tt[i,j])^2 )
    #     numer2 = numer + dtonarec[i,j]
    #     lambda[i,j] = numer2/denom 
    # else
    ## can't be on a rec on FMM
        lambda[i,j] = numer/denom
    # end   
        
    return #lambda
end

###############################################################################
###############################################################################

"""
$(TYPEDSIGNATURES)
"""
function eikgrad_FMM_hiord_SINGLESRC(ttime::Array{Float64,2},vel::Array{Float64,2},
                         src::Vector{Float64},rec::Array{Float64,2},
                         grd::Grid2DCart,pickobs::Vector{Float64},
                         ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(2,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel)) ## SAME SIZE!!
    nx,ny = grd.nx,grd.ny

    lambda = zeros((nx,ny))    
    onarec = zeros(Bool,nx,ny)
    onarec[:,:] .= false

    ##########
    ## Copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    
    ## source
    ## regular grid
    onsrc,xsrc,ysrc = sourceboxlocgrad!(tt,vel,src,grd, staggeredgrid=false )
    ## receivers
    onarec = recboxlocgrad!(tt,lambda,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=false)

    
    ######################################################
    neigh = SA[1  0;
             0  1;
            -1  0;
             0 -1]

    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,nx,ny)
    status[:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :] .= 2 ## set to accepted on boundary
    status[nx,:] .= 2 ## set to accepted on boundary
    status[:, 1] .= 2 ## set to accepted on boundary
    status[:,ny] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nx,ny),find(status.==2))
    ijrec = findall(status.==2)
    irec = [x[1] for x in ijrec]
    jrec = [x[2] for x in ijrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=nx*ny
    #bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))
    bheap = init_maxheap(Nmax)

    ## conversion cart to lin indices, old sub2ind
    linid_nxny = LinearIndices((nx,ny))
    ## conversion lin to cart indices, old sub2ind
    cartid_nxny = CartesianIndices((nx,ny))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:4 ## four potential neighbors
            
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nx-1) || (i<2) || (j>ny-1) || (j<2)
                continue
            end
            
            if status[i,j]==0 ## far

                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j],han)
                # change status, add to narrow band
                status[i,j]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = nx*ny
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh[]<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 2D indices from handle
        #ia,ja = ind2sub((nx,ny),han)
        ja = div(han,nx) +1
        ia = han - nx*(ja-1)
        # set status to accepted     
        status[ia,ja] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA_hiord!(tt,status,onsrc,onarec,dh,
                            xinit,yinit,xsrc,ysrc,lambda,ia,ja)

        ## try all neighbors of newly accepted point
        for ne=1:4 
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
 
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nx-1) || (i<2) || (j>ny-1) || (j<2)
                continue
            end
            
            if status[i,j]==0 ## far, active

                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j] #i+nx*(j-1)
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

"""
$(TYPEDSIGNATURES)
"""
function isoutrange(ib::Int64,jb::Int64,nx::Int64,ny::Int64)
    isoutb1st = false
    isoutb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<=1 || ib>=nx || jb<=1 || jb>=ny
        isoutb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<=2 || ib>=nx-1 || jb<=2 || jb>=ny-1
        isoutb2nd = true
    end
    return isoutb1st,isoutb2nd
end

##------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)
"""
function calcLAMBDA_hiord!(tt::Array{Float64,2},status::Array{Int64},
                             onsrc::Array{Bool,2},onarec::Array{Bool,2},
                             dh::Float64,xinit::Float64,yinit::Float64,
                             xsrc::Float64,ysrc::Float64,lambda::Array{Float64,2},i::Int64,j::Int64)

    nx,ny = size(lambda)
    isout1st,isout2nd = isoutrange(i,j,nx,ny)

    if onsrc[i,j]==true #  in the box containing the src

        aback,aforw,bback,bforw = adjderivonsource(tt,onsrc,i,j,xinit,yinit,dh,xsrc,ysrc,staggeredgrid=false)

    else # not on src
             
        ##
        ## Central differences in Leung & Qian, 2006 scheme!
        ##  
        if !isout2nd
            ##
            ## Fourth order central diff O(h^4); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ## 12*h => 12*dh/2 = 6*dh
            aback = -( -tt[i+1,j]+8.0*tt[i,j]-8.0*tt[i-1,j]+tt[i-2,j] )/(6.0*dh)
            aforw = -( -tt[i+2,j]+8.0*tt[i+1,j]-8.0*tt[i,j]+tt[i-1,j] )/(6.0*dh)
            bback = -( -tt[i,j+1]+8.0*tt[i,j]-8.0*tt[i,j-1]+tt[i,j-2] )/(6.0*dh)
            bforw = -( -tt[i,j+2]+8.0*tt[i,j+1]-8.0*tt[i,j]+tt[i,j-1] )/(6.0*dh)

        else
            ##
            ## Central diff (Leung & Qian, 2006); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ##  2*h => 2*dh/2 = dh
            aback = -(tt[i,j]  -tt[i-1,j])/dh
            aforw = -(tt[i+1,j]-tt[i,j])/dh
            bback = -(tt[i,j]  -tt[i,j-1])/dh
            bforw = -(tt[i,j+1]-tt[i,j])/dh
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
            aback = -(tt[i,j]  -tt[i-1,j])/dh
            aforw = -(tt[i+1,j]-tt[i,j])/dh
            bback = -(tt[i,j]  -tt[i,j-1])/dh
            bforw = -(tt[i,j+1]-tt[i,j])/dh
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
    numer = (abackplus * lambda[i-1,j] - aforwminus * lambda[i+1,j] ) / dh +
        (bbackplus * lambda[i,j-1] - bforwminus * lambda[i,j+1]) / dh
    
    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh
    
    lambda[i,j] = numer/denom

    ##================================================
    # try once more to fix denom==0.0
    if denom==0.0

        # set ttime on central pixel as the mean of neighbors
        ttmp = (tt[i+1,j]+tt[i-1,j]+tt[i,j+1]+tt[i,j-1])/4.0

        ## revert to smaller stencil
        aback = -(ttmp  -tt[i-1,j])/dh
        aforw = -(tt[i+1,j]-ttmp)/dh
        bback = -(ttmp  -tt[i,j-1])/dh
        bforw = -(tt[i,j+1]-ttmp)/dh
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
        # @show i,j
        # @show isout2nd
        # @show onsrc[i,j]
        # @show aforw,aback
        # @show bforw,bback
        # @show aforwplus,abackminus
        # @show bforwplus,bbackminus
        # @show aforwplus,aforwminus
        # @show abackplus,abackminus
        # @show bforwplus,bforwminus
        # @show bbackplus,bbackminus
        # @show tt[i-1,j-1],tt[i,j-1],tt[i+1,j-1]
        # @show tt[i-1,j],tt[i,j],tt[i+1,j]
        # @show tt[i-1,j+1],tt[i,j+1],tt[i+1,j+1]

        error("calcLAMBDA_hiord!(): denom==0, (i,j)=($i,$j), onsrc: $(onsrc[i,j]), 2nd ord.: $(!isout2nd)")
    end

    return #lambda
end


##################################################
##################################################


"""
$(TYPEDSIGNATURES)
"""
function sourceboxlocgrad_sph!(ttime::Array{Float64,2},vel::Array{Float64,2},srcpos::Vector{Float64},
                               grd::Grid2DSphere )
    
    ##########################
    ##   Init source
    ##########################
    mindistsrc = 0.01*grd.Δr
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
        if rsrc < max_r-0.02*grd.Δr
            rsrc = rsrc+0.01*grd.Δr
        else #(make sure it's not already at the bottom θ)
            rsrc = rsrc-0.01*grd.Δr
        end
        if θsrc < max_θ-0.02*grd.Δθ
            θsrc = θsrc+0.01*grd.Δθ
        else #(make sure it's not alreadθ at the bottom θ)
            θsrc = θsrc-0.01*grd.Δθ
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

"""
$(TYPEDSIGNATURES)
"""
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

    fisttimereconbord = true
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
            #############################################
            ##  FIX ME: receivers on the border...     ## <<<<===================#####
            #############################################
            if fisttimereconbord
                @warn(" Receiver(s) on border of model, \n still untested, spurious results may be encountered.")
                fisttimereconbord=false
            end
            ## Taking into account Neumann boundary condition
            ## lambda[i,j] = (ttpicks[r]-pickobs[r])/(n∇T * stdobs[r]^2)
            ## NOT taking into account Neumann boundary condition
            lambda[i,j] =  (ttpicks[r]-pickobs[r])/stdobs[r]^2

        else
            ## Receivers within the model
            onarec[i,j] = true
            lambda[i,j] = (ttpicks[r]-pickobs[r])/stdobs[r]^2
        end
    end
    
    return onarec
end

#############################################################################

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
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
    #bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))
    bheap = init_maxheap(Nmax)

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
        if bheap.Nh[]<1
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

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
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
        numer = (abackplus * lambda[i-1,j] - aforwminus * lambda[i+1,j] ) / deltar +
            (bbackplus * lambda[i,j-1] - bforwminus * lambda[i,j+1]) / (grd.r[i]*deltaθ)
        denom = (aforwplus - abackminus)/deltar + (bforwplus - bbackminus)/(grd.r[i]*deltaθ)
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

##################################################################################


"""
$(TYPEDSIGNATURES)
"""
function sourceboxlocgrad!(ttime::Array{Float64,2},vel::Array{Float64,2},srcpos::Vector{Float64},
                           grd::Grid2DCart; staggeredgrid::Bool )
    
    ##########################
    ##   Init source
    ##########################
    mindistsrc = 0.01*grd.hgrid
    xsrc,ysrc = srcpos[1],srcpos[2]
    dh = grd.hgrid

    if staggeredgrid==false
        ## regular grid
        onsrc = zeros(Bool,grd.nx,grd.ny)
        onsrc[:,:] .= false
        ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
        max_x = (grd.nx-1)*grd.hgrid+grd.xinit
        max_y = (grd.ny-1)*grd.hgrid+grd.yinit
        
    elseif staggeredgrid==true
        ## STAGGERED grid
        onsrc = zeros(Bool,grd.ntx,grd.nty)
        onsrc[:,:] .= false
        ## grd.xinit-hgr because TIME array on STAGGERED grid
        hgr = grd.hgrid/2.0
        ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
        max_x = (grd.ntx-1)*grd.hgrid+grd.xinit
        max_y = (grd.nty-1)*grd.hgrid+grd.yinit       

     end
   
    halfg = 0.0    
    src_on_nodeedge = false   

    ## Check if the source is on an edge/node
    ## loop to make sure we don't accidentally move the src to another node/edge
    while (sqrt(rx^2+ry^2)<=mindistsrc) || (abs(rx)<=mindistsrc) || (abs(ry)<=mindistsrc)
        src_on_nodeedge = true

        ## amount of shift
        sft = 0.05*dh  #0.001*dh
        clo = 0.02*dh

        ## shift the source 
        if xsrc < max_x-clo
            xsrc = xsrc+sft
        else #(make sure it's not already at the bottom y)
            xsrc = xsrc-sft
        end
        if ysrc < max_y-clo
            ysrc = ysrc+sft
        else #(make sure it's not already at the bottom y)
            ysrc = ysrc-sft
        end

        # print("new time at src:  $(tt[ix,iy]) ")
        ## recompute parameters related to position of source
        if staggeredgrid==false
            ## regular grid
            ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
            rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
            ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)        
        elseif staggeredgrid==true
            ## STAGGERED grid
            ix,iy = findclosestnode(xsrc,ysrc,grd.xinit-hgr,grd.yinit-hgr,grd.hgrid) 
            rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
            ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr) 
        end
    end

    ## To avoid singularities, the src can only be inside a box,
    ##  not on a grid node or edge... see above. So we need to have
    ##  always 4 nodes where onsrc is true
    if (rx>=halfg) & (ry>=halfg)
        onsrc[ix:ix+1,iy:iy+1] .= true
    elseif (rx<halfg) & (ry>=halfg)
        onsrc[ix-1:ix,iy:iy+1] .= true
    elseif (rx<halfg) & (ry<halfg)
        onsrc[ix-1:ix,iy-1:iy] .= true
    elseif (rx>=halfg) & (ry<halfg)
        onsrc[ix:ix+1,iy-1:iy] .= true
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
            if staggeredgrid==false
                ## regular grid
                xp = (i-1)*grd.hgrid+grd.xinit
                yp = (j-1)*grd.hgrid+grd.yinit
                ii = Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
                jj = Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
                #### vel[isrc[1,1],jsrc[1,1]] 
                ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]#[isrc[1,1],jsrc[1,1]]
            elseif staggeredgrid==true
                ## STAGGERED grid
                xp = (i-1)*grd.hgrid+grd.xinit
                yp = (j-1)*grd.hgrid+grd.yinit
                ii = i-1 
                jj = j-1 
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]
            end
        end 
    end
    # returning also xsrc,ysrc because the src may have been repositioned
    #   to avoid singularities
    return onsrc,xsrc,ysrc
end

#############################################################################
