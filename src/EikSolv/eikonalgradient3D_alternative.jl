
###############################################################################
########################################################################################
""" 
$(TYPEDSIGNATURES)

Calculate the gradient using the adjoint state method for 3D velocity models. 
Returns the gradient of the misfit function with respect to velocity calculated at the given point (velocity model).
The gradient is calculated using the adjoint state method. 
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 3D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y,z), a 3-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y,z), a 3-column array 
- `pickobs`: observed traveltime picks
- `stdobs`: standard deviation of error on observed traveltime picks, an array with same shape than `pickobs`
- `gradttalgo`: the algorithm to use to compute the forward and gradient, one amongst the following
    * "gradFS\\_podlec", fast sweeping method using Podvin-Lecomte stencils for forward, fast sweeping method for adjoint   
    * "gradFMM\\_podlec," fast marching method using Podvin-Lecomte stencils for forward, fast marching method for adjoint 
    * "gradFMM\\_hiord", second order fast marching method for forward, high order fast marching method for adjoint  
- `smoothgrad`: smooth the gradient? true or false

# Returns
- `grad`: the gradient as a 3D array

    """
function eikgradient3Dalt(vel::Array{Float64,3},grd::AbstractGridEik3D,coordsrc::Array{Float64,2},coordrec::Vector{Matrix{Float64}},
                          pickobs::Vector{Vector{Float64}},stdobs::Vector{Vector{Float64}} ;
                          gradttalgo::String="gradFMM_hiord",smoothgradsourceradius::Integer=0,smoothgrad::Bool=true)
    
    if typeof(grd)==Grid3DCart
        simtype = :cartesian
    elseif typeof(grd)==Grid3DSphere
        simtype = :spherical
        if gradttalgo!="gradFMM_hiord"
            error("gradttime3Dalt(): For spherical coordinates the only available algorithm is 'gradFMM_hiord'. ")
        end
    end
    if simtype==:cartesian
        n1,n2,n3 = grd.nx,grd.ny,grd.nz 
        ax1min,ax1max = grd.x[1],grd.x[end]
        ax2min,ax2max = grd.y[1],grd.y[end]
        ax3min,ax3max = grd.z[1],grd.z[end]
    elseif simtype==:spherical
        n1,n2,n3 = grd.nr,grd.nθ,grd.nφ
        ax1min,ax1max = grd.r[1],grd.r[end]
        ax2min,ax2max = grd.θ[1],grd.θ[end]
        ax3min,ax3max = grd.φ[1],grd.φ[end]
    end

    # some checks
    @assert size(coordsrc,2)==3
    @assert all(vel.>0.0)
    @assert size(coordsrc,1)==length(coordrec)
    @assert all(ax1min.<=coordsrc[:,1].<=ax1max)
    @assert all(ax2min.<=coordsrc[:,2].<=ax2max)
    @assert all(ax3min.<=coordsrc[:,3].<=ax3max)


    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(n1,n2,n3,nchu)
    ## do the calculations
    @sync for s=1:nchu
        igrs = grpsrc[s,1]:grpsrc[s,2]
        @async tmpgrad[:,:,:,s] = remotecall_fetch(calcgradsomesrc3Dalt,wks[s],vel,
                                                   coordsrc[igrs,:],coordrec[igrs],
                                                   grd,stdobs[igrs],pickobs[igrs],
                                                   gradttalgo,smoothgradsourceradius )
    end
    grad = dropdims(sum(tmpgrad,dims=4),dims=4)

    ## smooth gradient
    if smoothgrad
        l = 5  # 5 pixels kernel
        grad = smoothgradient(l,grad)
    end
    return grad
end

################################################################3

"""
$(TYPEDSIGNATURES)

Calculate the gradient for some requested sources 
"""
function calcgradsomesrc3Dalt(vel::Array{Float64,3},xyzsrc::Array{Float64,2},coordrec::Vector{Matrix{Float64}},
                              grd::AbstractGridEik3D,stdobs::Vector{Vector{Float64}},pickobs1::Vector{Vector{Float64}},
                              adjalgo::String,smoothgradsourceradius::Integer)

    nx,ny,nz=size(vel)
    nsrc = size(xyzsrc,1)
    grad1 = zeros(nx,ny,nz)

    if adjalgo=="gradFMM_hiord"
        # in this case velocity and time arrays have the same shape
        ttgrdonesrc = zeros(nx,ny,nz,nsrc)
        ## pre-allocate ttime and status arrays plus the binary heap
        fmmvars = FMMvars3D(nx,ny,nz,refinearoundsrc=true,
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
    
        if adjalgo=="gradFMM_podlec"
            ttonesrc = ttFMM_podlec(vel,xyzsrc[s,:],grd)
 
        elseif adjalgo=="gradFMM_hiord"
            ttFMM_hiord!(fmmvars,vel,xyzsrc[s,:],grd,adjvars)
            ttonesrc = fmmvars.ttime

        elseif adjalgo=="gradFS_podlec"
            ttonesrc = ttFS_podlec(vel,xyzsrc[s,:],grd)
 
        else
            println("\ncalcgradsomesrc(): Wrong adjalgo name: $(adjalgo)... \n")
            return nothing
        end
            
        ## ttime at receivers
        for i=1:curnrec
            ttpicks1[i] = trilinear_interp(ttonesrc,grd,coordrec[s][i,:])
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        if adjalgo=="gradFS_podlec"

            grad1 += eikgrad_FS_SINGLESRC(ttonesrc,vel,xyzsrc[s,:],coordrec[s],grd,
                                          pickobs1[s],ttpicks1,stdobs[s])

        elseif adjalgo=="gradFMM_podlec"

            grad1 += eikgrad_FMM_SINGLESRC(ttonesrc,vel,xyzsrc[s,:],coordrec[s],
                                           grd,pickobs1[s],ttpicks1,stdobs[s])
            
        elseif adjalgo=="gradFMM_hiord"
            grad1 += eikgrad_FMM_hiord_SINGLESRC(ttonesrc,vel,xyzsrc[s,:],coordrec[s],
                                                 grd,pickobs1[s],ttpicks1,stdobs[s])

        else
            println("Wrong adjalgo algo name: $(adjalgo)... ")
            return
        end

        #############################################
        ## smooth gradient around the source
        smoothgradaroundsrc3D!(grad1,xyzsrc[s,:],grd,radiuspx=smoothgradsourceradius)

    end
    
    return grad1
end

#################################################################################33


"""
$(TYPEDSIGNATURES)
"""
function adjderivonsource(tt::Array{Float64,3},onsrc::Array{Bool,3},i::Int64,j::Int64,k::Int64,
                          xinit::Float64,yinit::Float64,zinit::Float64,
                          dh::Float64,xsrc::Float64,ysrc::Float64,zsrc::Float64)

    ## If we are in the square containing the source,
    ## use real position of source for the derivatives.
    ## Calculate tt on x and y on the edges of the square
    ##  H is the projection of the point src onto X, Y and Z edges
    xp = Float64(i-1)*dh+xinit
    yp = Float64(j-1)*dh+yinit
    zp = Float64(k-1)*dh+zinit

    distPSx = abs(xp-xsrc) ## abs needed for later...
    distPSy = abs(yp-ysrc)
    distPSz = abs(zp-zsrc)
    dist2src = sqrt(distPSx^2+distPSy^2+distPSz^2)
    # assert dist2src>0.0 otherwise a singularity will occur
    @assert dist2src>0.0

    ## Calculate the traveltime to hit the x edge
    ## time at H along x
    thx = tt[i,j,k]*(sqrt(distPSy^2+distPSz^2))/dist2src
    ## Calculate the traveltime to hit the y edge
    ## time at H along y
    thy = tt[i,j,k]*(sqrt(distPSx^2+distPSz^2))/dist2src
    ## Calculate the traveltime to hit the z edge
    ## time at H along z
    thz = tt[i,j,k]*(sqrt(distPSx^2+distPSy^2))/dist2src
    
    if onsrc[i+1,j,k]==true              
        aback = -(tt[i,j,k]-tt[i-1,j,k])/dh
        if distPSx==0.0
            aforw = 0.0 # point exactly on source
        else
            aforw = -(thx-tt[i,j,k])/distPSx # dist along x
        end
    elseif onsrc[i-1,j,k]==true
        if distPSx==0.0
            aback = 0.0 # point exactly on source
        else
            aback = -(tt[i,j,k]-thx)/distPSx # dist along x
        end
        aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
    end
    if onsrc[i,j+1,k]==true
        bback = -(tt[i,j,k]-tt[i,j-1,k])/dh
        if distPSy==0.0
            bforw = 0.0 # point exactly on source
        else
            bforw = -(thy-tt[i,j,k])/distPSy # dist along y
        end
    else onsrc[i,j-1,k]==true
        if distPSy==0.0
            bback = 0.0 # point exactly on source
        else        
            bback = -(tt[i,j,k]-thy)/distPSy # dist along y
        end
        bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
    end
    if onsrc[i,j,k+1]==true     
        cback = -(tt[i,j,k]-tt[i,j,k-1])/dh
        if distPSz==0.0
            cforw = 0.0 # point exactly on source
        else        
            cforw = -(thz-tt[i,j,k])/distPSz # dist along z
        end
    elseif onsrc[i,j,k-1]==true
        if distPSz==0.0
            cback = 0.0 # point exactly on source
        else        
            cback = -(tt[i,j,k]-thz)/distPSz # dist along z
        end
        cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
    end

    return  aback,aforw,bback,bforw,cback,cforw
end

############################################################################

"""
$(TYPEDSIGNATURES)
"""
function eikgrad_FS_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                         src::Vector{Float64},rec::Array{Float64,2},grd::Grid3DCart,
                         pickobs::Vector{Float64},ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(3,)
    
    @assert size(ttime)==(size(vel).+1)
    ##nx,ny,nz = size(ttime)

    epsilon = 1e-5
    mindistsrc = 1e-5
    
    ## init adjoint variable
    lambdaold = zeros((grd.ntx,grd.nty,grd.ntz)) .+ 1e32
    lambdaold[:,1,:]   .= 0.0
    lambdaold[:,end,:] .= 0.0
    lambdaold[1,:,:]   .= 0.0
    lambdaold[end,:,:] .= 0.0
    lambdaold[:,:,1]   .= 0.0
    lambdaold[:,:,end] .= 0.0

 
    #copy ttime 'cause it might be changed around src
    tt = copy(ttime)
    dh = grd.hgrid

    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    zinit = grd.zinit
    ntx = grd.ntx
    nty = grd.nty
    ntz = grd.ntz
    
    ## source
    ## STAGGERED grid
    onsrc,xsrc,ysrc,zsrc = sourceboxlocgrad!(tt,vel,src,grd; staggeredgrid=true )
    # receivers
    # lambdaold!!!!
    onarec = recboxlocgrad!(tt,lambdaold,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=true)
 
    ##============================================================================

    iswe = [2 ntx-1  1; 2 ntx-1  1; 2 ntx-1  1; 2 ntx-1  1;
            ntx-1 2 -1; ntx-1 2 -1; ntx-1 2 -1; ntx-1 2 -1 ] 
    jswe = [2 nty-1  1; 2 nty-1  1; nty-1 2 -1; nty-1 2 -1;
            2 nty-1  1; 2 nty-1  1; nty-1 2 -1; nty-1 2 -1 ]
    kswe = [2 ntz-1  1; ntz-1 2 -1; 2 ntz-1  1; ntz-1 2 -1;
            2 ntz-1  1; ntz-1 2 -1; 2 ntz-1  1; ntz-1 2 -1 ]

    ##=============================================================================    
  
    lambdanew=copy(lambdaold)
   
    #############################
    pa=0
    swedifference = 100*epsilon
    while swedifference>epsilon 
        pa +=1
        lambdaold[:,:,:] =  lambdanew

        for swe=1:8

            for k=kswe[swe,1]:kswe[swe,3]:kswe[swe,2]            
                for j=jswe[swe,1]:jswe[swe,3]:jswe[swe,2]
                    for i=iswe[swe,1]:iswe[swe,3]:iswe[swe,2]
                        
                        if onsrc[i,j,k]==true # in the box containing the src

                            aback,aforw,bback,bforw,cback,cforw = adjderivonsource(tt,onsrc,i,j,k,xinit,yinit,zinit,dh,xsrc,ysrc,zsrc)
                            
                        else

                            ## along x
                            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
                            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
                            ## along y
                            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
                            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
                            ## along z
                            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
                            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
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
                        
                        ##-------------------------------------------

                        numer =
                            (abackplus * lambdanew[i-1,j,k] - aforwminus * lambdanew[i+1,j,k]) / dh +
                            (bbackplus * lambdanew[i,j-1,k] - bforwminus * lambdanew[i,j+1,k]) / dh +
                            (cbackplus * lambdanew[i,j,k-1] - cforwminus * lambdanew[i,j,k+1]) / dh
                        
                        denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
                            (cforwplus - cbackminus)/dh
                        
                        ###################################################################
                        ###################################################################

                        # if (i<6 && j==2 && k==2 && pa==1)
                        #     println("$i,$j,$k numer, denom $numer $denom")
                        # end
                            
                        # if( abs(denom)<=1e-9 ) 
                        #     println("\n$i,$j: (abs(denom)<=1e-9)  ")
                        #     @show swe,i,j,k,numer,denom,onsrc[i,j,k],aback,aforw,bback,bforw,cback,cforw
                        #     @show aforwplus,aforwminus,abackplus,abackminus,cbackplus,cbackminus
                        #     @show bforwplus,bforwminus,bbackplus,bbackminus,cbackplus,cbackminus
                        #     @show tt[i,j,k]#,thx,thy,thz
                        # end
                        

                        ####################################################
                        
                        if onarec[i,j,k]==true                            
                            #gradT = sqrt( (tt[i+1,j]-tt[i,j])^2 + (tt[i,j+1]-tt[i,j])^2 )
                            # numer2 = numer + dtonarec[i,j,k]
                            # lambdaprop = numer2/denom 
                            # lambdanew[i,j,k] = min(lambdaold[i,j,k],lambdaprop                                                   )
                            lambdanew[i,j,k] = lambdaold[i,j,k]
                        else 
                            lambdaprop = numer/denom
                            lambdanew[i,j,k] = min(lambdaold[i,j,k],lambdaprop)
                        end
  
                    end
                end
            end
        end
        ##################################################
        # println(">>>>err=',NP.sum(abs(lambdanew-lambdaold)),'\n'
        # print 'max old new',lambdanew.max(),lambdaold.max()
        swedifference = sum(abs.(lambdanew.-lambdaold))
        ##################################################
    end

    ## STAGGERED GRID, so lambdanew[1:end-1,1:end-1]... any better idea? Interpolation?
    # gradadj = -lambdanew[1:end-1,1:end-1,1:end-1]./vel.^3
    # Average grad at the center of cells (velocity cells -STAGGERED GRID)
    averlambda = ( lambdanew[1:end-1, 1:end-1, 1:end-1].+
                   lambdanew[2:end,   1:end-1, 1:end-1].+
                   lambdanew[1:end-1, 2:end,   1:end-1].+
                   lambdanew[2:end,   2:end,   1:end-1].+
                   lambdanew[1:end-1, 1:end-1, 2:end].+
                   lambdanew[2:end,   1:end-1, 2:end].+
                   lambdanew[1:end-1, 2:end,   2:end].+
                   lambdanew[2:end,   2:end,   2:end] ) ./ 8.0
    gradadj = -averlambda./vel.^3
        
    return gradadj
end


#############################################################################

"""
$(TYPEDSIGNATURES)
"""
function eikgrad_FMM_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                         src::Vector{Float64},rec::Array{Float64,2},
                         grd::Grid3DCart,pickobs::Vector{Float64},
                         ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(3,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel).+1)
    ##nx,ny,nz = size(ttime)

    lambda = zeros((grd.ntx,grd.nty,grd.ntz)) 
 
    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    ntx = grd.ntx
    nty = grd.nty
    ntz = grd.ntz
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    zinit = grd.zinit

    ## source
    ## source
    ## STAGGERED grid
    onsrc,xsrc,ysrc,zsrc = sourceboxlocgrad!(tt,vel,src,grd, staggeredgrid=true )
    ## receivers
    onarec = recboxlocgrad!(tt,lambda,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=true)
    
    ######################################################
    #-------------------------------
    ## init FMM 
    neigh = SA[1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]

    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,ntx,nty,ntz)
    status[:,:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :, : ] .= 2 ## set to accepted on boundary
    status[end,:, : ] .= 2 ## set to accepted on boundary
    status[:, 1, : ] .= 2 ## set to accepted on boundary
    status[:, end,: ] .= 2 ## set to accepted on boundary
    status[:, :, 1 ] .= 2 ## set to accepted on boundary
    status[:, :, end] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nx,nty),find(status.==2))
    ijkrec = findall(status.==2)
    irec = [x[1] for x in ijkrec]
    jrec = [x[2] for x in ijkrec]
    krec = [x[3] for x in ijkrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=ntx*nty*ntz
    #bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))
    bheap = init_maxheap(Nmax)

    ## conversion cart to lin indices, old sub2ind
    linid_ntxntyntz = LinearIndices((ntx,nty,ntz))
    ## conversion lin to cart indices, old sub2ind
    cartid_ntxntyntz = CartesianIndices((ntx,nty,ntz))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:6 ## six potential neighbors
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            k = krec[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For the adjoint SKIP BORDERS!!!
            if (i>ntx-1) || (i<2) || (j>nty-1) || (j<2) || (k>ntz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far

                # get handle
                #han = sub2ind((ntx,nty),i,j)
                han = linid_ntxntyntz[i,j,k]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = ntx*nty*ntz
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh[]<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 3D indices from handle
        cijka = cartid_ntxntyntz[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted     
        status[ia,ja,ka] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA!(tt,status,onsrc,onarec,dh,
                    xinit,yinit,zinit,xsrc,ysrc,zsrc,lambda,ia,ja,ka)
        
        ## try all neighbors of newly accepted point
        for ne=1:6 ## six potential neighbors
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
 
            ## if the point is out of bounds skip this iteration
            ## For the adjoint SKIP BORDERS!!!
            if (i>ntx-1) || (i<2) || (j>nty-1) || (j<2) || (k>ntz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far, active

                # get handle
                #han = sub2ind((ntx,nty),i,j)
                han = linid_ntxntyntz[i,j,k] #i+ntx*(j-1)
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1
                
                ## NO NEED to update in narrow band because
                ##   we already know the time tt everywhere...
            end
        end
    end

    ## STAGGERED GRID, so lambdanew[1:end-1,1:end-1]... any better idea? Interpolation?
    # gradadj = -lambdanew[1:end-1,1:end-1]./vel.^3
    # Average grad at the center of cells (velocity cells -STAGGERED GRID)
    averlambda = ( lambda[1:end-1, 1:end-1, 1:end-1].+
                   lambda[2:end,   1:end-1, 1:end-1].+
                   lambda[1:end-1, 2:end,   1:end-1].+
                   lambda[2:end,   2:end,   1:end-1].+
                   lambda[1:end-1, 1:end-1, 2:end].+
                   lambda[2:end,   1:end-1, 2:end].+
                   lambda[1:end-1, 2:end,   2:end].+
                   lambda[2:end,   2:end,   2:end] ) ./ 8.0
    gradadj = -averlambda./vel.^3
        
    return gradadj
end # eikadj_FMM_SINGLESRC  


#############################################################

"""
$(TYPEDSIGNATURES)
"""
function calcLAMBDA!(tt::Array{Float64,3},status::Array{Int64,3},onsrc::Array{Bool,3},onarec::Array{Bool,3},
                     dh::Float64,xinit::Float64,yinit::Float64,zinit::Float64,xsrc::Float64,ysrc::Float64,zsrc::Float64,
                     lambda::Array{Float64,3},i::Int64,j::Int64,k::Int64)

    
    if onsrc[i,j,k]==true # in the box containing the src

        aback,aforw,bback,bforw,cback,cforw = adjderivonsource(tt,onsrc,i,j,k,xinit,yinit,zinit,dh,xsrc,ysrc,zsrc)

    else # not on src
        
        ## along x
        aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
        aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
        ## along y
        bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
        bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
        ## along z
        cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
        cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
        
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
    
    ##-------------------------------------------
    ## make SURE lambda was INITIALIZED TO ZERO
    ## Leung & Qian, 2006
    numer =
        (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / dh +
        (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / dh +
        (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / dh
    
    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
        (cforwplus - cbackminus)/dh

    lambda[i,j,k] = numer/denom

    return
end


#############################################################################


"""
$(TYPEDSIGNATURES)
"""
function eikgrad_FMM_hiord_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                         src::Vector{Float64},rec::Array{Float64,2},
                         grd::Grid3DCart,pickobs::Vector{Float64},
                         ttpicks::Vector{Float64},stdobs::Vector{Float64})

    @assert size(src)==(3,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel)) ## SAME SIZE!!
    nx,ny,nz = grd.nx,grd.ny,grd.nz  ##size(ttime)

    lambda = zeros((grd.nx,grd.ny,grd.nz)) 
 
    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    # ntx = nx ##grd.ntx
    # nty = ny ##grd.nty
    # ntz = nz ##grd.nty
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    zinit = grd.zinit
    
    ## source
    ## regular grid
    onsrc,xsrc,ysrc,zsrc = sourceboxlocgrad!(tt,vel,src,grd, staggeredgrid=false )
    ## receivers
    onarec = recboxlocgrad!(tt,lambda,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=false)

    ######################################################
    #-------------------------------
    ## init FMM 
    neigh = SA[1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]
 
    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,nx,ny,nz)
    status[:,:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :, : ] .= 2 ## set to accepted on boundary
    status[nx,:, : ] .= 2 ## set to accepted on boundary
    status[:, 1, : ] .= 2 ## set to accepted on boundary
    status[:, ny,: ] .= 2 ## set to accepted on boundary
    status[:, :, 1 ] .= 2 ## set to accepted on boundary
    status[:, :, nz] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nx,ny),find(status.==2))
    ijkrec = findall(status.==2)
    irec = [x[1] for x in ijkrec]
    jrec = [x[2] for x in ijkrec]
    krec = [x[3] for x in ijkrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=nx*ny*nz
    #bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))
    bheap = init_maxheap(Nmax)

    ## conversion cart to lin indices, old sub2ind
    linid_nxnynz = LinearIndices((nx,ny,nz))
    ## conversion lin to cart indices, old sub2ind
    cartid_nxnynz = CartesianIndices((nx,ny,nz))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:6 ## six potential neighbors
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            k = jrec[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nx-1) || (i<2) || (j>ny-1) || (j<2) || (k>nz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far

                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxnynz[i,j,k]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = nx*ny*nz
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh[]<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 3D indices from handle
        cijka = cartid_nxnynz[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted     
        status[ia,ja,ka] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA_hiord!(tt,status,onsrc,onarec,dh,
                            xinit,yinit,zinit,xsrc,ysrc,zsrc,lambda,ia,ja,ka)
        
        ## try all neighbors of newly accepted point
        for ne=1:6 ## six potential neighbors
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nx-1) || (i<2) || (j>ny-1) || (j<2) || (k>nz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far, active

                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxnynz[i,j,k] #i+nx*(j-1)
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

"""
$(TYPEDSIGNATURES)
"""
function isoutrange(ib::Int64,jb::Int64,kb::Int64,nx::Int64,ny::Int64,nz::Int64)
    isoutb1st = false
    isoutb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<=1 || ib>=nx || jb<=1 || jb>=ny || kb<=1 || kb>=nz
        isoutb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<=2 || ib>=nx-1 || jb<=2 || jb>=ny-1 || kb<=2 || kb>=nz-1
        isoutb2nd = true
    end
    return isoutb1st,isoutb2nd
end

##====================================================================##

"""
$(TYPEDSIGNATURES)
"""
function calcLAMBDA_hiord!(tt::Array{Float64,3},status::Array{Int64},
                           onsrc::Array{Bool,3},onarec::Array{Bool,3},
                           dh::Float64,xinit::Float64,yinit::Float64,zinit::Float64,
                           xsrc::Float64,ysrc::Float64,zsrc::Float64,lambda::Array{Float64,3},
                           i::Int64,j::Int64,k::Int64)

    nx,ny,nz = size(lambda)
    isout1st,isout2nd = isoutrange(i,j,k,nx,ny,nz)

    if onsrc[i,j,k]==true # in the box containing the src

        aback,aforw,bback,bforw,cback,cforw = adjderivonsource(tt,onsrc,i,j,k,xinit,yinit,zinit,dh,xsrc,ysrc,zsrc)

    else # not on src
        
        ##
        ## Central differences in Leung & Qian, 2006 scheme!
        ##  
        # if !isout2nd
        #     ##
        #     ## Fourth order central diff O(h^4); a,b computed in between nodes of velocity
        #     ##  dh = 2*dist because distance from edge of cells is dh/2
        #     ## 12*h => 12*dh/2 = 6*dh
        #     aback = -( -tt[i+1,j,k]+8.0*tt[i,j,k]-8.0*tt[i-1,j,k]+tt[i-2,j,k] )/(6.0*dh)
        #     aforw = -( -tt[i+2,j,k]+8.0*tt[i+1,j,k]-8.0*tt[i,j,k]+tt[i-1,j,k] )/(6.0*dh)
        #     bback = -( -tt[i,j+1,k]+8.0*tt[i,j,k]-8.0*tt[i,j-1,k]+tt[i,j-2,k] )/(6.0*dh)
        #     bforw = -( -tt[i,j+2,k]+8.0*tt[i,j+1,k]-8.0*tt[i,j,k]+tt[i,j-1,k] )/(6.0*dh)
        #     cback = -( -tt[i,j,k+1]+8.0*tt[i,j,k]-8.0*tt[i,j,k-1]+tt[i,j,k-2] )/(6.0*dh)
        #     cforw = -( -tt[i,j,k+2]+8.0*tt[i,j,k+1]-8.0*tt[i,j,k]+tt[i,j,k-1] )/(6.0*dh)
            
        # else
            ##
            ## Central diff (Leung & Qian, 2006); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ##  2*h => 2*dh/2 = dh
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh

        #end         
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
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
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
        (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / dh +
        (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / dh +
        (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / dh
    
    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
        (cforwplus - cbackminus)/dh

    lambda[i,j,k] = numer/denom
   
    ##================================================
    # try once more to fix denom==0.0
    if denom==0.0

        # set ttime on central pixel as the mean of neighbors
        ttmp = (tt[i+1,j,k]+tt[i-1,j,k]+tt[i,j+1,k]+tt[i,j-1,k]+tt[i,j,k+1]+tt[i,j,k-1])/6.0

        ## revert to smaller stencil
        aback = -(tttmp  -tt[i-1,j,k])/dh
        aforw = -(tt[i+1,j,k]-tttmp)/dh
        bback = -(tttmp  -tt[i,j-1,k])/dh
        bforw = -(tt[i,j+1,k]-tttmp)/dh
        cback = -(tttmp  -tt[i,j,k-1])/dh
        cforw = -(tt[i,j,k+1]-tttmp)/dh
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

        ## recompute lambda
        numer =
            (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / dh +
            (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / dh +
            (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / dh        
        denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
            (cforwplus - cbackminus)/dh
        lambda[i,j,k] = numer/denom

    end

    # if the second fix didn't work exit...
    if denom==0.0
        # @show onsrc[i,j]
        # @show aforwplus,abackminus
        # @show bforwplus,bbackminus
        error("calcLAMBDA_hiord!(): denom==0, (i,j,k)=($i,$j,$k), 2nd ord.: $(!isout2nd)")
    end

    return nothing #lambda
end


############################################################################

"""
$(TYPEDSIGNATURES)
"""
function sourceboxlocgrad!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Vector{Float64},
                           grd::Grid3DCart; staggeredgrid::Bool )

    ##########################
    ##   Init source
    ##########################
    mindistsrc = 0.01*grd.hgrid
    xsrc,ysrc,zsrc = srcpos[1],srcpos[2],srcpos[3]
    dh = grd.hgrid

    if staggeredgrid==false
        ## regular grid
        onsrc = zeros(Bool,grd.nx,grd.ny,grd.nz)
        onsrc[:,:,:] .= false
        ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
        rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)
        max_x = (grd.nx-1)*grd.hgrid+grd.xinit
        max_y = (grd.ny-1)*grd.hgrid+grd.yinit
        max_z = (grd.nz-1)*grd.hgrid+grd.zinit
        
    elseif staggeredgrid==true
        ## STAGGERED grid
        onsrc = zeros(Bool,grd.ntx,grd.nty,grd.ntz)
        onsrc[:,:,:] .= false
        ## grd.xinit-hgr because TIME array on STAGGERED grid
        hgr = grd.hgrid/2.0
        ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
        rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)
        max_x = (grd.ntx-1)*grd.hgrid+grd.xinit
        max_y = (grd.nty-1)*grd.hgrid+grd.yinit
        max_z = (grd.ntz-1)*grd.hgrid+grd.zinit
    end

    halfg = 0.0
    src_on_nodeedge = false

    ## loop to make sure we don't accidentally move the src to another node/edge
    while (sqrt(rx^2+ry^2+rz^2)<=mindistsrc) || (abs(rx)<=mindistsrc) || (abs(ry)<=mindistsrc) || (abs(rz)<=mindistsrc)
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
        if zsrc < max_z-clo
            zsrc = zsrc+sft
        else #(make sure it's not already at the bottom z)
            zsrc = zsrc-sft
        end

        # print("new time at src:  $(tt[ix,iy]) ")
        ## recompute parameters related to position of source
        if staggeredgrid==false
            ## regular grid
            ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
            rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
            ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
            rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)        
        elseif staggeredgrid==true
            ## STAGGERED grid
            hgr = grd.hgrid/2.0
            ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
            rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
            ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
            rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)        
        end
    end

    ## To avoid singularities, the src can only be inside a box,
    ##  not on a grid node or edge... see above. So we need to have
    ##  always 4 nodes where onsrc is true
    if (rx>=halfg) & (ry>=halfg) & (rz>=halfg)
        onsrc[ix:ix+1,iy:iy+1,iz:iz+1] .= true
    elseif (rx<halfg) & (ry>=halfg) & (rz>=halfg)
        onsrc[ix-1:ix,iy:iy+1,iz:iz+1] .= true
    elseif (rx<halfg) & (ry<halfg) & (rz>=halfg)
        onsrc[ix-1:ix,iy-1:iy,iz:iz+1] .= true
    elseif (rx>=halfg) & (ry<halfg) & (rz>=halfg)
        onsrc[ix:ix+1,iy-1:iy,iz:iz+1] .= true

    elseif (rx>=halfg) & (ry>=halfg) & (rz<halfg)
        onsrc[ix:ix+1,iy:iy+1,iz-1:iz] .= true
    elseif (rx<halfg) & (ry>=halfg) & (rz<halfg)
        onsrc[ix-1:ix,iy:iy+1,iz-1:iz] .= true
    elseif (rx<halfg) & (ry<halfg) & (rz<halfg)
        onsrc[ix-1:ix,iy-1:iy,iz-1:iz] .= true
    elseif (rx>=halfg) & (ry<halfg) & (rz<halfg)
        onsrc[ix:ix+1,iy-1:iy,iz-1:iz] .= true
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
            if staggeredgrid==false
                ## regular grid
                xp = (i-1)*grd.hgrid+grd.xinit
                yp = (j-1)*grd.hgrid+grd.yinit
                zp = (k-1)*grd.hgrid+grd.zinit
                ii = Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
                jj = Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
                kk = Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
            elseif staggeredgrid==true
                ## STAGGERED grid
                xp = (i-1)*grd.hgrid+grd.xinit-hgr
                yp = (j-1)*grd.hgrid+grd.yinit-hgr
                zp = (k-1)*grd.hgrid+grd.zinit-hgr
                ii = i-1 ##Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
                jj = j-1 ##Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
                kk = k-1 ##Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
            end
        end
    end#

    return onsrc,xsrc,ysrc,zsrc
end


################################################################333


############################################################################

"""
$(TYPEDSIGNATURES)
"""
function recboxlocgrad!(ttime::Array{Float64,3},lambda::Array{Float64,3},ttpicks::Vector{Float64},grd::Grid3DCart,
                        rec::Array{Float64,2},pickobs::Vector{Float64},stdobs::Vector{Float64}; staggeredgrid::Bool)

    ##########################
    ## Init receivers
    ##########################

    if staggeredgrid==false
        ## regular grid
        onarec = zeros(Bool,grd.nx,grd.ny,grd.nz)
        nxmax = grd.nx
        nymax = grd.ny
        nzmax = grd.nz
    elseif staggeredgrid==true
        ## STAGGERED grid
        onarec = zeros(Bool,grd.ntx,grd.nty,grd.ntz)
        nxmax = grd.ntx
        nymax = grd.nty
        nzmax = grd.ntz
        hgr = grd.hgrid/2.0
    end
    onarec[:,:,:] .= false
    nrec=size(rec,1)
    
    ## init receivers
    nrec=size(rec,1)
    fisttimereconbord = true
    for r=1:nrec
        if staggeredgrid==false
            i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,3],grd.xinit,grd.yinit,grd.zinit,grd.hgrid)
        elseif staggeredgrid==true
            i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,3],grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid)
        end
        if (i==1) || (i==nxmax) || (j==1) || (j==nymax) || (k==1) || (k==nzmax)
            ## Receivers on the boundary of model
            ## (n⋅∇T)λ = Tcalc - Tobs
            if i==1
                # forward diff
                n∇T = - (ttime[2,j,k]-ttime[1,j,k])/grd.hgrid 
            elseif i==nxmax
                # backward diff
                n∇T = (ttime[end,j,k]-ttime[end-1,j,k])/grd.hgrid 
            elseif j==1
                # forward diff
                n∇T = - (ttime[i,2,k]-ttime[i,1,k])/grd.hgrid 
            elseif j==nymax
                # backward diff
                n∇T = (ttime[i,end,k]-ttime[i,end-1,k])/grd.hgrid 
            elseif k==1
                # forward diff
                n∇T = - (ttime[i,j,2]-ttime[i,j,1])/grd.hgrid 
            elseif k==nzmax
                # backward diff
                n∇T = (ttime[i,j,end]-ttime[i,j,end-1])/grd.hgrid 
            end
            onarec[i,j,k] = true
            #############################################
            ##  FIX ME: receivers on the border...     ## <<<<===================#####
            #############################################
            if fisttimereconbord
                @warn(" Receiver(s) on border of model, \n still untested, spurious results may be encountered.")
                @show i,j,k,nxmax,nymax,nzmax
                fisttimereconbord=false
            end
            ## Taking into account Neumann boundary condition
            ## lambda[i,j] = (ttpicks[r]-pickobs[r])/(n∇T * stdobs[r]^2)
            ## NOT taking into account Neumann boundary condition
            lambda[i,j,k] =  (ttpicks[r]-pickobs[r])/stdobs[r]^2

        else
            ## Receivers within the model
            onarec[i,j,k] = true
            lambda[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs[r]^2
        end
    end
  
    return onarec
end

###################################################################################################


"""
$(TYPEDSIGNATURES)

Calculate the gradient for some requested sources 
"""
function calcgradsomesrc3D(vel::Array{Float64,3},xθsrc::Array{Float64,2},coordrec::Vector{Matrix{Float64}},
                           grd::Grid3DSphere,stdobs::Vector{Vector{Float64}},pickobs1::Vector{Vector{Float64}} )

    nr,nθ,nφ=size(vel)
    nsrc = size(xθsrc,1)
    grad1 = zeros(nr,nθ,nφ)

    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        curnrec = size(coordrec[s],1) 
        ttpicks1 = zeros(curnrec)

        ###########################################
        ## calc ttime
 
        ## if adjalgo=="gradFMM_hiord"
        ttgrpsrc = ttFMM_hiord(vel,xθsrc[s,:],grd)

        ## ttime at receivers
        for i=1:curnrec
            ttpicks1[i] = trilinear_interp_sph( ttgrpsrc,grd,coordrec[s][i,1],
                                                coordrec[s][i,2],coordrec[s][i,3] )
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        ## if adjalgo=="gradFMM_hiord"
        grad1 += eikgrad_FMM_hiord_SINGLESRC( ttgrpsrc,vel,xθsrc[s,:],coordrec[s],
                                              grd,pickobs1[s],ttpicks1,stdobs[s])
    end
    
    return grad1
end

############################################################################
"""
$(TYPEDSIGNATURES)
"""
function sourceboxlocgrad_sph!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Vector{Float64},
                               grd::Grid3DSphere )

    ##########################
    ##   Init source
    ##########################
    mindistsrc = 0.01*grd.Δr
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
        if rsrc < max_r-0.02*grd.Δr
            rsrc = rsrc+0.01*grd.Δr
        else #(make sure it's not already at the bottom y)
            rsrc = rsrc-0.01*grd.Δr
        end
        if θsrc < max_θ-0.02*grd.Δθ
            θsrc = θsrc+0.01*grd.Δθ
        else #(make sure it's not already at the bottom y)
            θsrc = θsrc-0.01*grd.Δθ
        end
        if φsrc < max_φ-0.02*grd.Δφ
            φsrc = φsrc+0.01*grd.Δφ
        else #(make sure it's not already at the bottom z)
            φsrc = φsrc-0.01*grd.Δφ
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

"""
$(TYPEDSIGNATURES)
"""
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
    fisttimereconbord = true
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
            lambda[i,j,k] =  (ttpicks[r]-pickobs[r])/stdobs[r]^2

        else
            ## Receivers within the model
            onarec[i,j,k] = true
            lambda[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs[r]^2
        end
    end
  
    return onarec
end

###################################################################################################

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
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
    for l=1:naccinit 
        for ne=1:6 ## six potential neighbors
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
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
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
        for ne=1:6 ## six potential neighbors
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

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
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

    ##================================================
    # try once more to fix denom==0.0
    if denom==0.0

        # set ttime on central pixel as the mean of neighbors
        tttmp = (tt[i+1,j,k]+tt[i-1,j,k]+tt[i,j+1,k]+tt[i,j-1,k]+tt[i,j,k+1]+tt[i,j,k-1])/6.0

        ## revert to smaller stencil
        aback = -(tttmp  -tt[i-1,j,k])/deltar
        aforw = -(tt[i+1,j,k]-tttmp)/deltar
        bback = -(tttmp  -tt[i,j-1,k])/(grd.r[i]*deltaθ)
        bforw = -(tt[i,j+1,k]-tttmp)/(grd.r[i]*deltaθ)
        cback = -(tttmp  -tt[i,j,k-1])/(grd.r[i]*sind(grd.θ[j])*deltaφ)
        cforw = -(tt[i,j,k+1]-tttmp)/(grd.r[i]*sind(grd.θ[j])*deltaφ)
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

        ## recompute lambda
        numer =
            (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / deltar +
            (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / (grd.r[i]*deltaθ) +
            (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / (grd.r[i]*sind(grd.θ[j])*deltaφ)
        denom = (aforwplus - abackminus)/deltar + (bforwplus - bbackminus)/(grd.r[i]*deltaθ) +
            (cforwplus - cbackminus)/(grd.r[i]*sind(grd.θ[j])*deltaφ)
        lambda[i,j,k] = numer/denom

    end
    
    if denom==0.0
        # @show onsrc[i,j]
        # @show aforwplus,abackminus
        # @show bforwplus,bbackminus
        error("calcLAMBDA_hiord!(): denom==0, (i,j,k)=($i,$j,$k), 2nd ord.: $(!isout2nd)")
    end

    return nothing #lambda
end


##################################################
