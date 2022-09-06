

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################

##########################################################################

"""
$(TYPEDSIGNATURES)

Calculate traveltime for 2D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 2D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model
- `extraparams` (optional) : a struct containing some "extra" parameters, namely
    * `refinearoundsrc`: whether to perform a refinement of the grid around the source location
    * `allowfixsqarg`: brute-force fix negative saqarg

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime2D(vel::Array{Float64,2},grd::GridEik2D,coordsrc::Array{Float64,2},
                      coordrec::Vector{Array{Float64,2}} ; returntt::Bool=false,
                      extraparams::Union{ExtraParams,Nothing}=nothing  )
        
    if extraparams==nothing
        extraparams =  ExtraParams()
    end

    if typeof(grd)==Grid2D
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
    end
    if simtype==:cartesian
        n1,n2 = grd.nx,grd.ny 
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
    
    nsrc = size(coordsrc,1)
    ttpicks = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicks[i] = zeros(curnrec)
    end

    ## calculate how to subdivide the srcs among the workers
    #nsrc = size(coordsrc,1)
   
    if returntt
        # return traveltime array and picks at receivers
        ttime = zeros(n1,n2,nsrc)
    end

    
    if extraparams.parallelkind==:distribmem
        ##====================================
        ## Distributed memory
        ##====================
        nw = nworkers()
        grpsrc = distribsrcs(nsrc,nw)
        nchu = size(grpsrc,1)
        ## array of workers' ids
        wks = workers()

        if returntt 
            # return traveltime picks at receivers and at all grid points 
            @sync for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                                        vel,coordsrc[igrs,:],
                                                                        coordrec[igrs],grd,extraparams,
                                                                        returntt=returntt)
            end
        else
            # return ONLY traveltime picks at receivers 
            @sync for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                        vel,coordsrc[igrs,:],
                                                        coordrec[igrs],grd,extraparams,
                                                        returntt=returntt)
            end
        end

        
    elseif extraparams.parallelkind==:sharedmem
        ##====================================
        ## Shared memory
        ##====================
        nth = Threads.nthreads()
        grpsrc = distribsrcs(nsrc,nth)
        nchu = size(grpsrc,1)      

        if returntt            
            # return both traveltime picks at receivers and at all grid points
            Threads.@threads for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                ttime[:,:,igrs],ttpicks[igrs] = ttforwsomesrc2D(vel,view(coordsrc,igrs,:),
                                                                  view(coordrec,igrs),grd,extraparams,
                                                                  returntt=returntt )
            end

        else
            # return ONLY traveltime picks at receivers
            Threads.@threads for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                ttpicks[igrs] = ttforwsomesrc2D(vel,view(coordsrc,igrs,:),
                                                view(coordrec,igrs),grd,extraparams,
                                                returntt=returntt )
            end
        end


    elseif extraparams.parallelkind==:serial
        ##====================================
        ## Serial run
        ##====================
        if returntt            
            # return both traveltime picks at receivers and at all grid points
            ttime[:,:,:],ttpicks = ttforwsomesrc2D(vel,coordsrc,coordrec,grd,extraparams,
                                                           returntt=returntt )
        else
            # return ONLY traveltime picks at receivers
            ttpicks = ttforwsomesrc2D(vel,coordsrc,coordrec,grd,extraparams,
                                      returntt=returntt )
        end

    end
    
    if returntt 
        return ttpicks,ttime
    end
    return ttpicks
end

###########################################################################
"""
$(TYPEDSIGNATURES)

  Compute the forward problem for a group of sources.
"""
function ttforwsomesrc2D(vel::Array{Float64,2},coordsrc::AbstractArray{Float64,2},
                         coordrec::AbstractVector{Array{Float64,2}},grd::GridEik2D,
                         extrapars::ExtraParams ; returntt::Bool=false )
    
    if typeof(grd)==Grid2D
        #simtype = :cartesian
        n1,n2 = grd.nx,grd.ny
    elseif typeof(grd)==Grid2DSphere
        #simtype = :spherical
        n1,n2 = grd.nr,grd.nθ
    end

    nsrc = size(coordsrc,1)
    
    ttpicksGRPSRC = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicksGRPSRC[i] = zeros(curnrec)
    end

    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars = FMMvars2D(n1,n2,refinearoundsrc=extrapars.refinearoundsrc,
                        allowfixsqarg=extrapars.allowfixsqarg)

    ## pre-allocate discrete adjoint variables
    ##  No adjoint calculations
    adjvars = nothing

    if returntt
        ttGRPSRC = zeros(n1,n2,nsrc)
    end

    ## group of pre-selected sources
    for s=1:nsrc    
        ##
        ttFMM_hiord!(fmmvars,vel,view(coordsrc,s,:),grd,adjvars)

        ## interpolate at receivers positions
        for i=1:size(coordrec[s],1)
            #  view !!
            ttpicksGRPSRC[s][i] = bilinear_interp( fmmvars.ttime,grd,view(coordrec[s],i,:) )

        end

        if returntt
            ## Compute traveltime and interpolation at receivers in one go for parallelization
            ttGRPSRC[:,:,s] .= fmmvars.ttime
        end
    end

    if returntt 
        return ttGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC 
end


#################################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 2D using traditional stencils on regular grid. 
"""
function ttFMM_hiord!(fmmvars::FMMvars2D,vel::Array{Float64,2},src::AbstractVector{Float64},grd::GridEik2D,
                      adjvars::Union{AdjointVars2D,Nothing} )
                     
    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end

    ## Sizes
    if typeof(grd)==Grid2D
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
    end
    if simtype==:cartesian
        n1,n2 = grd.nx,grd.ny 
    elseif simtype==:spherical
        n1,n2 = grd.nr,grd.nθ
    end
    n12 = n1*n2
    
    ## 
    ## Time array
    ##
    inittt = 1e30
    #ttime = Array{Float64}(undef,n1,n2)
    fmmvars.ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    #status = Array{Int64}(undef,n1,n2)
    fmmvars.status[:,:] .= 0   ## set all to far
    ##
    ptij = MVector(0,0)
    # derivative codes
    idD = MVector(0,0)

    ##======================================================
    if dodiscradj
        ##
        ##  discrete adjoint: init stuff
        ## 
        # idxconv = MapOrderGridFMM2D(n1,n2)
        # fmmord = VarsFMMOrder2D(n1,n2)
        # codeDxy = zeros(Int64,n12,2)
        adjvars.fmmord.vecDx.lastrowupdated[] = 0
        adjvars.fmmord.vecDy.lastrowupdated[] = 0
        # for safety, zeroes the traveltime
        adjvars.fmmord.ttime[:] .= 0.0
        # for safety, zeroes the codes for derivatives
        adjvars.codeDxy[:,:] .= 0

        # init row stuff
        colindsonsrc = MVector(0)
        colvalsonsrc = SA[1.0]
    end    
    ##======================================================
    
    ##########################################
    
    if fmmvars.refinearoundsrc
        ##---------------------------------
        ## 
        ## Refinement around the source      
        ##
        if dodiscradj
            ijsrc = ttaroundsrc!(fmmvars,vel,src,grd,inittt,dodiscradj=dodiscradj,
                                 idxconv=adjvars.idxconv,fmmord=adjvars.fmmord)
        else
            ijsrc = ttaroundsrc!(fmmvars,vel,src,grd,inittt)
        end

        ## number of accepted points       
        naccinit = size(ijsrc,2)

        ##======================================================
        if dodiscradj
            ## 
            ## DISCRETE ADJOINT WORKAROUND FOR DERIVATIVES
            ##

            # How many initial points to skip, considering them as "onsrc"?
            skipnptsDxy = 4
            
            ## pre-compute some of the mapping between fmm and orig order
            for i=1:naccinit
                ifm = adjvars.idxconv.lfmm2grid[i]
                adjvars.idxconv.lgrid2fmm[ifm] = i
            end

      
            # loop on points "on" source
            for l=1:naccinit

                if l<=skipnptsDxy
                    #################################################################
                    # Here we store a 1 in the diagonal because we are on a source node...
                    #  store arrival time for first points in FMM order
                    colindsonsrc[1] = l
                    nnzcol = 1
                    addrowCSRmat!(adjvars.fmmord.vecDx,l,colindsonsrc,colvalsonsrc,nnzcol)
                    addrowCSRmat!(adjvars.fmmord.vecDy,l,colindsonsrc,colvalsonsrc,nnzcol)
                    # addentry!(adjvars.fmmord.vecDx,l,l,1.0)   
                    # addentry!(adjvars.fmmord.vecDy,l,l,1.0)     
                    #################################################################

                else

                    ## "reconstruct" derivative stencils from known FMM order and arrival times
                    derivaroundsrcfmm2D!(l,adjvars.idxconv,idD)

                    if idD==[0,0]
                        #################################################################
                        # Here we store a 1 in the diagonal because we are on a source node...
                        #  store arrival time for first points in FMM order
                        colindsonsrc[1] = l
                        nnzcol = 1
                        addrowCSRmat!(adjvars.fmmord.vecDx,l,colindsonsrc,colvalsonsrc,nnzcol)
                        addrowCSRmat!(adjvars.fmmord.vecDy,l,colindsonsrc,colvalsonsrc,nnzcol)
                        # addentry!(adjvars.fmmord.vecDx,l,l,1.0)
                        # addentry!(adjvars.fmmord.vecDy,l,l,1.0)
                        #################################################################
                    else
                        l_fmmord = adjvars.idxconv.lfmm2grid[l]
                        adjvars.codeDxy[l_fmmord,:] .= idD
                    end

                end
            end

        end # dodiscradj
        ##======================================================

  
    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        #println("ttFMM_hiord(): NO refinement around the source! ")

        ## source location, etc.      
        ## REGULAR grid
        if simtype==:cartesian
            ijsrc = sourceboxloctt!(fmmvars,vel,src,grd )
        elseif simtype==:spherical
            ijsrc = sourceboxloctt_sph!(fmmvars,vel,src,grd )
        end
        ## number of accepted points       
        naccinit = size(ijsrc,2)        

        ##======================================================
        if dodiscradj

            # discrete adjoint: first visited points in FMM order
            ttfirstpts = Vector{Float64}(undef,naccinit)
            for l=1:naccinit
                i,j = ijsrc[1,l],ijsrc[2,l]
                ttij = fmmvars.ttime[i,j]
                # store initial points' FMM order
                ttfirstpts[l] = ttij
            end
            # sort according to arrival time
            spidx = sortperm(ttfirstpts)
            # store the first accepted points' order
            for l=1:naccinit
                # ordered index
                p = spidx[l]
                # go from cartesian (i,j) to linear
                i,j = ijsrc[1,l],ijsrc[2,l]
                adjvars.idxconv.lfmm2grid[l] = cart2lin2D(i,j,n1)

                ######################################
                # store arrival time for first points in FMM order
                ttgrid = fmmvars.ttime[i,j]
                ## The following to avoid a singular upper triangular matrix in the
                ##  adjoint equation. Otherwise there will be a row of only zeros in the LHS.
                if ttgrid==0.0
                    adjvars.fmmord.ttime[l] = 10.0*eps()
                    @warn("Source is exactly on a node, spurious results may appear. Work is in progress to fix the problem.\n Set smoothgradsourceradius>0 to mitigate the issue")
                else
                    adjvars.fmmord.ttime[l] = ttgrid
                end

                #################################################################
                # Here we store a 1 in the diagonal because we are on a source node...
                #  store arrival time for first points in FMM order
                #
                colindsonsrc[1] = l
                nnzcol = 1
                addrowCSRmat!(adjvars.fmmord.vecDx,l,colindsonsrc,colvalsonsrc,nnzcol)
                addrowCSRmat!(adjvars.fmmord.vecDy,l,colindsonsrc,colvalsonsrc,nnzcol)
                # addentry!(adjvars.fmmord.vecDx,l,l,1.0)
                # addentry!(adjvars.fmmord.vecDy,l,l,1.0)
                #
                #################################################################

            end
        end # if dodiscradj
        ##======================================================
      
    end # if refinearoundsrc
    ###################################################### 

    #-------------------------------
    ## init FMM 
    neigh = SA[1  0;
               0  1;
              -1  0;
               0 -1]

    # ## Init the min binary heap with void arrays but max size
    # Nmax = n1*n2
    # bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 

    ##-----------------------------------------
    ## construct initial narrow band
    for l=1:naccinit ##
        
        for ne=1:4 ## four potential neighbors

            i = ijsrc[1,l] + neigh[ne,1]
            j = ijsrc[2,l] + neigh[ne,2]

            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if fmmvars.status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(fmmvars,vel,grd,i,j,idD)

                # get handle
                han = cart2lin2D(i,j,n1)
                # insert into heap
                insert_minheap!(fmmvars.bheap,tmptt,han)
                # change status, add to narrow band
                fmmvars.status[i,j]=1

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    adjvars.codeDxy[han,:] .= idD
                end
            end
        end
    end

    #-------------------------------
    ## main FMM loop
    totnpts = n1*n2
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if fmmvars.bheap.Nh[]<1
            break
        end

        han,tmptt = pop_minheap!(fmmvars.bheap)
        #ia,ja = ind2sub((n1,n2),han)
        lin2cart2D!(han,n1,ptij)
        ia,ja = ptij[1],ptij[2]
        #ja = div(han-1,n1) +1
        #ia = han - n1*(ja-1)
        # set status to accepted
        fmmvars.status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        fmmvars.ttime[ia,ja] = tmptt

        if dodiscradj
            # store the linear index of FMM order
            adjvars.idxconv.lfmm2grid[node] = cart2lin2D(ia,ja,n1)
            # store arrival time for first points in FMM order
            adjvars.fmmord.ttime[node] = tmptt  
        end
        
        ## try all neighbors of newly accepted point
        for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if fmmvars.status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(fmmvars,vel,grd,i,j,idD)
                han = cart2lin2D(i,j,n1)
                insert_minheap!(fmmvars.bheap,tmptt,han)
                # change status, add to narrow band
                fmmvars.status[i,j]=1

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    adjvars.codeDxy[han,:] .= idD
                end

            elseif fmmvars.status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord!(fmmvars,vel,grd,i,j,idD)

                # get handle
                han = cart2lin2D(i,j,n1)
                # update the traveltime for this point in the heap
                update_node_minheap!(fmmvars.bheap,tmptt,han)

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    adjvars.codeDxy[han,:] .= idD
                end

            end
        end
        ##-------------------------------
    end

    ##======================================================
    if dodiscradj

        ## pre-compute the mapping between fmm and orig order
        for i=1:n12
            ifm = adjvars.idxconv.lfmm2grid[i]
            adjvars.idxconv.lgrid2fmm[ifm] = i
        end

        # pre-determine derivative coefficients for positive codes (0,+1,+2)
        if simtype==:cartesian
            hgrid = grd.hgrid
            #allcoeff = [[-1.0/hgrid, 1.0/hgrid], [-3.0/(2.0*hgrid), 4.0/(2.0*hgrid), -1.0/(2.0*hgrid)]]
            allcoeffx = CoeffDerivCartesian( MVector(-1.0/hgrid,
                                                       1.0/hgrid), 
                                               MVector(-3.0/(2.0*hgrid),
                                                       4.0/(2.0*hgrid),
                                                       -1.0/(2.0*hgrid)) )
            # coefficients along X are the same than along Y
            allcoeffy = allcoeffx

        elseif simtype==:spherical
            Δr = grd.Δr
            ## DEG to RAD !!!!
            Δarc = [grd.r[i] * deg2rad(grd.Δθ) for i=1:grd.nr]
            # coefficients
            coe_r_1st = [-1.0/Δr  1.0/Δr]
            coe_r_2nd = [-3.0/(2.0*Δr)  4.0/(2.0*Δr) -1.0/(2.0*Δr) ]
            coe_θ_1st = [-1.0 ./ Δarc  1.0 ./ Δarc ]
            coe_θ_2nd = [-3.0./(2.0*Δarc)  4.0./(2.0*Δarc)  -1.0./(2.0*Δarc) ]
            #@show size(coe_θ_1st),size(coe_θ_2nd)

            allcoeffx = CoeffDerivSpherical2D( coe_r_1st, coe_r_2nd )
            allcoeffy = CoeffDerivSpherical2D( coe_θ_1st, coe_θ_2nd )

        end

        #@time begin 

        ##
        ## Set the derivative operators in FMM order, from source onwards
        ## 
        ##   for from adjvars.fmmord.vecDx.lastrowupdated[]+1 onwards because some rows have already
        ##   been updated above and the CSR format used here requires to add rows in sequence to
        ##   avoid expensive re-allocations
        startloop = adjvars.fmmord.vecDx.lastrowupdated[]+1
        for irow=startloop:n12
            
            # compute the coefficients for X  derivatives
            setcoeffderiv2D!(adjvars.fmmord.vecDx,irow,adjvars.idxconv,adjvars.codeDxy,allcoeffx,ptij,
                           axis=:X,simtype=simtype)
            
            # compute the coefficients for Y derivatives
            setcoeffderiv2D!(adjvars.fmmord.vecDy,irow,adjvars.idxconv,adjvars.codeDxy,allcoeffy,ptij,
                           axis=:Y,simtype=simtype)
            
        end

        #end # @time
        
        # # create the actual sparse arrays from the vectors
        # Nxnnz = adjvars.fmmord.vecDx.Nnnz[]
        # Dx_fmmord = sparse(adjvars.fmmord.vecDx.i[1:Nxnnz],
        #                    adjvars.fmmord.vecDx.j[1:Nxnnz],
        #                    adjvars.fmmord.vecDx.v[1:Nxnnz],
        #                    adjvars.fmmord.vecDx.Nsize[1], adjvars.fmmord.vecDx.Nsize[2] )

        # Nynnz = adjvars.fmmord.vecDy.Nnnz[]
        # Dy_fmmord = sparse(adjvars.fmmord.vecDy.i[1:Nynnz],
        #                    adjvars.fmmord.vecDy.j[1:Nynnz],
        #                    adjvars.fmmord.vecDy.v[1:Nynnz],
        #                    adjvars.fmmord.vecDy.Nsize[1], adjvars.fmmord.vecDy.Nsize[2] ) 
        
        ## return all the stuff for discrete adjoint computations
        #return Dx_fmmord,Dy_fmmord

        
        
    end # if dodiscradj
    ##======================================================
    
    return 
end

##########################################################################

"""
$(TYPEDSIGNATURES)

 Test if point is on borders of domain.
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

##########################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order. 
   Two-dimensional Cartesian or  spherical grid depending on the type of 'grd'.
"""
function calcttpt_2ndord!(fmmvars::FMMvars2D,vel::Array{Float64,2},
                         grd::GridEik2D,i::Int64,j::Int64,codeD::MVector{2,Int64})
    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    if typeof(grd)==Grid2D
        simtype=:cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype=:spherical
    end

    # sizes, etc.
    if simtype==:cartesian
        n1 = grd.nx
        n2 = grd.ny
        Δh = MVector(grd.hgrid,grd.hgrid)
    elseif simtype==:spherical
        n1 = grd.nr
        n2 = grd.nθ
        Δh = MVector(grd.Δr, grd.r[i]*deg2rad(grd.Δθ))
    end
    # slowness
    slowcurpt = 1.0/vel[i,j]

    ## Finite differences:
    ##
    ##  Dx_fwd = (tX[i+1] - tcur[i])/dx 
    ##  Dx_bwd = (tcur[i] - tX[i-1])/dx 
    ##

    ##################################################
    ### Solve the quadratic equation
    #  "A second-order fast marching eikonal solver"
    #    James Rickett and Sergey Fomel, 2000
    ##################################################
    
    alpha = 0.0
    beta  = 0.0
    gamma = - slowcurpt^2 ## !!!!
    HUGE = 1.0e30
    codeD[:] .= 0.0 

    ## 2 directions
    for axis=1:2
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## two sides for each direction
        for l=1:2

            ## map the 4 cases to an integer as in linear indexing...
            lax = l + 2*(axis-1)
            if lax==1 # axis==1
                ish = 1
                jsh = 0
            elseif lax==2 # axis==1
                ish = -1
                jsh = 0
            elseif lax==3 # axis==2
                ish = 0
                jsh = 1
            elseif lax==4 # axis==2
                ish = 0
                jsh = -1
            end

            ## check if on boundaries
            isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                                    
            ## 1st order
            if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
                testval1 = fmmvars.ttime[i+ish,j+jsh]

                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only
                    chosenval1 = testval1
                    use1stord = true

                    # save derivative choices
                    axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                    
                    ## 2nd order
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    if !isonb2nd && fmmvars.status[i+ish2,j+jsh2]==2 ## 2==accepted
                        testval2 = fmmvars.ttime[i+ish2,j+jsh2]
                        ## pick the lowest value of the two
                        ## <=, compare to chosenval 1, *not* 2!!
                        ## This because the direction has already been chosen
                        ##  at the line "testval1<chosenval1"
                        if testval2<=chosenval1 
                            chosenval2=testval2
                            use2ndord=true

                            # save derivative choices
                            axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)

                        else
                            chosenval2=HUGE
                            use2ndord=false # this is needed!

                            # save derivative choices
                            axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

                        end
                    end
                    
                end
            end
        end # end two sides

        ## spacing
        deltah = Δh[axis]
                
        if use2ndord && use1stord # second order
            tmpa2 = 1.0/3.0 * (4.0*chosenval1-chosenval2)
            ## curalpha: make sure you multiply only times the
            ##   current alpha for beta and gamma...
            curalpha = 9.0/(4.0 * deltah^2)
            alpha += curalpha
            beta  += ( -2.0*curalpha * tmpa2 )
            gamma += curalpha * tmpa2^2

        elseif use1stord # first order
            ## curalpha: make sure you multiply only times the
            ##   current alpha for beta and gamma...
            curalpha = 1.0/deltah^2 
            alpha += curalpha
            beta  += ( -2.0*curalpha * chosenval1 )
            gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
        end

    end

    ## compute discriminant 
    sqarg = beta^2-4.0*alpha*gamma

    ## To get a non-negative discriminant, need to fulfil:
    ##    (tx-ty)^2 - 2*s^2/curalpha <= 0
    ##    where tx,ty can be
    ##     t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order
    ##     t? = chosenval1  if 1st order 
    ##  
    ## If discriminant is negative (probably because of sharp contrasts in
    ##  velocity) revert to 1st order for both x and y
    if sqarg<0.0

        begin
            alpha = 0.0
            beta  = 0.0
            gamma = - slowcurpt^2 ## !!!!

            ## 2 directions
            for axis=1:2
                
                use1stord = false
                chosenval1 = HUGE
                
                ## two sides for each direction
                for l=1:2

                    ## map the 4 cases to an integer as in linear indexing...
                    lax = l + 2*(axis-1)
                    if lax==1 # axis==1
                        ish = 1
                        jsh = 0
                    elseif lax==2 # axis==1
                        ish = -1
                        jsh = 0
                    elseif lax==3 # axis==2
                        ish = 0
                        jsh = 1
                    elseif lax==4 # axis==2
                        ish = 0
                        jsh = -1
                    end

                    ## check if on boundaries
                    isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                    
                    ## 1st order
                    if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
                        testval1 = fmmvars.ttime[i+ish,j+jsh]
                        ## pick the lowest value of the two
                        if testval1<chosenval1 ## < only
                            chosenval1 = testval1
                            use1stord = true

                            # save derivative choices
                            axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

                        end
                    end
                end # end two sides

                ## spacing
                deltah = Δh[axis]
                
                if use1stord # first order
                    ## curalpha: make sure you multiply only times the
                    ##   current alpha for beta and gamma...
                    curalpha = 1.0/deltah^2 
                    alpha += curalpha
                    beta  += ( -2.0*curalpha * chosenval1 )
                    gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
                end
            end
            
            ## recompute sqarg
            sqarg = beta^2-4.0*alpha*gamma

        end ## begin...

        if sqarg<0.0

            if fmmvars.allowfixsqarg==true
            
                gamma = beta^2/(4.0*alpha)
                sqarg = beta^2-4.0*alpha*gamma
                println("calcttpt_2ndord(): ### Brute force fixing problems with 'sqarg', results may be quite inaccurate. ###")
                
            else
                println("\n To get a non-negative discriminant, need to fulfil: ")
                println(" (tx-ty)^2 - 2*s^2/curalpha <= 0")
                println(" where tx,ty can be")
                println(" t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order")
                println(" t? = chosenval1  if 1st order ")
                
                error("calcttpt_2ndord(): sqarg<0.0, negative discriminant (at i=$i, j=$j)")
            end
        end
    end ## if sqarg<0.0

    ### roots of the quadratic equation
    tmpsq = sqrt(sqarg)
    soughtt1 =  (-beta + tmpsq)/(2.0*alpha)
    soughtt2 =  (-beta - tmpsq)/(2.0*alpha)

    ## choose the largest solution
    soughtt = max(soughtt1,soughtt2)

    return soughtt
end

################################################################################

"""
$(TYPEDSIGNATURES)

  Refinement of the grid around the source. FMM calculated inside a finer grid 
    and then passed on to coarser grid
"""
function ttaroundsrc!(fmmcoarse::FMMvars2D,vel::Array{Float64,2},src::AbstractVector{Float64},
                      grdcoarse::GridEik2D, inittt::Float64 ;                      
                      dodiscradj::Bool=false,idxconv::Union{MapOrderGridFMM2D,Nothing}=nothing,
                      fmmord::Union{VarsFMMOrder2D,Nothing}=nothing )
    
    ##
    if typeof(grdcoarse)==Grid2D
        simtype=:cartesian
    elseif typeof(grdcoarse)==Grid2DSphere
        simtype=:spherical
    end

    ##
    ## 2x10 nodes -> 2x50 nodes
    ##
    downscalefactor::Int = 0
    noderadius::Int = 0
    if dodiscradj
        downscalefactor = 5
        ## 2 instead of 5 for adjoint to avoid messing up...
        noderadius = 2 
    else
        downscalefactor = 5
        noderadius = 5
    end        

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    if simtype==:cartesian
        n1_coarse,n2_coarse = grdcoarse.nx,grdcoarse.ny
        ixsrcglob,iysrcglob = findclosestnode(src[1],src[2],grdcoarse.xinit,grdcoarse.yinit,grdcoarse.hgrid) 
    elseif simtype==:spherical
        n1_coarse,n2_coarse = grdcoarse.nr,grdcoarse.nθ
        ixsrcglob,iysrcglob = findclosestnode_sph(src[1],src[2],grdcoarse.rinit,grdcoarse.θinit,grdcoarse.Δr,grdcoarse.Δθ) 
    end
  
    ##
    ## Define chunck of coarse grid
    ##
    i1coarsevirtual = ixsrcglob - noderadius
    i2coarsevirtual = ixsrcglob + noderadius
    j1coarsevirtual = iysrcglob - noderadius
    j2coarsevirtual = iysrcglob + noderadius
    # if hitting borders
    outxmin = i1coarsevirtual<1
    outxmax = i2coarsevirtual>n1_coarse
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>n2_coarse
    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=n1_coarse : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=n2_coarse : j2coarse=j2coarsevirtual
    
    ##
    ## Refined grid parameters
    ##
    # fine grid size
    n1 = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    n2 = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)    

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,n1,n2)
    for j=1:n2
        for i=1:n1
            di=div(i-1,downscalefactor)
            ri=i-di*downscalefactor
            ii = ri>=downscalefactor/2+1 ? di+2 : di+1

            dj=div(j-1,downscalefactor)
            rj=j-dj*downscalefactor
            jj = rj>=downscalefactor/2+1 ? dj+2 : dj+1

            velfinegrd[i,j] = velcoarsegrd[ii,jj]
        end
    end

    ##
    ## Init arrays
    ##
    fmmfine = FMMvars2D(n1,n2,refinearoundsrc=false,allowfixsqarg=false)

    ## 
    ## Time array
    ##
    inittt = 1e30
    #ttime = Array{Float64}(undef,n1,n2)
    fmmfine.ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    #status = Array{Int64}(undef,n1,n2)
    fmmfine.status[:,:] .= 0   ## set all to far
    # derivative codes
    idD = MVector(0,0)
    ##
    ptij = MVector(0,0)
    ##
    ijsrc_coarse = Matrix{Int64}(undef,2,n1*n2)
    counter_ijsrccoarse::Int64 = 1
    
    ##
    ## Reset coodinates to match the fine grid
    ##
    # xorig = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    # yorig = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    xsrc = src[1] #- xorig #- grdcoarse.xinit
    ysrc = src[2] #- yorig #- grdcoarse.yinit
    srcfine = MVector(xsrc,ysrc)

    
    if simtype==:cartesian
        # set origin of the fine grid
        xinit = grdcoarse.x[i1coarse]
        yinit = grdcoarse.y[j1coarse]
        dh = grdcoarse.hgrid/downscalefactor
        # fine grid
        grdfine = Grid2D(hgrid=dh,xinit=xinit,yinit=yinit,nx=n1,ny=n2)
        ##
        ## Source location, etc. within fine grid
        ##  
        ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
        ijsrc_fine = sourceboxloctt!(fmmfine,velfinegrd,srcfine,grdfine )
        
    elseif simtype==:spherical
        # set origin of the fine grid
        rinit = grdcoarse.r[i1coarse]
        θinit = grdcoarse.θ[j1coarse]
        dr = grdcoarse.Δr/downscalefactor
        dθ = grdcoarse.Δθ/downscalefactor
        # fine grid
        grdfine = Grid2DSphere(Δr=dr,Δθ=dθ,nr=n1,nθ=n2,rinit=rinit,θinit=θinit)
        ##
        ## Source location, etc. within fine grid
        ##  
        ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
        ijsrc_fine = sourceboxloctt_sph!(fmmfine,velfinegrd,srcfine,grdfine)

    end

    ######################################################
    neigh = SA[1  0;
               0  1;
              -1  0;
               0 -1]

    #-------------------------------
    ## init FMM 
    
    ## number of accepted points       
    naccinit = size(ijsrc_fine,2)

    ##===================================
    for l=1:naccinit
        ia = ijsrc_fine[1,l]
        ja = ijsrc_fine[2,l]
        
        # if the node coincides with the coarse grid, store values
        oncoa,ia_coarse,ja_coarse = isacoarsegridnode(ia,ja,downscalefactor,i1coarse,j1coarse)
        if oncoa
            ##===================================
            ## UPDATE the COARSE GRID
            fmmcoarse.ttime[ia_coarse,ja_coarse]  = fmmfine.ttime[ia,ja]
            fmmcoarse.status[ia_coarse,ja_coarse] = fmmfine.status[ia,ja]
            ijsrc_coarse[:,counter_ijsrccoarse] .= (ia_coarse,ja_coarse)
            counter_ijsrccoarse +=1
            ##==================================
        end
    end
    
 
    ##==========================================================##
    # discrete adjoint
    if dodiscradj       

        # discrete adjoint: first visited points in FMM order
        ttfirstpts = Vector{Float64}(undef,naccinit)
        for l=1:naccinit
            i,j = ijsrc_fine[1,l],ijsrc_fine[2,l]
            ttij = fmmfine.ttime[i,j]
            # store initial points' FMM order
            ttfirstpts[l] = ttij
        end
        # sort according to arrival time
        spidx = sortperm(ttfirstpts)
        
        # store the first accepted points' order
        node_coarse = 1
        for l=1:naccinit
            # ordered index
            p = spidx[l]

            ## if the point belongs to the coarse grid, add it
            i,j = ijsrc_fine[1,l],ijsrc_fine[2,l]
            oncoa,ia_coarse,ja_coarse = isacoarsegridnode(i,j,downscalefactor,i1coarse,j1coarse)
            if oncoa
                # go from cartesian (i,j) to linear
                idxconv.lfmm2grid[node_coarse] = cart2lin2D(ia_coarse,ja_coarse,idxconv.nx)

                # store arrival time for first points in FMM order
                fmmord.ttime[node_coarse] = fmmfine.ttime[i,j]
                ## update counter!
                node_coarse +=1
            end
            #@show is[p],js[p],ttime[is[p],js[p]],idxconv.lfmm2grid[l]
        end
    end # if dodiscradj
    ##==========================================================##

    # ## Init the min binary heap with void arrays but max size
    # Nmax=n1*n2
    # bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## construct initial narrow band
    for l=1:naccinit ##
        
         for ne=1:4 ## four potential neighbors

            i = ijsrc_fine[1,l] + neigh[ne,1]
            j = ijsrc_fine[2,l] + neigh[ne,2]

             ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end


            if fmmfine.status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(fmmfine,velfinegrd,grdfine,i,j,idD)
                # get handle
                han = cart2lin2D(i,j,n1)
                # insert into heap
                insert_minheap!(fmmfine.bheap,tmptt,han)
                # change status, add to narrow band
                fmmfine.status[i,j]=1

            end            
        end
    end

    #-------------------------------
    ## main FMM loop
    firstwarning=true
    totnpts = n1*n2
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if fmmfine.bheap.Nh[]<1
            break
        end

        han,tmptt = pop_minheap!(fmmfine.bheap)
        #ia,ja = ind2sub((n1,n2),han)
        cija = lin2cart2D!(han,n1,ptij)
        ia,ja = ptij[1],ptij[2]
        # set status to accepted
        fmmfine.status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        fmmfine.ttime[ia,ja] = tmptt

        ##===================================
        oncoa,ia_coarse,ja_coarse = isacoarsegridnode(ia,ja,downscalefactor,i1coarse,j1coarse)
        if oncoa

            ##===================================
            ## UPDATE the COARSE GRID
            fmmcoarse.ttime[ia_coarse,ja_coarse] = tmptt
            fmmcoarse.status[ia_coarse,ja_coarse] = 2
            ijsrc_coarse[:,counter_ijsrccoarse] .= (ia_coarse,ja_coarse)
            counter_ijsrccoarse +=1
            ##===================================

            ##===================================
            # Discrete adjoint
            if dodiscradj
                # store the linear index of FMM order
                idxconv.lfmm2grid[node_coarse] = cart2lin2D(ia_coarse,ja_coarse,idxconv.nx)
                # store arrival time for first points in FMM order
                fmmord.ttime[node_coarse] = tmptt
                # increase the counter
                node_coarse += 1
            end
        end
        ##===================================

        ##########################################################
        ##
        ## If the the accepted point is on the edge of the
        ##  fine grid, stop computing and jump to coarse grid
        ##
        ##########################################################
        #  if (ia==n1) || (ia==1) || (ja==n2) || (ja==1)
        if (ia==1 && !outxmin) || (ia==n1 && !outxmax) || (ja==1 && !outymin) || (ja==n2 && !outymax)

            ## delete current narrow band to avoid problems when returned to coarse grid
            fmmcoarse.status[fmmcoarse.status.==1] .= 0

            ## Prevent the difficult case of traveltime hitting the borders but
            ##   not the coarse grid, which would produce an empty "statuscoarse" and an empty "ttimecoarse".
            ## Probably needs a better fix..."
            if count(fmmcoarse.status.>0)<1
                if firstwarning
                    @warn("Traveltime hitting the borders but not the coarse grid, continuing.")
                    firstwarning=false
                end
                continue
            end
            
            return ijsrc_coarse[:,1:counter_ijsrccoarse-1]
        end
        ##########################################################

        ## try all neighbors of newly accepted point
        for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if fmmfine.status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(fmmfine,velfinegrd,grdfine,i,j,idD)
                han = cart2lin2D(i,j,n1)
                insert_minheap!(fmmfine.bheap,tmptt,han)
                # change status, add to narrow band
                fmmfine.status[i,j]=1                

            elseif fmmfine.status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord!(fmmfine,velfinegrd,grdfine,i,j,idD)
                # get handle
                han = cart2lin2D(i,j,n1)
                # update the traveltime for this point in the heap
                update_node_minheap!(fmmfine.bheap,tmptt,han)

            end
        end
        ##-------------------------------
    end
    error("ttaroundsrc!(): Ouch...")
    return
end

#################################################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(fmmvars::FMMvars2D,vel::Array{Float64,2},srcpos::AbstractVector,
                         grd::Grid2D )

    ## source location, etc.      
    mindistsrc = 1e-5   
    xsrc,ysrc=srcpos[1],srcpos[2]

    ## regular grid
    ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
    rx = xsrc-grd.x[ix]
    ry = ysrc-grd.y[iy]
 
    halfg = 0.0 #hgrid/2.0
    # Euclidean distance
    dist = sqrt(rx^2+ry^2)
    #@show dist,src,rx,ry
    if dist<=mindistsrc
        ## single point
        ijsrc = @MMatrix [ix; iy]
        ## set status = accepted == 2
        fmmvars.status[ix,iy] = 2
        ## set traveltime for the source
        fmmvars.ttime[ix,iy] = 0.0

    else
        # pre-allocate array of indices for source
        ijsrc = @MMatrix zeros(Int64,2,4)

        ## eight points
        if rx>=halfg
            srci = (ix,ix+1)
        else
            srci = (ix-1,ix)
        end
        if ry>=halfg
            srcj = (iy,iy+1)
        else
            srcj = (iy-1,iy)
        end
       
        l=1
        for j=1:2, i=1:2
            ijsrc[:,l] .= (srci[i],srcj[j])
            l+=1
        end

        ## set ttime around source ONLY FOUR points!!!
        for l=1:size(ijsrc,2)
            i = ijsrc[1,l]
            j = ijsrc[2,l]

            ## set status = accepted == 2
            fmmvars.status[i,j] = 2

            ## regular grid
            xp = grd.x[i] 
            yp = grd.y[j]
            ii = Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
            jj = Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1             
            ## set traveltime for the source
            fmmvars.ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]
        end
    end
    
    return ijsrc
end 

#################################################################################


"""
$(TYPEDSIGNATURES)
"""
function sourceboxloctt_sph!(fmmvars::FMMvars2D,vel::Array{Float64,2},srcpos::AbstractVector,
                             grd::Grid2DSphere )

    # minimum distance between node and source position
    mindistsrc = 10.0*eps()
  
    rsrc,θsrc=srcpos[1],srcpos[2]

    ## regular grid
    ir,iθ = findclosestnode_sph(rsrc,θsrc,grd.rinit,grd.θinit,grd.Δr,grd.Δθ)
    rr = rsrc-grd.r[ir]
    rθ = θsrc-grd.θ[iθ]

    halfg = 0.0 #hgrid/2.0
    ## distance in POLAR COORDINATES
    ## sqrt(r1^+r2^2 - 2*r1*r2*cos(θ1-θ2))
    r1=rsrc
    r2=grd.r[ir]
    dist = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[iθ]) )  #sqrt(rr^2+grd.r[ir]^2*rθ^2)
    #@show dist,src,rr,rθ
    if dist<=mindistsrc
        ## single point
        ijsrc = @MMatrix [ir; iθ]
        ## set status = accepted == 2
        fmmvars.status[ir,iθ] = 2
        ## set traveltime for the source
        fmmvars.ttime[ir,iθ] =0.0
        
    else
        # pre-allocate array of indices for source
        ijsrc = @MMatrix zeros(Int64,2,4)

        ## eight points
        if rr>=halfg
            srci = (ir,ir+1)
        else
            srci = (ir-1,ir)
        end
        if rθ>=halfg
            srcj = (iθ,iθ+1)
        else
            srcj = (iθ-1,iθ)
        end

        l=1
        for j=1:2, i=1:2
            ijsrc[:,l] .= (srci[i],srcj[j])
            l+=1
        end

        ## set ttime around source ONLY FOUR points!!!
        for l=1:size(ijsrc,2)
            i = ijsrc[1,l]
            j = ijsrc[2,l]

            ## set status = accepted == 2
            fmmvars.status[i,j] = 2
            
            ## regular grid
            # xp = (i-1)*grd.Δr+grd.rinit
            # yp = (j-1)*grd.Δθ+grd.θinit
            ii = Int(floor((rsrc-grd.rinit)/grd.Δr)) +1
            jj = Int(floor((θsrc-grd.θinit)/grd.Δθ)) +1
            ##ttime[i,j] = sqrt((rsrc-xp)^2+(θsrc-yp)^2) / vel[ii,jj]
            ## sqrt(r1^+r2^2 -2*r1*r2*cos(θ1-θ2))
            r1=rsrc
            r2=grd.r[ii]
            # distance
            distp = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[iθ]))
            ## set traveltime for the source
            fmmvars.ttime[i,j] = distp / vel[ii,jj]
        end
    end

    return ijsrc
end

#########################################################

