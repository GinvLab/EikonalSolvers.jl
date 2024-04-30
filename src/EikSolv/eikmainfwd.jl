

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################

##########################################################################

"""
$(TYPEDSIGNATURES)

Calculate the traveltime for 2D or 3D velocity models at requested receivers locations.
Optionally return the array(s) of traveltime on the entire gridded model.

# Arguments
- `vel`: the 2D or 3D velocity model 
- `grd`: a struct specifying the geometry and size of the model (e.g., Grid3DCart)
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
- `returntt`: a boolean (default false) specifying whether to retur the array(s) of traveltime on the entire gridded model or not
- `extraparams` (optional): a struct containing some "extra" parameters, namely
    * `parallelkind`: serial, Threads or Distributed run? (:serial, :sharedmem, :distribmem)
    * `refinearoundsrc`: whether to perform a refinement of the grid around the source location
    * `grdrefpars`: refined grid around the source parameters (`downscalefactor` and `noderadius`)
    * `radiussmoothgradsrc`: radius for smoothing each individual gradient *only* around the source. Zero means no smoothing.
    * `smoothgradkern`: smooth the final gradient with a kernel of size `smoothgradkern` (in grid nodes). Zero means no smoothing.
    * `manualGCtrigger`: trigger garbage collector (GC) manually at selected points.

# Returns
- A vector of vectors containing traveltimes at the receivers for each source 
- If `returntt==true`, additionally returns the array(s) of traveltime on the entire gridded model
"""
function eiktraveltime(vel::Array{Float64,N},
                       grd::AbstractGridEik,
                       coordsrc::Array{Float64,2},
                       coordrec::Vector{Array{Float64,2}} ;
                       returntt::Bool=false,
                       extraparams::Union{ExtraParams,Nothing}=nothing ) where N
        
    if extraparams==nothing
        extraparams =  ExtraParams()
    end

    @assert all(vel.>0.0)
    checksrcrecposition(grd,coordsrc,coordrec)

    # init some arrays
    nsrc = size(coordsrc,1)
    ttpicks = Vector{Vector{Float64}}(undef,nsrc)
    # for i=1:nsrc
    #     curnrec = size(coordrec[i],1) 
    #     ttpicks[i] = zeros(curnrec)
    # end

   
    if returntt
        # return traveltime array and picks at receivers
        #ttime = [zeros(grddims...) for i=1:nsrc]
        ttime = Vector{Array{Float64}}(undef,nsrc) 
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
                @async ttime[igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc,wks[s],
                                                                    vel,coordsrc[igrs,:],
                                                                    coordrec[igrs],grd,extraparams,
                                                                    returntt=returntt)
            end
        else
            # return ONLY traveltime picks at receivers 
            @sync for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc,wks[s],
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
                ttime[igrs],ttpicks[igrs] = ttforwsomesrc(vel,coordsrc[igrs,:],
                                                          coordrec[igrs],grd,extraparams,
                                                          returntt=returntt )
            end

        else
            # return ONLY traveltime picks at receivers
            Threads.@threads for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                ttpicks[igrs] = ttforwsomesrc(vel,coordsrc[igrs,:],
                                              coordrec[igrs],grd,extraparams,
                                              returntt=returntt )
            end
        end


    elseif extraparams.parallelkind==:serial
        ##====================================
        ## Serial run
        ##====================
        if returntt            
            # return both traveltime picks at receivers and at all grid points
            ttime,ttpicks = ttforwsomesrc(vel,coordsrc,coordrec,grd,extraparams,
                                          returntt=returntt )
        else
            # return ONLY traveltime picks at receivers
            ttpicks = ttforwsomesrc(vel,coordsrc,coordrec,grd,extraparams,
                                    returntt=returntt )
        end

    end
    
    if returntt 
        return ttpicks,ttime
    end
    return ttpicks
end

#######################################################

"""
$(TYPEDSIGNATURES)

  Compute the forward problem for a group of sources.
"""
function ttforwsomesrc(vel::Array{Float64,N},coordsrc::Array{Float64,2},
                       coordrec::Vector{Array{Float64,2}},grd::AbstractGridEik,
                       extrapars::ExtraParams ; returntt::Bool=false ) where N
    
    nsrc = size(coordsrc,1)
    # init some arrays
    ttpicksGRPSRC = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicksGRPSRC[i] = zeros(curnrec)
    end

    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars = createFMMvars(grd,amIcoarsegrid=true,
                            refinearoundsrc=extrapars.refinearoundsrc)
                            

    ## pre-allocate discrete adjoint variables
    ##  No adjoint calculations
    ## adjvars = nothing

    if returntt
        ttGRPSRC = Vector{Array{Float64}}(undef,nsrc) #zeros(n1,n2,nsrc)
    end

    ## group of pre-selected sources
    for s=1:nsrc    
        ## run the FMM forward
        ttFMM_hiord!(fmmvars,vel,coordsrc[s,:],grd,extrapars)

        ## interpolate at receivers positions
        for i=1:size(coordrec[s],1)
            #  view !!
            #ttpicksGRPSRC[s][i] = bilinear_interp( fmmvars.ttime,grd,view(coordrec[s],i,:) )
            ttpicksGRPSRC[s][i] = interpolate_receiver( fmmvars.ttime,grd,view(coordrec[s],i,:) )
        end

        if returntt
            ## Compute traveltime and interpolation at receivers in one go for parallelization
            ttGRPSRC[s] = copy(fmmvars.ttime) ## copy!!!
        end
    end

    if returntt 
        return ttGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC 
end

######################################################

function createFMMvars(grd::AbstractGridEik2D;
                       amIcoarsegrid::Bool,
                       refinearoundsrc::Bool)
    n1,n2 = grd.grsize
    fmmvars = FMMVars2D(n1,n2,amIcoarsegrid=true,
                        refinearoundsrc=refinearoundsrc)
    return fmmvars
end

function createFMMvars(grd::AbstractGridEik3D;
                       amIcoarsegrid::Bool,
                       refinearoundsrc::Bool)
    n1,n2,n3 = grd.grsize
    fmmvars = FMMVars3D(n1,n2,n3,amIcoarsegrid=true,
                        refinearoundsrc=refinearoundsrc)
    return fmmvars
end

#######################################################

# Default constructor for forward calculations (adjvars=nothing)
ttFMM_hiord!(fmmvars::AbstractFMMVars,vel::Array{Float64,N},src::AbstractVector{Float64},grd::AbstractGridEik,
             extrapars::ExtraParams ) where N = ttFMM_hiord!(fmmvars,vel,src,grd,nothing,extrapars)

##----------------------------------

# Fast marching computations
function ttFMM_hiord!(fmmvars::AbstractFMMVars,vel::Array{Float64,N},src::AbstractVector{Float64},grd::AbstractGridEik,
                      adjvars::Union{AbstractAdjointVars,Nothing},extrapars::ExtraParams ) where N
    
    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end

    Ndim = ndims(vel)
    @assert Ndim==length(grd.grsize)

    ##==================================================##
    ###
    ###  Init FMM arrays
    ###  
    ## Time array
    ##
    # max possible val for type of ttime
    fmmvars.ttime .= typemax(eltype(fmmvars.ttime)) 
    ##
    ## Status of nodes
    ##
    fmmvars.status .= 0   ## set all to far
    ##==================================================

    ##==================================================
    if dodiscradj
        ##
        ##  Init discrete adjoint stuff
        ##
        nxyz = prod(grd.grsize)
        ## Initialize size of derivative matrices as full nxyz*nxyz,
        ##   then rows corresponding to "source" points will be removed
        ##   later on
        ##
        for i=1:Ndim
            adjvars.fmmord.Deriv[i].Nsize .= MVector(nxyz,nxyz)
            adjvars.fmmord.Deriv[i].lastrowupdated[] = 0
            adjvars.fmmord.Deriv[i].Nnnz[] = 0
        end
        adjvars.fmmord.onsrccols .= false
        adjvars.fmmord.onhpoints .= false
        # for safety, zeroes the traveltime
        adjvars.fmmord.ttime .= -1.0 #0.0
        # init the valued of last ttime computed
        adjvars.fmmord.lastcomputedtt[] = 0
        # for safety, zeroes the codes for derivatives
        adjvars.codeDeriv .= 0
    end
    ##==================================================
    
    if fmmvars.refinearoundsrc && dodiscradj==false
        ##===================================================
        ## 
        ## DO refinement around the source, NO adjoint run  
        ##
        ##===================================================
        # fmmvars_fine,adjvars_fine need to be *re-allocated* for each source
        #  because the size of the grid may change when hitting borders, etc.
        runrefinementaroundsrc!(fmmvars,vel,src,grd,adjvars,extrapars)


    elseif fmmvars.refinearoundsrc && dodiscradj
        ##===================================================
        ## 
        ## DO refinement around the source and adjoint run    
        ##
        ##===================================================
        # fmmvars_fine,adjvars_fine need to be *re-allocated* for each source
        #  because the SIZE of the grid may CHANGE when hitting borders, etc.
        fmmvars_fine,adjvars_fine,grd_fine,srcrefvars = runrefinementaroundsrc!(fmmvars,vel,src,grd,
                                                                                adjvars,extrapars)
    
    else        
        ##================================================
        ## 
        ## NO refinement around the source, init stuff      
        ##
        ##================================================
        initnorefsrc(fmmvars,vel,src,grd,adjvars)

    end

    ##=========================================
    ## 
    ## Run main simulation (coarse grid)
    ##
    ##=========================================
    ttFMM_core!( fmmvars,vel,grd,adjvars )

    if fmmvars.refinearoundsrc && dodiscradj 
        return fmmvars_fine,adjvars_fine,grd_fine,srcrefvars
    end
    
    return
end

#######################################################

function runrefinementaroundsrc!(fmmvars::AbstractFMMVars,vel::Array{Float64,N},
                                 xyzsrc::AbstractVector{Float64},
                                 grd::AbstractGridEik,
                                 adjvars::Union{AbstractAdjointVars,Nothing},
                                 extrapars::ExtraParams) where N
                              
    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end

    if typeof(grd)==Grid2DCart || typeof(grd)==Grid2DSphere 
        Ndim = 2
    elseif typeof(grd)==Grid3DCart || typeof(grd)==Grid3DSphere
        Ndim = 3        
    end
    
    ##==================================================
    ## create the refined grid
    grd_fine,srcrefvars = createfinegrid(grd,xyzsrc,vel,
                                         extrapars.grdrefpars)

    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars_fine = createFMMvars(grd_fine,amIcoarsegrid=false,
                                 refinearoundsrc=false)                           
    
    ##==================================================
    # init ttime 
    inittt = typemax(eltype(fmmvars_fine.ttime))
    fmmvars_fine.ttime .= inittt
    # Status of nodes
    fmmvars_fine.status .= 0   ## set all to far

    ##==================================================
    if dodiscradj
        ## pre-allocate adjoint variables for the fine grid
        adjvars_fine = createAdjointVars(size(srcrefvars.velcart_fine))
        ##
        ##  discrete adjoint: init stuff
        ## 
        for i=1:Ndim
            adjvars_fine.fmmord.Deriv[i].lastrowupdated[] = 0
            adjvars_fine.fmmord.Deriv[i].Nnnz[] = 0
        end
        adjvars_fine.fmmord.onsrccols .= false
        adjvars_fine.fmmord.onhpoints .= false
        # for safety, zeroes the traveltime
        adjvars_fine.fmmord.ttime .= -1.0
        # init the valued of last ttime computed
        adjvars_fine.fmmord.lastcomputedtt[] = 0
        # for safety, zeroes the codes for derivatives
        adjvars_fine.codeDeriv .= 0

    else
        adjvars_fine = nothing

    end
    ##==================================================
    ## source location, etc.
    sourceboxloctt!(fmmvars_fine,srcrefvars.velcart_fine,xyzsrc,grd_fine)

    ##==================================================================
    ## Run forward simulation (FMM) within the FINE grid
    #ijsrc_coarse = ttFMM_core!(fmmvars_fine,vel_fine,grd_fine,ijsrc_fine,adjvars_fine,
    ttFMM_core!(fmmvars_fine,srcrefvars.velcart_fine,grd_fine,adjvars_fine,
                fmmvars_coarse=fmmvars, adjvars_coarse=adjvars,
                srcrefvars=srcrefvars)
         
    ##==================================================================
    if dodiscradj
        return fmmvars_fine,adjvars_fine,grd_fine,srcrefvars
    else
        return 
    end
end

###########################################################################

function initnorefsrc(fmmvars::AbstractFMMVars,vel::Array{Float64,N},
                      src::AbstractVector{Float64},
                      grd::AbstractGridEik,
                      adjvars::Union{AbstractAdjointVars,Nothing}) where N

    ## source location, etc.      
    sourceboxloctt!(fmmvars,vel,src,grd)

    return 
end

##################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 2D using traditional stencils on regular grid. 
"""
function ttFMM_core!(fmmvars::AbstractFMMVars,vel::Array{Float64,N},grd::AbstractGridEik,
                     adjvars::Union{AbstractAdjointVars,Nothing} ;
                     fmmvars_coarse::Union{AbstractFMMVars,Nothing}=nothing,
                     adjvars_coarse::Union{AbstractAdjointVars,Nothing}=nothing,
                     srcrefvars::Union{AbstractSrcRefinVars,Nothing}=nothing) where N
                   

    if srcrefvars==nothing
        isthisrefinementsrc = false
    else
        isthisrefinementsrc = true
    end
                     
    if isthisrefinementsrc
        @assert fmmvars.refinearoundsrc==false "ttFMM_core!(): Inconsistency: isthisrefinementsrc and fmmvars.refinearoundsrc are both true."
    end
    if isthisrefinementsrc
        downscalefactor = srcrefvars.downscalefactor
        ijkorigincoarse = srcrefvars.ijkorigincoarse
        outijk_min = srcrefvars.outijk_min
        outijk_max = srcrefvars.outijk_max
    end

    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end    

    ##----------------------------------------------------------------------
    ## fmmvars.srcboxpar.ijksrc can be a field of either
    ##  - a SrcRefinVars if we are in the coarse grid with no refinement,
    ##     or in the fine grid
    ##  - a SourcePtsFromFineGrid if we are in the coarse grid after the
    ##     FMM has been already run in the fine grid (multiple source points)
    ##
    ijksrc = fmmvars.srcboxpar.ijksrc

    ## number of accepted points       
    naccinit = size(ijksrc,1)  

    ## Establish a mapping between linear and Cartesian indexing 
    grsize = grd.grsize
    totnpts = prod(grsize)
    ND = ndims(vel)
    @assert ND==N
    
    ##======================================================
    # MVector size needs to be know at compile time, so...
    # a 2D point
    curptijk = MVector{ND,Int64}(undef)
    accptijk = MVector{ND,Int64}(undef)
    # derivative codes
    idD = MVector{ND,Int64}(undef)

    if ND==2
        # neighboring points
        neigh = SA[ 1  0;
                    0  1;
                   -1  0;
                    0 -1]
 
    elseif ND==3
        # neighboring points
        neigh = SA[ 1  0  0;
                    0  1  0;
                    0  0  1;
                   -1  0  0;
                    0 -1  0;
                    0  0 -1]
    end   

    ##=========================================
    if dodiscradj
        ## number of accepted points       
        naccinit = size(ijksrc,1)
        ## total number of traveltimes computed so far
        adjvars.fmmord.lastcomputedtt[] = naccinit

        # discrete adjoint: first visited points in FMM order
        ttfirstpts = Vector{Float64}(undef,naccinit)
        for l=1:naccinit
            #i,j = 
            curptijk .= ijksrc[l,:]
            # get the linear index/handle 
            curpthan = cart2lin(curptijk,grsize)
            # ttime 
            ttij = fmmvars.ttime[curpthan]
            # store initial points' FMM order
            ttfirstpts[l] = ttij
        end
        # sort according to arrival time
        spidx = sortperm(ttfirstpts)

        # store the first accepted points' order
        for p=1:naccinit
            # ordered index
            l = spidx[p]
            # go from cartesian (i,j) to linear
            curptijk .= ijksrc[l,:]
            # get the linear index/handle 
            curpthan = cart2lin(curptijk,grsize)
            # adj stuff
            adjvars.idxconv.lfmm2grid[l] = curpthan
            
            ######################################
            # store arrival time for first points in FMM order
            ttgrid = fmmvars.ttime[curpthan]
            adjvars.fmmord.ttime[l] = ttgrid

            ##======================================================
            ##  We are on a source node
            ##================================
            # add true for this point being on src
            adjvars.fmmord.onsrccols[l] = true
            # remove one row because of point being on source!
            for d=1:ND
                adjvars.fmmord.Deriv[d].Nsize[1] -= 1
            end
        end
    end # if dodiscradj
    ##======================================================

     
    ##======================================================
    ## Only while running in the fine grid
    ##======================================================
    if isthisrefinementsrc
        # init source indices array and some counters
        ijksrc_coarse = Matrix{Int64}(undef,totnpts,ND)
        counter_ijksrccoarse::Int64 = 1
        counter_adjcoarse::Int64 = 1

        for l=1:naccinit
            curptijk .= ijksrc[l,:]
            oncoa,ijk_coarse = isacoarsegridnode(curptijk,downscalefactor,ijkorigincoarse)

            if oncoa
                # get the linear index/handle 
                curpthan = cart2lin(curptijk,grsize)
                pthan_coarse = cart2lin(ijk_coarse,size(fmmvars_coarse.ttime))
                # update stuff
                fmmvars_coarse.ttime[pthan_coarse]  = fmmvars.ttime[curpthan]
                fmmvars_coarse.status[pthan_coarse] = fmmvars.status[curpthan]
                ijksrc_coarse[counter_ijksrccoarse,:] .= ijk_coarse
                ## update counter
                counter_ijksrccoarse += 1

                if dodiscradj
                    ## next line on FINE grid:
                    adjvars.fmmord.onhpoints[l] = true
                end
            end
        end
    end
    ##======================================================

    ######################################################
    ## init FMM 
    ######################################################
                
    ##-----------------------------------------
    ## Construct INITIAL narrow band
    for l=1:naccinit ##

        for ne=1:size(neigh,1) ## four potential neighbors
            #
            curptijk .= ijksrc[l,:] .+ neigh[ne,:]
            # get the linear index/handle 
            curpthan = cart2lin(curptijk,grsize)

            ## if the point is out of bounds skip this iteration
            #if (i>n1) || (i<1) || (j>n2) || (j<1)
            if any(curptijk.<1) || any(curptijk.>grsize)
                continue
            end
            
                
            if fmmvars.status[curpthan]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(fmmvars,vel,grd,curptijk,idD)                              
                # insert into heap
                insert_minheap!(fmmvars.bheap,tmptt,curpthan)
                # change status, add to narrow band
                fmmvars.status[curpthan]=1

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    adjvars.codeDeriv[curpthan,:] .= idD
                end
                              
            end ## if fmmvars.status[i,j]==0 ## far
        end ## for ne=1:size(neigh,1)
    end
        

    ######################################################
    ## main FMM loop
    ######################################################
    firstwarning=true
    node = naccinit

    for iter=1:totnpts-naccinit  #naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if fmmvars.bheap.Nh[]<1
            break
        end

        ##===================================
        ##
        ##  Loop over same min values
        ##    (pop all at the same time...)
        ##
        maxnpop = 128 ## Arbitrary number...???
        accptijk_ls = MMatrix{maxnpop,ND,Int64}(undef)
        sameminval = true
        npopped::Int64 = 0

        while sameminval

            ##
            ## Update node index/counter!
            ##  ( starts from naccinit+1 )
            node += 1
            
            ## if no top left exit the game...
            if fmmvars.bheap.Nh[]<1
                break
            end
            
            # pop value from min-heap
            acchan,tmptt = pop_minheap!(fmmvars.bheap)
            # check if there are other values with same tt
            if topval_heap(fmmvars.bheap)[1] == tmptt
                sameminval = true
            else
                sameminval = false
            end
            # get the indices
            lin2cart!(acchan,grsize,accptijk)
            # set status to accepted
            fmmvars.status[acchan] = 2 # 2=accepted
            # set traveltime of the new accepted point
            fmmvars.ttime[acchan] = tmptt

            ##===================================
            # Discrete adjoint stuff
            if dodiscradj
                # store the linear index of FMM order
                adjvars.idxconv.lfmm2grid[node] = acchan #cart2lin(accptijk,grsize)
                # store arrival time for first points in FMM order
                adjvars.fmmord.ttime[node] = tmptt
                ## total number of traveltimes computed so far
                adjvars.fmmord.lastcomputedtt[] = node 
            end
           
            ##======================================================
            ## Are we running a refinement around the source? If so, 
            ##    UPDATE the COARSE GRID
            ##======================================================
            if isthisrefinementsrc
                oncoa,ijk_coarse = isacoarsegridnode(accptijk,downscalefactor,ijkorigincoarse)

                if oncoa
                    # get the linear index/handle 
                    pthan_coarse = cart2lin(ijk_coarse,size(fmmvars_coarse.status))
                    # update stuff
                    fmmvars_coarse.ttime[pthan_coarse]  = fmmvars.ttime[acchan]
                    fmmvars_coarse.status[pthan_coarse] = fmmvars.status[acchan]
                    ijksrc_coarse[counter_ijksrccoarse,:] .= ijk_coarse
                    
                    ## update counter
                    counter_ijksrccoarse += 1
                    
                    if dodiscradj
                        ## next line on FINE grid
                        adjvars.fmmord.onhpoints[node] = true ## adjvars of FINE grid
                        # Discrete adjoint on COARSE grid
                        # go from cartesian (i,j) to linear
                        adjvars_coarse.idxconv.lfmm2grid[counter_adjcoarse] = pthan_coarse
                        # store arrival time for first points in FMM order
                        adjvars_coarse.fmmord.ttime[counter_adjcoarse] = fmmvars.ttime[acchan]
                        ## update counter
                        counter_adjcoarse += 1
                    end
                end

                ##########################################################
                ##
                ## If the the accepted point is on the edge of the
                ##  fine grid, stop computing and jump to coarse grid
                ##
                ##########################################################
                #if (ia==1 && !outxmin) || (ia==n1 && !outxmax) || (ja==1 && !outymin) || (ja==n2 && !outymax)
                ## outijkmin/max determine if we are touching the edges of the model
                if (any(accptijk.==1 .&& .!outijk_min)) || (any(accptijk.==grsize .&& .!outijk_max))
                    
                    ## Prevent the difficult case of traveltime hitting the borders but
                    ##   not the coarse grid, which would produce an empty "statuscoarse" and an empty "ttimecoarse".
                    ## Probably needs a better fix..."
                    if count(fmmvars_coarse.status.>0)<1
                        if firstwarning
                            @warn("Traveltime hitting the borders but not the coarse grid, continuing.")
                            firstwarning=false
                        end
                        continue
                    end

                    ##==============================================================
                    ##  Create the derivative matrices before quitting (sparse)
                    if dodiscradj
                        createsparsederivativematrices!(grd,adjvars,fmmvars.status)
                    end
                    ##==============================================================

                    ## delete current narrow band to avoid problems when returned to coarse grid
                    fmmvars_coarse.status[fmmvars_coarse.status.==1] .= 0

                    ###############################################
                    ##  Save SOURCE points for the COARSE grid!
                    ##############################################
                    ## re-allocate srcboxpar.ijksrc of type SourcePtsFromFineGrid (mutable struct)
                    fmmvars_coarse.srcboxpar.ijksrc = ijksrc_coarse[1:counter_ijksrccoarse-1,:]

                    return 
                end
                
            end # if isthisrefinementsrc      
            ##===================================
            
            npopped += 1
            accptijk_ls[npopped,:] .= accptijk

        end ## while sameminval
        ##===================================
 

        #############################################
        ## loop over all the popped points
        for ipop=1:npopped

            ##===========================================
            ## try all neighbors of newly accepted point
            for ne=1:size(neigh,1)
                #
                curptijk .= accptijk_ls[ipop,:] .+ neigh[ne,:]
                # get the linear index/handle 
                curpthan = cart2lin(curptijk,grsize)

                ## if the point is out of bounds skip this iteration
                #if (i>n1) || (i<1) || (j>n2) || (j<1)
                if any(curptijk.<1) || any(curptijk.>grsize)
                    continue
                end

    
                if fmmvars.status[curpthan]==0 ## far, active
                    
                    ## add tt of point to binary heap and give handle
                    tmptt = calcttpt_2ndord!(fmmvars,vel,grd,curptijk,idD)
                    # insert new point into min-heap
                    insert_minheap!(fmmvars.bheap,tmptt,curpthan)
                    # change status, add to narrow band
                    fmmvars.status[curpthan]=1

                    if dodiscradj
                        # codes of chosen derivatives for adjoint
                        adjvars.codeDeriv[curpthan,:] .= idD
                    end

                elseif fmmvars.status[curpthan]==1 ## narrow band                

                    # update the traveltime for this point
                    tmptt = calcttpt_2ndord!(fmmvars,vel,grd,curptijk,idD)
                    # update the traveltime for this point in the heap
                    update_node_minheap!(fmmvars.bheap,tmptt,curpthan)

                    if dodiscradj
                        # codes of chosen derivatives for adjoint
                        adjvars.codeDeriv[curpthan,:] .= idD
                    end

                end ## if far or active
            end # for ne=1:size(neigh,1)
       end # for ipop=1:npoppts
    end # for iter=totnpts-naccinit

    ##======================================================
    if dodiscradj
        createsparsederivativematrices!(grd,adjvars,fmmvars.status)
    end # if dodiscradj
    ##======================================================
    
    return 
end

##################################################################

function createsparsederivativematrices!(grd::AbstractGridEik,
                                         adjvars::AbstractAdjointVars,
                                         status::Array{UInt8,N}) where N

    # pre-determine derivative coefficients for positive codes (0,+1,+2)
    if typeof(grd)==Grid2DCart
        simtype = :cartesian
        Ndim = 2
        n1,n2 = grd.grsize
        npts = n1*n2
        hgrid = grd.hgrid
        #allcoeff = [[-1.0/hgrid, 1.0/hgrid], [-3.0/(2.0*hgrid), 4.0/(2.0*hgrid), -1.0/(2.0*hgrid)]]
        allcoeff = Vector{CoeffDerivCartesian}(undef,Ndim)
        allcoeff[1] = CoeffDerivCartesian( MVector(-1.0/hgrid,
                                                   1.0/hgrid), 
                                           MVector(-3.0/(2.0*hgrid),
                                                   4.0/(2.0*hgrid),
                                                   -1.0/(2.0*hgrid)) )
        # coefficients along X are the same than along Y
        allcoeff[2] = allcoeff[1]

    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
        Ndim = 2
        n1,n2 = grd.grsize
        npts = n1*n2
        Δr = grd.Δr
        Δθ = grd.Δθ
        ## DEG to RAD !!!!
        Δarc = [grd.r[i] * deg2rad(Δθ) for i=1:n1]
        # coefficients
        coe_r_1st = MVector(-1.0/Δr,  1.0/Δr)
        coe_r_2nd = MVector(-3.0/(2.0*Δr),  4.0/(2.0*Δr), -1.0/(2.0*Δr) )
        coe_θ_1st = [-1.0./Δarc  1.0./Δarc] 
        coe_θ_2nd = [-3.0./(2.0.*Δarc)  4.0./(2.0.*Δarc)  -1.0./(2.0.*Δarc)] 

        allcoeff = Vector{CoeffDerivSpherical2D}(undef,Ndim)
        # @show typeof(coe_r_1st),typeof(coe_r_2nd)
        # @show typeof(coe_θ_1st),typeof(coe_θ_2nd)
        allcoeff[1] = CoeffDerivSpherical2D( coe_r_1st, coe_r_2nd )
        allcoeff[2] = CoeffDerivSpherical2D( coe_θ_1st, coe_θ_2nd )

    elseif typeof(grd)==Grid3DCart
        simtype = :cartesian
        Ndim = 3
        n1,n2,n3 = grd.grsize
        npts = n1*n2*n3
        hgrid = grd.hgrid
        #allcoeff = [[-1.0/hgrid, 1.0/hgrid], [-3.0/(2.0*hgrid), 4.0/(2.0*hgrid), -1.0/(2.0*hgrid)]]
        allcoeff = Vector{CoeffDerivCartesian}(undef,Ndim)
        allcoeff[1]= CoeffDerivCartesian( MVector(-1.0/hgrid,
                                                 1.0/hgrid), 
                                         MVector(-3.0/(2.0*hgrid),
                                                 4.0/(2.0*hgrid),
                                                 -1.0/(2.0*hgrid)) )
        # coefficients along X are the same than along Y and Z
        allcoeff[2] = allcoeff[1]
        allcoeff[3] = allcoeff[1]
        

    elseif typeof(grd)==Grid3DSphere

        error("createsparsederivativematrices!(): Not yet implemented for Grid3DSphere.")

    end

    ##--------------------------------------------------------
    ## pre-compute the mapping between fmm and original order
    
    nptsonsrc = count(adjvars.fmmord.onsrccols)
    #nptsonh   = count(adjvars.fmmord.onhpoints)

    for i=1:npts
        # ifm = index from fast marching ordering
        ifm = adjvars.idxconv.lfmm2grid[i]
        if ifm==0
            # ifm is zero = end of indices for fmm ordering [lfmm2grid=zeros(Int64,nxyz)]
            Nnnzsteps = i-1
            # remove rows corresponding to points in the source region
            for d=1:Ndim
                adjvars.fmmord.Deriv[d].Nsize[1] = Nnnzsteps - nptsonsrc
                adjvars.fmmord.Deriv[d].Nsize[2] = Nnnzsteps
            end
            break
        end
        adjvars.idxconv.lgrid2fmm[ifm] = i
    end

    ##
    ## Set the derivative operators in FMM order, skipping source points onwards
    ## 
    ##   From adjvars.fmmord.vecDx.lastrowupdated[]+1 onwards because some rows might have already
    ##   been updated above and the CSR format used here requires to add rows in sequence to
    ##   avoid expensive re-allocations
    #startloop = adjvars.fmmord.vecDx.lastrowupdated[]+1
    #startloop = count(adjvars.fmmord.onsrccols)+1
    nptsfixedtt = count(adjvars.fmmord.onsrccols )

    ## init MVectors as 3D even if we are in the 2D case, only
    ##   the first indices will be used...
    ijkpt = MVector{Ndim,Int64}(undef)
    ijkpt .= 0
    colinds = MVector(0,0,0) # 3 elem because max stencil length is 3
    colvals = MVector(0.0,0.0,0.0) # 3 elem because max stencil length is 3
    idxperm = MVector(0,0,0) # 3 elem because max stencil length is 3

    #for irow=startloop:n12
    axis = (:X, :Y, :Z)
    for irow=1:adjvars.fmmord.Deriv[1].Nsize[1]

        for d=1:Ndim
            # compute the coefficients for X,Y,Z derivatives
            setcoeffderiv!(adjvars.fmmord.Deriv[d],status,irow,adjvars.idxconv,
                           adjvars.codeDeriv,allcoeff[d],ijkpt,
                           colinds,colvals,idxperm,nptsfixedtt,
                           axis=axis[d],simtype=simtype)
        end

    end

    return
end

##########################################################################

function createfinegrid(grd::AbstractGridEik,xyzsrc::AbstractVector{Float64},
                        vel::Array{Float64,N},
                        grdrefpars::GridRefinementPars) where N

    if typeof(grd)==Grid2DCart || typeof(grd)==Grid3DCart 
        simtype=:cartesian
    elseif typeof(grd)==Grid2DSphere || typeof(grd)==Grid3DSphere
        simtype=:spherical
    end
    
    downscalefactor = grdrefpars.downscalefactor 
    noderadius = grdrefpars.noderadius 
    Ndim = ndims(vel)

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    if simtype==:cartesian
        grsize_coarse = size(vel)
        ijksrccorn = findenclosingbox(grd,xyzsrc)
            
    elseif simtype==:spherical
        # n1_coarse,n2_coarse,n3_coarse = grd.nr,grd.nθ,grd.φ
        # ixsrcglob,iysrcglob = findclosestnode_sph(xyzsrc[1],xyzsrc[2],xyzsrc[3],
        #                                           grd.rinit,grd.θinit,grd.φinit,
        #                                           grd.Δr,grd.Δθ,grd.Δφ)
        grsize_coarse = size(vel)
        ijksrccorn = findenclosingbox(grd,xyzsrc)
    end
    
    ##
    ## Define chunck of coarse grid
    ##
    ijk1coarsevirtual = MVector{Ndim,Int64}(undef)
    ijk2coarsevirtual = MVector{Ndim,Int64}(undef)
    outijkmin = MVector{Ndim,Bool}(undef)
    outijkmax = MVector{Ndim,Bool}(undef)

    ijk1coarsevirtual = MVector{Ndim,Int64}(undef)
    ijk2coarsevirtual = MVector{Ndim,Int64}(undef)
    for d=1:Ndim
        ijk1coarsevirtual[d] = minimum(ijksrccorn[:,d]) - noderadius
        ijk2coarsevirtual[d] = maximum(ijksrccorn[:,d]) + noderadius
    end

    # if hitting borders
    ijk1coarse = MVector{Ndim,Int64}(undef)
    ijk2coarse = MVector{Ndim,Int64}(undef)
    for d=1:Ndim
        # define outijk...
        outijkmin[d] = ijk1coarsevirtual[d] < 1
        outijkmax[d] = ijk2coarsevirtual[d] > grsize_coarse[d]
        # reset ijk1coarse, etc. following outijkmin/max
        outijkmin[d] ? ijk1coarse[d]=1  : ijk1coarse[d]=ijk1coarsevirtual[d]
        outijkmax[d] ? ijk2coarse[d]=grsize_coarse[d] : ijk2coarse[d]=ijk2coarsevirtual[d]
    end

    ##
    ## Refined grid parameters
    ##
    grsize_fine = Tuple((ijk2coarse.-ijk1coarse).*downscalefactor.+1)
    
    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,[ijk1coarse[d]:ijk2coarse[d] for d=1:Ndim]...)    
    #velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)    

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ##
    grsize_window_coarse = Tuple(ijk2coarse.-ijk1coarse.+1)
    # n1_window_coarse = i2coarse-i1coarse+1
    # n2_window_coarse = j2coarse-j1coarse+1

    #nearneigh_oper = spzeros(n1_fine*n2_fine, n1_window_coarse*n2_window_coarse)
    nearneigh_oper = spzeros(prod(grsize_fine), prod(grsize_window_coarse))
    nearneigh_idxcoarse = zeros(Int64,size(nearneigh_oper,1))


    caind = CartesianIndices(grsize_fine)
    linind_fine = LinearIndices(grsize_fine)
    linind_coarse = LinearIndices(grsize_window_coarse)
    # @show grsize_window_coarse
    # @show linind_coarse
    i_coarse = MVector{Ndim,Int64}(undef)
    Nax = ndims(caind)
    # @show Nax
    for ci_fine in caind
        # loop over dimensions
        for d=1:Nax
            i_fine = ci_fine[d]
            di=div(i_fine-1,downscalefactor)
            ri=i_fine - di*downscalefactor
            i_coarse[d] = ri>=downscalefactor/2+1 ? di+2 : di+1           
        end

        # @show ci_fine
        # @show i_coarse
        irow = linind_fine[ci_fine]
        jcol = linind_coarse[CartesianIndex(i_coarse...)]
        # @show irow,jcol
        nearneigh_oper[irow,jcol] = 1.0     

        # track column index for gradient calculations
        nearneigh_idxcoarse[irow] = jcol

    end

    ##--------------------------------------------------
    ## get the interpolated velocity for the fine grid
    tmp_vel_fine = nearneigh_oper * vec(velcoarsegrd)
    vel_fine = reshape(tmp_vel_fine,grsize_fine...)

    if simtype==:cartesian
        if Ndim==2
            # set origin of the fine grid
            xinit = grd.x[ijk1coarse[1]]
            yinit = grd.y[ijk1coarse[2]]
            dh = grd.hgrid/downscalefactor
            # fine grid
            grdfine = Grid2DCart(hgrid=dh,cooinit=(xinit,yinit),
                                 grsize=grsize_fine)

        elseif Ndim==3
            # set origin of the fine grid
            xinit = grd.x[ijk1coarse[1]]
            yinit = grd.y[ijk1coarse[2]]
            zinit = grd.z[ijk1coarse[3]]
            dh = grd.hgrid/downscalefactor
            # fine grid
            grdfine = Grid3DCart(hgrid=dh,cooinit=(xinit,yinit,zinit),
                                 grsize=grsize_fine)
        end

    elseif simtype==:spherical
        
        if Ndim==2
            # set origin of the fine grid
            rinit = grd.r[ijk1coarse[1]]
            θinit = grd.θ[ijk1coarse[2]]
            dr = grd.Δr/downscalefactor
            dθ = grd.Δθ/downscalefactor
            # fine grid
            grdfine = Grid2DSphere(Δr=dr,Δθ=dθ,grsize=grsize_fine,cooinit=(rinit,θinit))

        elseif Ndim==3
            
            error("createfinegrid(): spherical coordinates in 3D still work in progress...")
            
        end
    end

    ##
    srcrefvars = SrcRefinVars(downscalefactor,
                              Tuple(ijk1coarse),
                              Tuple(outijkmin),Tuple(outijkmax),
                              nearneigh_oper,nearneigh_idxcoarse,vel_fine)

    return grdfine,srcrefvars
end

###########################################################


"""
$(TYPEDSIGNATURES)

Find closest node on a grid to a given point.
"""
function findenclosingbox(grd::AbstractGridEik,xyzsrc::AbstractVector)::AbstractMatrix{<:Int}

    grsize = SVector(grd.grsize...)
    xyzinit = SVector(grd.cooinit...)

    Ndim = length(xyzsrc)

    ## set an absolute tolerance for the remainder
    if typeof(grd)==Grid2DCart 
        atol = 1e-4*grd.hgrid #sqrt(eps())
        gridspac = (grd.hgrid,grd.hgrid)
    elseif typeof(grd)==Grid3DCart
        atol = 1e-4*grd.hgrid #sqrt(eps())
        gridspac = (grd.hgrid,grd.hgrid,grd.hgrid)
    elseif typeof(grd)==Grid2DSphere
        atol = 1e-4*min(grd.Δr,grd.Δθ)
        gridspac = (grd.Δr,grd.Δθ)
    end

    # xyzres = (xyzsrc.-xyzinit)./grd.hgrid
    # # make sure to get integers
    # ijkpt = floor.(Int64,xyzres) .+ 1 # .+1 julia indexing...

    xyzres = MVector{Ndim,Float64}(undef)
    remainder = MVector{Ndim,Float64}(undef)
    for d=1:Ndim
        xyzres[d],remainder[d] = divrem(xyzsrc[d]-xyzinit[d],gridspac[d])
    end
    ijkpt = floor.(Int64,xyzres) .+ 1 # .+1 julia indexing...            

  
    if all(isapprox.(remainder,0.0,atol=1e-8))   #all(remainder.==0.0)
        #####################################################
        ##
        ##  The point COINCIDES with a grid point
        ##
        ijksrccorn = SMatrix{1,Ndim,Int64}(ijkpt')
        return ijksrccorn

    else
        #####################################################
        ##
        ##  The point does NOT coincide with a grid point
        ##

        ## if at the edges of domain choose previous square...
        for d=1:Ndim
            if ijkpt[d]==grsize[d]
                ijkpt[d] -= 1 
            end
        end

        ## vvvv WRITE the stuff below in a better way!!!  vvvv
        ## ORDER MATTERS for the rest of the algorithm!!
        if Ndim==2
            ijksrccorn = SMatrix{2^Ndim,Ndim,Int64}([ijkpt';
                                                     (ijkpt .+ (1,0))';
                                                     (ijkpt .+ (0,1))';
                                                     (ijkpt .+ (1,1))'])
        elseif Ndim==3
            ijksrccorn = SMatrix{2^Ndim,Ndim,Int64}([ijkpt';
                                                     (ijkpt .+ (1,0,0))';
                                                     (ijkpt .+ (0,1,0))';
                                                     (ijkpt .+ (0,0,1))';
                                                     (ijkpt .+ (1,1,0))';
                                                     (ijkpt .+ (0,1,1))';
                                                     (ijkpt .+ (1,0,1))';
                                                     (ijkpt .+ (1,1,1))'])
        end
        return ijksrccorn
        
    end
end

#############################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(fmmvars::AbstractFMMVars,vel::Array{Float64,N},srcpos::AbstractVector,
                         grd::AbstractGridEik ) where N

    Ndim = length(srcpos)
    # get the position and velocity of corners around source
    ijkcorn = findenclosingbox(grd,srcpos)
    Ncorn = size(ijkcorn,1)

    if Ncorn==1
        ##
        ## Source on a grid point, re-allocate some arrays...
        ##        
        fmmvars.srcboxpar.ijksrc   = MMatrix{Ncorn,Ndim,Int64}(undef)
        fmmvars.srcboxpar.velcorn  = MVector{Ncorn,Float64}(undef)
        fmmvars.srcboxpar.distcorn = MVector{Ncorn,Float64}(undef)
    end

    ## Set some srcboxpar fields
    fmmvars.srcboxpar.ijksrc .= ijkcorn
    fmmvars.srcboxpar.xyzsrc .= srcpos

    ## corner position
    xyzpt = MVector{Ndim,Float64}(undef)

    ## set ttime around source 
    for l=1:Ncorn
        #i,j = ijcorn[l,:]
        ijkpt = CartesianIndex(Tuple(ijkcorn[l,:]))

        ## set status = accepted == 2
        fmmvars.status[ijkpt] = 2

        if typeof(grd)==Grid2DCart
            xyzpt .= (grd.x[ijkpt[1]],grd.y[ijkpt[2]])
        elseif typeof(grd)==Grid3DCart
            xyzpt .= (grd.x[ijkpt[1]],grd.y[ijkpt[2]],grd.z[ijkpt[3]])
        elseif typeof(grd)==Grid2DSphere
            xyzpt .= (grd.r[ijkpt[1]],grd.θ[ijkpt[2]])
        elseif typeof(grd)==Grid3DSphere
            xyzpt .= (grd.r[ijkpt[1]],grd.θ[ijkpt[2]],grd.φ[ijkpt[3]])
        end    

        # set the distance from corner to origin
        distcorn = distance2points(grd,xyzpt,srcpos)

        ## Set some srcboxpar fields
        fmmvars.srcboxpar.velcorn[l]  = vel[ijkpt]
        fmmvars.srcboxpar.distcorn[l] = distcorn

        # set the traveltime to corner
        fmmvars.ttime[ijkpt] = distcorn / vel[ijkpt]

    end

   
    ## If the source is exactly in the middle of four corners, move it by a bit...
    if Ncorn>1
        ## amount to move the source position
        if typeof(grd)==Grid2DCart 
            srcshift = 1e-4*grd.hgrid   #1e3*sqrt(eps())
            gridspac = (grd.hgrid,grd.hgrid)
        elseif typeof(grd)==Grid3DCart
            srcshift = 1e-4*grd.hgrid #sqrt(eps())
            gridspac = (grd.hgrid,grd.hgrid,grd.hgrid)
        elseif typeof(grd)==Grid2DSphere
            srcshift = 1e-4*min(grd.Δr,grd.Δθ)
            gridspac = (grd.Δr,grd.Δθ)
        elseif typeof(grd)==Grid3DSphere
            srcshift = 1e-4*min(grd.Δr,grd.Δθ,grd.Δφ)
            gridspac = (grd.Δr,grd.Δθ,grd.Δφ)
        end
        ## origin of coordinates
        xyzinit = SVector(grd.cooinit...)

        # check if the source position is halfway between grid points
        #   in each direction
        remainder = MVector{Ndim,Float64}(undef)
        for d=1:Ndim
            remainder[d] = rem(srcpos[d]-xyzinit[d],gridspac[d])
        end
        issrchalfway = remainder.≈gridspac./2.0

        # if any source coordinate is halfway, shift the source
        if any(issrchalfway)

            new_srcpos = copy(srcpos)
            for d=1:Ndim
                if issrchalfway[d]
                    ## if the source position is halfway between nodes
                    ##   along current direction [d], shift it away
                    new_srcpos[d] = srcpos[d] .+ srcshift
                end
            end
   
            whichdir = findall(issrchalfway)
            @warn "Shifting the source position along dimension(s) $(whichdir) by $(round(srcshift,sigdigits=3)) [1e-4*grd.hgrid] "
            fmmvars.srcboxpar.xyzsrc .= new_srcpos

            for l=1:Ncorn
                ijkpt = CartesianIndex(Tuple(ijkcorn[l,:]))
                ## corner position
                if typeof(grd)==Grid2DCart
                    xyzpt .= (grd.x[ijkpt[1]],grd.y[ijkpt[2]])
                elseif typeof(grd)==Grid3DCart
                    xyzpt .= (grd.x[ijkpt[1]],grd.y[ijkpt[2]],grd.z[ijkpt[3]])
                elseif typeof(grd)==Grid2DSphere
                    xyzpt .= (grd.r[ijkpt[1]],grd.θ[ijkpt[2]])
                elseif typeof(grd)==Grid3DSphere
                    xyzpt .= (grd.r[ijkpt[1]],grd.θ[ijkpt[2]],grd.φ[ijkpt[3]])
                end
                # set the distance from corner to origin
                fmmvars.srcboxpar.distcorn[l] = distance2points(grd,xyzpt,srcpos)
                # traveltime
                fmmvars.ttime[ijkpt] = fmmvars.srcboxpar.distcorn[l] / vel[ijkpt]
            end
        end
    end

    return
end 

#############################################
