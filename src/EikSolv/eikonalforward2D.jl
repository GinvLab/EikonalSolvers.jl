

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################

##########################################################################

"""
$(TYPEDSIGNATURES)

Calculate traveltime for 2D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the value of `extraparams.parallelkind`.

# Arguments
- `vel`: the 2D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model
- `extraparams` (optional) : a struct containing some "extra" parameters, namely
    * `parallelkind`: serial, Threads or Distributed run? (:serial, :sharedmem, :distribmem)
    * `refinearoundsrc`: whether to perform a refinement of the grid around the source location
    * `allowfixsqarg`: brute-force fix negative saqarg. Don't use this.
    * `manualGCtrigger`: trigger garbage collector (GC) manually at selected points.

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
    fmmvars = FMMvars2D(n1,n2,amIcoarsegrid=true, refinearoundsrc=extrapars.refinearoundsrc,
                        allowfixsqarg=extrapars.allowfixsqarg)

    ## pre-allocate discrete adjoint variables
    ##  No adjoint calculations
    ## adjvars = nothing

    if returntt
        ttGRPSRC = zeros(n1,n2,nsrc)
    end

    ## group of pre-selected sources
    for s=1:nsrc    
        ##
        ttFMM_hiord!(fmmvars,vel,view(coordsrc,s,:),grd,extrapars)

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

# Default constructor for forward calculations (adjvars=nothing)
ttFMM_hiord!(fmmvars::FMMvars2D,vel::Array{Float64,2},src::AbstractVector{Float64},grd::GridEik2D,
             extrapars::ExtraParams ) = ttFMM_hiord!(fmmvars,vel,src,grd,nothing,extrapars)

# Fast marching computations
function ttFMM_hiord!(fmmvars::FMMvars2D,vel::Array{Float64,2},src::AbstractVector{Float64},grd::GridEik2D,
                      adjvars::Union{AdjointVars2D,Nothing},extrapars::ExtraParams )
    
    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end

    if typeof(grd)==Grid2D
        simtype=:cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype=:spherical
    end
    
    ##==================================================##
    ###
    ###  Init FMM arrays
    ###  
    ## Time array
    ##
    # max possible val for type of ttime
    fmmvars.ttime[:,:] .= typemax(eltype(fmmvars.ttime)) 
    ##
    ## Status of nodes
    ##
    fmmvars.status[:,:] .= 0   ## set all to far
    ##==================================================

    ##==================================================
    if dodiscradj
        ##
        ##  Init discrete adjoint stuff
        ##
        nxy = prod(size(vel))
        ## Initialize size of derivative matrices as full nxy*nxy,
        ##   then rows corresponding to "source" points will be removed
        ##   later on
        adjvars.fmmord.vecDx.Nsize .= MVector(nxy,nxy)
        adjvars.fmmord.vecDy.Nsize .= MVector(nxy,nxy)
        ##
        adjvars.fmmord.vecDx.lastrowupdated[] = 0
        adjvars.fmmord.vecDy.lastrowupdated[] = 0
        adjvars.fmmord.vecDx.Nnnz[] = 0
        adjvars.fmmord.vecDy.Nnnz[] = 0
        adjvars.fmmord.onsrccols[:] .= false
        # for safety, zeroes the traveltime
        adjvars.fmmord.ttime[:] .= 0.0
        # for safety, zeroes the codes for derivatives
        adjvars.codeDxy[:,:] .= 0
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
        fmmvars_fine,adjvars_fine,grd_fine,vel_fine = runrefinementaroundsrc!(fmmvars,vel,src,grd,
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
        return fmmvars_fine,adjvars_fine,grd_fine,vel_fine #,ijsrc
    end
    
    return
end

###########################################################################

function runrefinementaroundsrc!(fmmvars::FMMvars2D,vel::Array{Float64,2},xysrc::AbstractVector{Float64},
                                 grd::GridEik2D,adjvars::Union{AdjointVars2D,Nothing},
                                 extrapars::ExtraParams)
                              
    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end
    if typeof(grd)==Grid2D
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
    end
    
    ##==================================================
    grd_fine,vel_fine,srcrefvars = createfinegrid(grd,xysrc,vel)

    if simtype==:cartesian
        n1_fine,n2_fine = grd_fine.nx,grd_fine.ny
    elseif simtype==:spherical
        n1_fine,n2_fine = grd_fine.nr,grd_fine.nθ
    end
    
    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars_fine = FMMvars2D(n1_fine,n2_fine,amIcoarsegrid=false,refinearoundsrc=false,
                             allowfixsqarg=extrapars.allowfixsqarg)
    
    ##==================================================
    # init ttime 
    inittt = typemax(eltype(fmmvars_fine.ttime))
    fmmvars_fine.ttime[:,:] .= inittt
    # Status of nodes
    fmmvars_fine.status[:,:] .= 0   ## set all to far
    ##==================================================
    if dodiscradj
        ## pre-allocate adjoint variables for the fine grid
        adjvars_fine = AdjointVars2D(n1_fine,n2_fine)
        ##
        ##  discrete adjoint: init stuff
        ## 
        adjvars_fine.fmmord.vecDx.lastrowupdated[] = 0
        adjvars_fine.fmmord.vecDy.lastrowupdated[] = 0
        adjvars_fine.fmmord.vecDx.Nnnz[] = 0
        adjvars_fine.fmmord.vecDy.Nnnz[] = 0
        adjvars_fine.fmmord.onsrccols[:] .= false
        adjvars_fine.fmmord.onhpoints[:] .= false
        # for safety, zeroes the traveltime
        adjvars_fine.fmmord.ttime[:] .= 0.0
        # for safety, zeroes the codes for derivatives
        adjvars_fine.codeDxy[:,:] .= 0

    else
        adjvars_fine = nothing

    end
    ##==================================================

    ## source location, etc.      
    ## REGULAR grid
    if simtype==:cartesian
        #ijsrc_fine = sourceboxloctt!(fmmvars_fine,vel_fine,xysrc,grd_fine )
        sourceboxloctt!(fmmvars_fine,vel_fine,xysrc,grd_fine )
               
    elseif simtype==:spherical
        #ijsrc_fine = sourceboxloctt_sph!(fmmvars_fine,vel_fine,xysrc,grd_fine )
        sourceboxloctt_sph!(fmmvars_fine,vel_fine,xysrc,grd_fine )
       
    end

    ##==================================================================
    ## Run forward simulation (FMM) within the fine grid
    #ijsrc_coarse = ttFMM_core!(fmmvars_fine,vel_fine,grd_fine,ijsrc_fine,adjvars_fine,
    ttFMM_core!(fmmvars_fine,vel_fine,grd_fine,adjvars_fine,
                fmmvars_coarse=fmmvars, adjvars_coarse=adjvars,srcrefvars=srcrefvars)


    
    ##==================================================================

    # if dogradsrcpos
    #     ##======================================================
    #     ## Derivatives w.r.t. source position.  !!!FINE GRID!!!
    #     ##======================================================
    #     @show extrema(adjvars_fine.fmmord.vecDx.v)
    #     ∂u_h∂x_s = calc_∂u_h∂x_s_finegrid(fmmvars_fine,adjvars_fine,xysrc,vel_fine,grd_fine)

    #     println("hello from runrefinementaroundsrc(): ∂u_h∂x_s = $∂u_h∂x_s")
    # end

    
    # ##======================================================
    # if dodiscradj  #&& dogradsrcpos==false
    #     ## 
    #     ## DISCRETE ADJOINT WORKAROUND FOR DERIVATIVES !!!COARSE GRID!!!
    #     ##
    #     # derivative codes
    #     idD = MVector(0,0)
    #     # number of source points
    #     naccinit = size(ijsrc,1)

    #     # How many initial points to skip, considering them as "onsrc"?
    #     skipnptsDxy = 4
        
    #     ## pre-compute some of the mapping between fmm and orig order
    #     for i=1:naccinit
    #         ifm = adjvars.idxconv.lfmm2grid[i]
    #         adjvars.idxconv.lgrid2fmm[ifm] = i
    #     end
        
    #     # loop on points "on" source
    #     for l=1:naccinit

    #         if l<=skipnptsDxy
    #             #################################################################
    #             ##  We are on a source node
    #             ###############################
    #             # add true for this point being on src
    #             adjvars.fmmord.onsrccols[l] = true
    #             # remove one row because of point being on source!
    #             adjvars.fmmord.vecDx.Nsize[1] -= 1
    #             adjvars.fmmord.vecDy.Nsize[1] -= 1
    #             #################################################################

    #         else

    #             ## "reconstruct" derivative stencils from known FMM order and arrival times
    #             derivaroundsrcfmm2D!(l,adjvars.idxconv,idD)

    #             if idD==[0,0]
    #                 #################################################################
    #                 ##  We are on a source node
    #                 ###############################
    #                 # add true for this point being on src
    #                 adjvars.fmmord.onsrccols[l] = true
    #                 # remove one row because of point being on source!
    #                 adjvars.fmmord.vecDx.Nsize[1] -= 1
    #                 adjvars.fmmord.vecDy.Nsize[1] -= 1
    #                 #################################################################

    #             else
    #                 l_fmmord = adjvars.idxconv.lfmm2grid[l]
    #                 adjvars.codeDxy[l_fmmord,:] .= idD
    #             end

    #         end
    #     end
    # end # dodiscradj
    # ##======================================================

    if dodiscradj
        return ijsrc_coarse,fmmvars_fine,adjvars_fine,grd_fine,vel_fine
    else
        return ijsrc_coarse
    end
end

###########################################################################

function initnorefsrc(fmmvars::FMMvars2D,vel::Array{Float64,2},src::AbstractVector{Float64},
                      grd::GridEik2D,adjvars::Union{AdjointVars2D,Nothing}  )

    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end

    if typeof(grd)==Grid2D
        simtype=:cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype=:spherical
    end
    
    ## source location, etc.      
    ## REGULAR grid
    if simtype==:cartesian
        #ijsrc = sourceboxloctt!(fmmvars,vel,src,grd )
        sourceboxloctt!(fmmvars,vel,src,grd )
    elseif simtype==:spherical
        #ijsrc = sourceboxloctt_sph!(fmmvars,vel,src,grd )
        sourceboxloctt_sph!(fmmvars,vel,src,grd )
    end

    return #ijsrc
end

#################################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 2D using traditional stencils on regular grid. 
"""
function ttFMM_core!(fmmvars::FMMvars2D,vel::Array{Float64,2},grd::GridEik2D,
                     #ijsrc::AbstractArray{<:Integer,2},
                     adjvars::Union{AdjointVars2D,Nothing} ;
                     fmmvars_coarse::Union{FMMvars2D,Nothing}=nothing,
                     adjvars_coarse::Union{AdjointVars2D,Nothing}=nothing,
                     srcrefvars::Union{SrcRefinVars2D,Nothing}=nothing)

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
        i1coarse = srcrefvars.ij1coarse[1]
        j1coarse = srcrefvars.ij1coarse[2]
        outxmin = srcrefvars.outxyminmax[1]
        outxmax = srcrefvars.outxyminmax[2]
        outymin = srcrefvars.outxyminmax[3]
        outymax = srcrefvars.outxyminmax[4]
    end

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

    ##----------------------------------------------------------------------
    ## fmmvars.srcboxpar.ijsrc can be a field of either
    ##  - a SrcRefinVars2D if we are in the coarse grid with no refinement,
    ##     or in the fine grid
    ##  - a SourcePtsFromFineGrid if we are in the coarse grid after the
    ##     FMM has been already run in the fine grid (multiple source points)
    ##
    ijsrc = fmmvars.srcboxpar.ijsrc

    ##=========================================
    if dodiscradj
        ## number of accepted points       
        naccinit = size(ijsrc,1)        
        n1,n2 = size(vel)

        # discrete adjoint: first visited points in FMM order
        ttfirstpts = Vector{Float64}(undef,naccinit)
        for l=1:naccinit
            i,j = ijsrc[l,1],ijsrc[l,2]
            ttij = fmmvars.ttime[i,j]
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
            ia,ja = ijsrc[l,1],ijsrc[l,2]
            # adj stuff
            adjvars.idxconv.lfmm2grid[l] = cart2lin2D(ia,ja,n1)
            
            ######################################
            # store arrival time for first points in FMM order
            ttgrid = fmmvars.ttime[ia,ja]
            ## The following to avoid a singular upper triangular matrix in the
            ##  adjoint equation. Otherwise there will be a row of only zeros in the LHS.
            if ttgrid==0.0
                adjvars.fmmord.ttime[l] = 0.0 #eps()
                @warn("Source is exactly on a node, spurious results may appear. Work is in progress to fix the problem.\n Set smoothgradsourceradius>0 to mitigate the issue")
            else
                adjvars.fmmord.ttime[l] = ttgrid
            end

            ##======================================================
            ##  We are on a source node
            ##================================
            # add true for this point being on src
            adjvars.fmmord.onsrccols[l] = true
            # remove one row because of point being on source!
            adjvars.fmmord.vecDx.Nsize[1] -= 1
            adjvars.fmmord.vecDy.Nsize[1] -= 1

        end
    end # if dodiscradj
    ##======================================================


    
    ##======================================================
    ## 
    ## Time array
    ##
    # inittt = 1e30
    # #ttime = Array{Float64}(undef,n1,n2)
    # fmmvars.ttime[:,:] .= inittt
    # ##
    # ## Status of nodes
    # ##
    # #status = Array{Int64}(undef,n1,n2)
    # fmmvars.status[:,:] .= 0   ## set all to far
    ##
    ptij = MVector(0,0)
    # derivative codes
    idD = MVector(0,0)

    ##======================================================
    if isthisrefinementsrc
        ijsrc_coarse = Matrix{Int64}(undef,n1*n2,2)
        counter_ijsrccoarse::Int64 = 1
        counter_adjcoarse::Int64 = 1
    end

    ##======================================================
    if dodiscradj
        # init row stuff
        colindsonsrc = MVector(0)
        #colvalsonsrc = SA[1.0]
    end    
    ##======================================================

    ## number of accepted points       
    naccinit = size(ijsrc,1)        

    ##======================================================
    ## Only while running in the fine grid
    ##======================================================
    if isthisrefinementsrc
        for l=1:naccinit
            ia = ijsrc[l,1]
            ja = ijsrc[l,2]

            oncoa,ia_coarse,ja_coarse = isacoarsegridnode(ia,ja,downscalefactor,i1coarse,j1coarse)

            if oncoa
                fmmvars_coarse.ttime[ia_coarse,ja_coarse]  = fmmvars.ttime[ia,ja]
                fmmvars_coarse.status[ia_coarse,ja_coarse] = fmmvars.status[ia,ja]
                ijsrc_coarse[counter_ijsrccoarse,:] .= (ia_coarse,ja_coarse)
                ## update counter
                counter_ijsrccoarse += 1

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
    # per = [[(1, 0), (0, 1), (-1, 0), (0, -1)], [(1, 0), (0, 1), (0, -1), (-1, 0)], [(1, 0), (-1, 0), (0, 1), (0, -1)], [(1, 0), (-1, 0), (0, -1), (0, 1)], [(1, 0), (0, -1), (0, 1), (-1, 0)], [(1, 0), (0, -1), (-1, 0), (0, 1)], [(0, 1), (1, 0), (-1, 0), (0, -1)], [(0, 1), (1, 0), (0, -1), (-1, 0)], [(0, 1), (-1, 0), (1, 0), (0, -1)], [(0, 1), (-1, 0), (0, -1), (1, 0)], [(0, 1), (0, -1), (1, 0), (-1, 0)], [(0, 1), (0, -1), (-1, 0), (1, 0)], [(-1, 0), (1, 0), (0, 1), (0, -1)], [(-1, 0), (1, 0), (0, -1), (0, 1)], [(-1, 0), (0, 1), (1, 0), (0, -1)], [(-1, 0), (0, 1), (0, -1), (1, 0)], [(-1, 0), (0, -1), (1, 0), (0, 1)], [(-1, 0), (0, -1), (0, 1), (1, 0)], [(0, -1), (1, 0), (0, 1), (-1, 0)], [(0, -1), (1, 0), (-1, 0), (0, 1)], [(0, -1), (0, 1), (1, 0), (-1, 0)], [(0, -1), (0, 1), (-1, 0), (1, 0)], [(0, -1), (-1, 0), (1, 0), (0, 1)], [(0, -1), (-1, 0), (0, 1), (1, 0)]]


    # p = 1 # 8
    # neigh = @SMatrix [ per[p][i][j] for i=1:4,j=1:2 ]
    
    neigh = SA[ 0  1;
                1  0;
                0 -1;
               -1  0] 
                

    ## >>> BEST so far!!! <<<
    # neigh = SA[0  1;
    #            1  0;
    #            0 -1;
    #           -1  0]
    

    ## pre-allocate
    #tmptt::Float64 = 0.0 
    
    ##-----------------------------------------
    ## construct initial narrow band
    for l=1:naccinit ##
        
        for ne=1:4 ## four potential neighbors

            i = ijsrc[l,1] + neigh[ne,1]
            j = ijsrc[l,2] + neigh[ne,2]

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
               
            # elseif fmmvars.status[i,j]==1 ## narrow band
            #     ##======================================
            #     ## NNNNEEEEWWWW !!!!
            #     #@show "Hello, construct initial narrow band $i, $j"
            #     # update the traveltime for this point
            #     tmptt = calcttpt_2ndord!(fmmvars,vel,grd,i,j,idD)

            #     # get handle
            #     han = cart2lin2D(i,j,n1)
            #     # update the traveltime for this point in the heap
            #     update_node_minheap!(fmmvars.bheap,tmptt,han)

            #     if dodiscradj
            #         # codes of chosen derivatives for adjoint
            #         adjvars.codeDxy[han,:] .= idD
            #     end
            #     ##======================================
                
            end ## if fmmvars.status[i,j]==0 ## far
        end ## for ne=1:4
    end


    
    ######################################################
    ## main FMM loop
    ######################################################
    firstwarning=true
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


        ##===================================
        # Discrete adjoint
        if dodiscradj
            # store the linear index of FMM order
            adjvars.idxconv.lfmm2grid[node] = cart2lin2D(ia,ja,n1)
            # store arrival time for first points in FMM order
            adjvars.fmmord.ttime[node] = tmptt  
        end

        ##======================================================
        ## Are we running a refinement around the source? If so, 
        ##    UPDATE the COARSE GRID
        ##======================================================
        if isthisrefinementsrc
            oncoa,ia_coarse,ja_coarse = isacoarsegridnode(ia,ja,downscalefactor,i1coarse,j1coarse)

            if oncoa
                fmmvars_coarse.ttime[ia_coarse,ja_coarse]  = fmmvars.ttime[ia,ja]
                fmmvars_coarse.status[ia_coarse,ja_coarse] = fmmvars.status[ia,ja]
                ijsrc_coarse[counter_ijsrccoarse,:] .= (ia_coarse,ja_coarse)
                ## update counter
                counter_ijsrccoarse += 1
                
                if dodiscradj
                    ## next line on FINE grid
                    adjvars.fmmord.onhpoints[node] = true

                    # Discrete adjoint on COARSE grid
                    # go from cartesian (i,j) to linear
                    adjvars_coarse.idxconv.lfmm2grid[counter_adjcoarse] = cart2lin2D(ia_coarse,ja_coarse,
                                                                                     adjvars_coarse.idxconv.nx)
                    # store arrival time for first points in FMM order
                    adjvars_coarse.fmmord.ttime[counter_adjcoarse] = fmmvars.ttime[ia,ja]
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
            #  if (ia==n1) || (ia==1) || (ja==n2) || (ja==1)
            if (ia==1 && !outxmin) || (ia==n1 && !outxmax) || (ja==1 && !outymin) || (ja==n2 && !outymax)
            
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
                    createsparsederivativematrices!(grd,adjvars)
                end
                ##==============================================================

                ## delete current narrow band to avoid problems when returned to coarse grid
                fmmvars_coarse.status[fmmvars_coarse.status.==1] .= 0

                ## re-allocate srcboxpar.ijsrc of type SourcePtsFromFineGrid (mutable struct)
                fmmvars_coarse.srcboxpar.ijsrc = ijsrc_coarse[1:counter_ijsrccoarse-1,:]
                return #ijsrc_coarse[1:counter_ijsrccoarse-1,:]
            end
            
        end # if isthisrefinementsrc      
        ##===================================
        

        ##===========================================
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
        end # for ne=1:4

    end # for node=naccinit+1:totnpts 

    ##======================================================
    if dodiscradj
        createsparsederivativematrices!(grd,adjvars)
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
    HUGE = typemax(eltype(vel)) #1.0e30
    codeD[:] .= 0 # integers

    ## 2 axis, x and y
    for axis=1:2
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## 2 directions: forward or backward
        for l=1:2

            ## map the 4 cases to an integer as in linear indexing...
            lax = l + 2*(axis-1)
            if lax==1 # axis==1
                ish = 1 #1
                jsh = 0 #0
            elseif lax==2 # axis==1
                ish = -1  #-1
                jsh = 0  #0
            elseif lax==3 # axis==2
                ish = 0 #0
                jsh = 1 #1
            elseif lax==4 # axis==2
                ish = 0 # 0
                jsh = -1 #-1
            end

            ## check if on boundaries
            isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                                    
            ##==== 1st order ================
            if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
                ## first test value
                testval1 = fmmvars.ttime[i+ish,j+jsh]

                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only!!!
                    chosenval1 = testval1
                    use1stord = true

                    # save derivative choices
                    axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                    
                    ##==== 2nd order ================
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    if !isonb2nd && fmmvars.status[i+ish2,j+jsh2]==2 ## 2==accepted
                        # second test value
                        testval2 = fmmvars.ttime[i+ish2,j+jsh2]
                        ## pick the lowest value of the two
                        ##    compare to chosenval 1, *not* 2!!
                        ## This because the direction has already been chosen
                        ##  at the line "testval1<chosenval1"
                        if testval2<chosenval1 ## < only!!!
                            chosenval2 = testval2
                            use2ndord = true 
                            # save derivative choices
                            axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)
                        else
                            chosenval2=HUGE
                            # below in case first direction gets 2nd ord
                            #   but second direction, with a smaller testval1,
                            #   does *not* get a second order
                            use2ndord=false # this is needed!
                            # save derivative choices
                            axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                        end

                    end ##==== END 2nd order ================
                    
                end 
            end ##==== END 1st order ================


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
            gamma += curalpha * tmpa2^2 ## see init of gamma : - slowcurpt^2

        elseif use1stord # first order
            ## curalpha: make sure you multiply only times the
            ##   current alpha for beta and gamma...
            curalpha = 1.0/deltah^2 
            alpha += curalpha
            beta  += ( -2.0*curalpha * chosenval1 )
            gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
        end

    end ## for axis=1:2

    ## compute discriminant 
    sqarg = beta^2-4.0*alpha*gamma

    ## To get a non-negative discriminant, need to fulfil:
    ##    (tx-ty)^2 - 2*s^2/curalpha <= 0
    ##    where tx,ty can be
    ##     t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order
    ##     t? = chosenval1  if 1st order 
    ##

    ##=========================================================================================
    ## If discriminant is negative (probably because of sharp contrasts in
    ##  velocity) revert to 1st order for both x and y
    # if sqarg<0.0

    #     begin    
    #         codeD[:] .= 0 # integers
    #         alpha = 0.0
    #         beta  = 0.0
    #         gamma = - slowcurpt^2 ## !!!!

    #         ## 2 directions
    #         for axis=1:2
                
    #             use1stord = false
    #             chosenval1 = HUGE
                
    #             ## two sides for each direction
    #             for l=1:2

    #                 ## map the 4 cases to an integer as in linear indexing...
    #                 lax = l + 2*(axis-1)
    #                 if lax==1 # axis==1
    #                     ish = 1
    #                     jsh = 0
    #                 elseif lax==2 # axis==1
    #                     ish = -1
    #                     jsh = 0
    #                 elseif lax==3 # axis==2
    #                     ish = 0
    #                     jsh = 1
    #                 elseif lax==4 # axis==2
    #                     ish = 0
    #                     jsh = -1
    #                 end

    #                 ## check if on boundaries
    #                 isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                    
    #                 ## 1st order
    #                 if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
    #                     testval1 = fmmvars.ttime[i+ish,j+jsh]
    #                     ## pick the lowest value of the two
    #                     if testval1<chosenval1 ## < only
    #                         chosenval1 = testval1
    #                         use1stord = true

    #                         # save derivative choices
    #                         axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

    #                     end
    #                 end
    #             end # end two sides

    #             ## spacing
    #             deltah = Δh[axis]
                
    #             if use1stord # first order
    #                 ## curalpha: make sure you multiply only times the
    #                 ##   current alpha for beta and gamma...
    #                 curalpha = 1.0/deltah^2 
    #                 alpha += curalpha
    #                 beta  += ( -2.0*curalpha * chosenval1 )
    #                 gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
    #             end
    #         end
            
    #         ## recompute sqarg
    #         sqarg = beta^2-4.0*alpha*gamma

    #     end ## begin...

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
    # end ## if sqarg<0.0
    ##=========================================================================================

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

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(fmmvars::FMMvars2D,vel::Array{Float64,2},srcpos::AbstractVector,
                         grd::Grid2D )

    ## source location, etc.      
    mindistsrc = 10.0*eps()
    xsrc,ysrc=srcpos[1],srcpos[2]

    # get the position and velocity of corners around source
    _,velcorn,ijsrc = bilinear_interp(vel,grd,srcpos,outputcoeff=true)
    

    ## Set srcboxpar
    Ncorn = size(ijsrc,1)
    fmmvars.srcboxpar.ijsrc .= ijsrc
    fmmvars.srcboxpar.xysrc .= srcpos
    fmmvars.srcboxpar.velcorn .= velcorn

    ## set ttime around source ONLY FOUR points!!!
    for l=1:Ncorn
        i,j = ijsrc[l,:]

        ## set status = accepted == 2
        fmmvars.status[i,j] = 2

        ## corner position 
        xp = grd.x[i] 
        yp = grd.y[j]

        # set the distance from corner to origin
        distcorn = sqrt((xsrc-xp)^2+(ysrc-yp)^2)
        fmmvars.srcboxpar.distcorn[l] = distcorn

        # set the traveltime to corner
        fmmvars.ttime[i,j] = distcorn / vel[i,j] #velsrc


        # ##-----------
        # ## old way
        # ii = Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
        # jj = Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1
        # fmmvars.ttime[i,j] = distcorn / vel[ii,jj]

    end

    # @show fmmvars.srcboxpar.distcorn
    # @show fmmvars.ttime[ijsrc[:,1],ijsrc[:,2]]

    # println("=== Messing up the ttime at the source points ===")
    # for l=1:Ncorn
    #     i,j = ijsrc[l,:]
    #     fmmvars.ttime[i,j] = fmmvars.ttime[ijsrc[2,1],ijsrc[2,2]]
    # end

    # save src reg points
    # xysrcpts = [grd.x[ijsrc[i,j]] for i=1:4,j=1:2]
    # println("Saving src region points...")
    # h5open("srcpoints.h5","w") do fl
    #     write(fl,"srcpts", xysrcpts)
    # end


    #println("ijsrc from bilin. interp.: $ijsrc ")
    

    ###########################################
    # ## regular grid
    # ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
    # rx = xsrc-grd.x[ix]
    # ry = ysrc-grd.y[iy]
 
    # #halfg = 0.0 #hgrid/2.0
    # # Euclidean distance
    # dist = sqrt(rx^2+ry^2)

    # if dist<=mindistsrc
    #     println("WARNING: Source exactly on a grid point!\n \
    #                                   ijsrc: $ijsrc")
    # end

    # halfg = 0.0

    # # pre-allocate array of indices for source
    # ijsrc = @MMatrix zeros(Int64,2,4)

    # ## four points
    # if rx>=halfg
    #     srci = (ix,ix+1)
    # else
    #     srci = (ix-1,ix)
    # end
    # if ry>=halfg
    #     srcj = (iy,iy+1)
    # else
    #     srcj = (iy-1,iy)
    # end
    
    # l=1
    # for j=1:2, i=1:2
    #     ijsrc[:,l] .= (srci[i],srcj[j])
    #     l+=1
    # end
    # #println("ijsrc from four points:    $(ijsrc') ")

    # ## set ttime around source ONLY FOUR points!!!
    # for l=1:size(ijsrc,2)
    #     i = ijsrc[1,l]
    #     j = ijsrc[2,l]

    #     ## set status = accepted == 2
    #     fmmvars.status[i,j] = 2
    #     fmmvars.srcboxpar.ijsrc[l,:] .= (i,j)
        
    #     ## regular grid, the velocity must be the same for the 4 points
    #     xp = grd.x[i] 
    #     yp = grd.y[j]
    #     #@show "4p",xp,yp
    #     ii = Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
    #     jj = Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1             
    #     ## set traveltime for the source
    #     dist = sqrt((xsrc-xp)^2+(ysrc-yp)^2)
    #     tt = dist / vel[ii,jj]
    #     #fmmvars.ttime[i,j] = tt ##sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]
    #     #println("$l ttime from four points   : $(tt), $dist ")
        
    # end
    ###########################################

    
    return #ijsrc
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

    #halfg = 0.0 #hgrid/2.0
    ## distance in POLAR COORDINATES
    ## sqrt(r1^+r2^2 - 2*r1*r2*cos(θ1-θ2))
    r1=rsrc
    r2=grd.r[ir]
    dist = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[iθ]) )  #sqrt(rr^2+grd.r[ir]^2*rθ^2)
    #@show dist,src,rr,rθ
    if dist<=mindistsrc
        println("WARNING: Source exactly on a grid point!\n \
                 ijsrc: $ijsrc")
    end

    # get the interpolated velocity
    coeff,velcorn,ijsrc = bilinear_interp_sph(vel,grd,srcpos,outputcoeff)
    
    ## Set srcboxpar
    Ncorn = size(ijsrc,1)
    fmmvars.srcboxpar.ijsrc .= ijsrc
    fmmvars.srcboxpar.coeff .= coeff
    fmmvars.srcboxpar.velcorn .= velcorn # [vel[ijsrc[i,:]] for i=1:Ncorn]
    velsrc = dot(coeff,velcorn)

    
    ## set ttime around source ONLY FOUR points!!!
    for l=1:Ncorn
        i = ijsrc[1,l]
        j = ijsrc[2,l]

        ## set status = accepted == 2
        fmmvars.status[i,j] = 2
        
        r1=rsrc
        r2=grd.r[ii]
        # distance
        distp = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[iθ]))
        ## set traveltime for the source
        fmmvars.ttime[i,j] = distp / velsrc
    end

    return #ijsrc
end

#########################################################

function createsparsederivativematrices!(grd,adjvars)

    if typeof(grd)==Grid2D
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
    end

    ptij = MVector(0,0)

    # pre-determine derivative coefficients for positive codes (0,+1,+2)
    if simtype==:cartesian
        n1,n2 = grd.nx,grd.ny
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
        n1,n2 = grd.nr,grd.nθ
        Δr = grd.Δr
        ## DEG to RAD !!!!
        Δarc = [grd.r[i] * deg2rad(grd.Δθ) for i=1:grd.nr]
        # coefficients
        coe_r_1st = [-1.0/Δr  1.0/Δr]
        coe_r_2nd = [-3.0/(2.0*Δr)  4.0/(2.0*Δr) -1.0/(2.0*Δr) ]
        coe_θ_1st = [-1.0 ./ Δarc  1.0 ./ Δarc ]
        coe_θ_2nd = [-3.0./(2.0*Δarc)  4.0./(2.0*Δarc)  -1.0./(2.0*Δarc) ]

        allcoeffx = CoeffDerivSpherical2D( coe_r_1st, coe_r_2nd )
        allcoeffy = CoeffDerivSpherical2D( coe_θ_1st, coe_θ_2nd )

    end

    ##--------------------------------------------------------
    ## pre-compute the mapping between fmm and original order
    n12 = n1*n2
    nptsonsrc = count(adjvars.fmmord.onsrccols)
    #nptsonh   = count(adjvars.fmmord.onhpoints)

    for i=1:n12
        # ifm = index from fast marching ordering
        ifm = adjvars.idxconv.lfmm2grid[i]
        if ifm==0
            # ifm is zero = end of indices for fmm ordering [lfmm2grid=zeros(Int64,nxyz)]
            Nnnzsteps = i-1
            # remove rows corresponding to points in the source region
            adjvars.fmmord.vecDx.Nsize[1] = Nnnzsteps - nptsonsrc
            adjvars.fmmord.vecDx.Nsize[2] = Nnnzsteps
            adjvars.fmmord.vecDy.Nsize[1] = Nnnzsteps - nptsonsrc
            adjvars.fmmord.vecDy.Nsize[2] = Nnnzsteps
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

    colinds = MVector(0,0,0)
    colvals = MVector(0.0,0.0,0.0)
    idxperm = MVector(0,0,0)

    #for irow=startloop:n12
    for irow=1:adjvars.fmmord.vecDx.Nsize[1]
        
        # compute the coefficients for X  derivatives
        setcoeffderiv2D!(adjvars.fmmord.vecDx,irow,adjvars.idxconv,adjvars.codeDxy,allcoeffx,ptij,
                         colinds,colvals,idxperm,nptsfixedtt, axis=:X,simtype=simtype)

        
        # compute the coefficients for Y derivatives
        setcoeffderiv2D!(adjvars.fmmord.vecDy,irow,adjvars.idxconv,adjvars.codeDxy,allcoeffy,ptij,
                         colinds,colvals,idxperm,nptsfixedtt, axis=:Y,simtype=simtype)
        
    end

    return
end

##########################################################################

function createfinegrid(grd,xysrc,vel)

    if typeof(grd)==Grid2D
        simtype=:cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype=:spherical
    end
    ##
    ## 2x10 nodes -> 2x50 nodes
    ##
    downscalefactor::Int = 0
    noderadius::Int = 0
    
    downscalefactor = 5
    ## 2 instead of 5 for adjoint to avoid messing up...
    ## if noderadius is not the same for forward and adjoint,
    ##   then troubles with comparisons with brute-force fin diff
    ##   will occur...
    noderadius = 2 

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    if simtype==:cartesian
        n1_coarse,n2_coarse = grd.nx,grd.ny
        ixsrcglob,iysrcglob = findclosestnode(xysrc[1],xysrc[2],grd.xinit,grd.yinit,grd.hgrid)

        rx = xysrc[1]-grd.x[ixsrcglob]
        ry = xysrc[2]-grd.y[iysrcglob]

        ## four points
        ijsrcpts = @MMatrix zeros(Int64,4,2)
        halfg = 0.0
        if rx>=halfg
            srci = (ixsrcglob,ixsrcglob+1)
        else
            srci = (ixsrcglob-1,ixsrcglob)
        end
        if ry>=halfg
            srcj = (iysrcglob,iysrcglob+1)
        else
            srcj = (iysrcglob-1,iysrcglob)
        end
        
        l=1
        for j=1:2, i=1:2
            ijsrcpts[l,:] .= (srci[i],srcj[j])
            l+=1
        end
        
    elseif simtype==:spherical
        n1_coarse,n2_coarse = grd.nr,grd.nθ
        ixsrcglob,iysrcglob = findclosestnode_sph(xysrc[1],xysrc[2],grd.rinit,grd.θinit,grd.Δr,grd.Δθ)
        error(" ttaroundsrc!(): spherical coordinates still work in progress...")
    end
    
    ##
    ## Define chunck of coarse grid
    ##
    # i1coarsevirtual = ixsrcglob - noderadius
    # i2coarsevirtual = ixsrcglob + noderadius
    # j1coarsevirtual = iysrcglob - noderadius
    # j2coarsevirtual = iysrcglob + noderadius
    i1coarsevirtual = minimum(ijsrcpts[1,:]) - noderadius
    i2coarsevirtual = maximum(ijsrcpts[1,:]) + noderadius
    j1coarsevirtual = minimum(ijsrcpts[2,:]) - noderadius
    j2coarsevirtual = maximum(ijsrcpts[2,:]) + noderadius
    # if hitting borders
    outxmin = i1coarsevirtual<1
    outxmax = i2coarsevirtual>n1_coarse
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>n2_coarse
    outxmin ? i1coarse=1         : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=n1_coarse : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1         : j1coarse=j1coarsevirtual
    outymax ? j2coarse=n2_coarse : j2coarse=j2coarsevirtual
    
    ##
    ## Refined grid parameters
    ##
    # fine grid size
    n1_fine = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    n2_fine = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
   
    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)    

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    nearneigh_oper = spzeros(n1_fine*n2_fine, n1_coarse*n2_coarse)
    vel_fine = Array{Float64}(undef,n1_fine,n2_fine)
    for j=1:n2_fine
        for i=1:n1_fine
            di=div(i-1,downscalefactor)
            ri=i-di*downscalefactor
            ii = ri>=downscalefactor/2+1 ? di+2 : di+1

            dj=div(j-1,downscalefactor)
            rj=j-dj*downscalefactor
            jj = rj>=downscalefactor/2+1 ? dj+2 : dj+1

            vel_fine[i,j] = velcoarsegrd[ii,jj]

            # compute the matrix acting as nearest-neighbor operator (for gradient calculations)
            f = i  + (j-1)*n1_fine
            c = ii + (jj-1)*n1_coarse
            nearneigh_oper[f,c] = 1.0
        end
    end

    
    if simtype==:cartesian
        # set origin of the fine grid
        xinit = grd.x[i1coarse]
        yinit = grd.y[j1coarse]
        dh = grd.hgrid/downscalefactor
        # fine grid
        grdfine = Grid2D(hgrid=dh,xinit=xinit,yinit=yinit,nx=n1_fine,ny=n2_fine)

    elseif simtype==:spherical
        # set origin of the fine grid
        rinit = grd.r[i1coarse]
        θinit = grd.θ[j1coarse]
        dr = grd.Δr/downscalefactor
        dθ = grd.Δθ/downscalefactor
        # fine grid
        grdfine = Grid2DSphere(Δr=dr,Δθ=dθ,nr=n1_fine,nθ=n2_fine,rinit=rinit,θinit=θinit)
    end

    srcrefvars = SrcRefinVars2D(downscalefactor,(i1coarse,j1coarse),(outxmin,outxmax,outymin,outymax))

    return grdfine,vel_fine,srcrefvars,nearneigh_oper
end

###########################################################################
