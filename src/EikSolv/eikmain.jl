

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################

##########################################################################

"""
$(TYPEDSIGNATURES)

Calculate traveltime for 2D or 3D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the value of `extraparams.parallelkind`.

# Arguments
- `vel`: the 2D/3D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y[,z]), a 2-column (3-column) array
- `coordrec`: the coordinates of the receiver(s) (x,y[,z]) for each single source, a vector of 2-column (3-column) arrays 
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model
- `extraparams` (optional) : a struct containing some "extra" parameters, namely
    * `parallelkind`: serial, Threads or Distributed run? (:serial, :sharedmem, :distribmem)
    * `refinearoundsrc`: whether to perform a refinement of the grid around the source location
    * `grdrefpars`: refined grid around the source parameters (`downscalefactor` and `noderadius`)
    * `allowfixsqarg`: brute-force fix negative saqarg. Don't use this.
    * `manualGCtrigger`: trigger garbage collector (GC) manually at selected points.

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

    """

function eiktraveltime(vel::Array{Float64,N},grd::AbstractGridEik,coordsrc::Array{Float64,N},
                       coordrec::Vector{Array{Float64,N}} ; returntt::Bool=false,
                       extraparams::Union{ExtraParams,Nothing}=nothing  ) where N
        
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
                @async ttime[igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
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
                ttime[igrs],ttpicks[igrs] = ttforwsomesrc2D(vel,view(coordsrc,igrs,:),
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
            ttime,ttpicks = ttforwsomesrc2D(vel,coordsrc,coordrec,grd,extraparams,
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

#######################################################

"""
$(TYPEDSIGNATURES)

  Compute the forward problem for a group of sources.
"""
function ttforwsomesrc2D(vel::Array{Float64,N},coordsrc::Array{Float64,N},
                         coordrec::Vector{Array{Float64,N}},grd::AbstractGridEik,
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
                            refinearoundsrc=extrapars.refinearoundsrc,
                            allowfixsqarg=extrapars.allowfixsqarg)

    ## pre-allocate discrete adjoint variables
    ##  No adjoint calculations
    ## adjvars = nothing

    if returntt
        ttGRPSRC = Vector{Array{Float64}}(undef,nsrc) #zeros(n1,n2,nsrc)
    end

    ## group of pre-selected sources
    for s=1:nsrc    
        ## run the FMM forward
        ttFMM_hiord!(fmmvars,vel,view(coordsrc,s,:),grd,extrapars)

        ## interpolate at receivers positions
        for i=1:size(coordrec[s],1)
            #  view !!
            #ttpicksGRPSRC[s][i] = bilinear_interp( fmmvars.ttime,grd,view(coordrec[s],i,:) )
            ttpicksGRPSRC[s][i] = interpolate_receiver( fmmvars.ttime,grd,view(coordrec[s],i,:) )
        end

        if returntt
            ## Compute traveltime and interpolation at receivers in one go for parallelization
            #ttGRPSRC[:,:,s] .= fmmvars.ttime
            ttGRPSRC = fmmvars.ttime
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
                       refinearoundsrc::Bool,
                       allowfixsqarg::Bool)
    if typeof(grd)==Grid2DCart
        n1,n2 = grd.nx,grd.ny
    elseif typeof(grd)==Grid2DSphere
        n1,n2 = grd.r,grd.θ
    end       
    fmmvars = FMMVars2D(n1,n2,amIcoarsegrid=true,
                        refinearoundsrc=refinearoundsrc,
                        allowfixsqarg=allowfixsqarg)
    return fmmvars
end

function createFMMvars(grd::AbstractGridEik3D;
                       amIcoarsegrid::Bool,
                       refinearoundsrc::Bool,
                       allowfixsqarg::Bool)
    if typeof(grd)==Grid3DCart
        n1,n2,n3 = grd.nx,grd.ny,grd.nz
    elseif typeof(grd)==Grid3DSphere
        n1,n2,n3 = grd.r,grd.θ,grd.φ
    end   
    fmmvars = FMMVars3D(n1,n2,n3,amIcoarsegrid=true,
                        refinearoundsrc=refinearoundsrc,
                        allowfixsqarg=allowfixsqarg)
    return fmmvars
end


######################################################

function createAdjointVars(grd::AbstractGridEik2D;
                           amIcoarsegrid::Bool,
                           refinearoundsrc::Bool,
                           allowfixsqarg::Bool)
    if typeof(grd)==Grid2DCart
        n1,n2 = grd.nx,grd.ny
    elseif typeof(grd)==Grid2DSphere
        n1,n2 = grd.r,grd.θ
    end  
    adjvars = FMMVars2D(n1,n2,amIcoarsegrid=true,
                        refinearoundsrc=refinearoundsrc,
                        allowfixsqarg=allowfixsqarg)
    return adjvars
end

function createAdjointVars(grd::AbstractGridEik3D;
                           amIcoarsegrid::Bool,
                           refinearoundsrc::Bool,
                           allowfixsqarg::Bool)
    if typeof(grd)==Grid3DCart
        n1,n2,n3 = grd.nx,grd.ny,grd.nz
    elseif typeof(grd)==Grid3DSphere
        n1,n2,n3 = grd.r,grd.θ,grd.φ
    end 
    adjvars = FMMVars3D(n1,n2,n3,amIcoarsegrid=true,
                        refinearoundsrc=refinearoundsrc,
                        allowfixsqarg=allowfixsqarg)
    return adjvars
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
        nxyz = prod(size(vel))
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
        adjvars.fmmord.ttime .= 0.0
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

        #@show size(fmmvars.srcboxpar.ijsrc)

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
    if typeof(grd)==Grid2DCart || typeof(grd)==Grid3DCart 
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere || typeof(grd)==Grid3DSphere
        simtype = :spherical
    end
    
    ##==================================================
    ## create the refined grid
    grd_fine,srcrefvars = createfinegrid(grd,xyzsrc,vel,
                                         extrapars.grdrefpars)
    # @show grd_fine
    # @show srcrefvars.ijcoarse
    # @show srcrefvars.outxyminmax
    # @show size(srcrefvars.nearneigh_oper)
    # @show grd_fine.nx,grd_fine.ny

    # if simtype==:cartesian
    #     n1_fine,n2_fine = grd_fine.nx,grd_fine.ny
    # elseif simtype==:spherical
    #     n1_fine,n2_fine = grd_fine.nr,grd_fine.nθ
    # end
    
    ## pre-allocate ttime and status arrays plus the binary heap
    # fmmvars_fine = FMMVars2D(n1_fine,n2_fine,amIcoarsegrid=false,
    #                          refinearoundsrc=false,
    #                          allowfixsqarg=extrapars.allowfixsqarg)
    fmmvars_fine = createFMMvars(grd_fine,amIcoarsegrid=false,
                                 refinearoundsrc=false,
                                 allowfixsqarg=extrapars.allowfixsqarg)
    
    ##==================================================
    # init ttime 
    inittt = typemax(eltype(fmmvars_fine.ttime))
    fmmvars_fine.ttime .= inittt
    # Status of nodes
    fmmvars_fine.status .= 0   ## set all to far

    ##==================================================
    if dodiscradj
        ## pre-allocate adjoint variables for the fine grid
        adjvars_fine = createAdjointVars(grd_fine)
        ##
        ##  discrete adjoint: init stuff
        ## 
        for i=1:Ndim
            adjvars_fine.fmmord.Deriv[i].lastrowupdated[] = 0
            adjvars_fine.fmmord.Deriv[i].Nnnz[] = 0
        end
        adjvars_fine.fmmord.onsrccols[:] .= false
        adjvars_fine.fmmord.onhpoints[:] .= false
        # for safety, zeroes the traveltime
        adjvars_fine.fmmord.ttime .= 0.0
        # init the valued of last ttime computed
        adjvars_fine.fmmord.lastcomputedtt[] = 0
        # for safety, zeroes the codes for derivatives
        adjvars_fine.codeDeriv .= 0

    else
        adjvars_fine = nothing

    end
    ##==================================================

    ## source location, etc.      
    sourceboxloctt!(fmmvars_fine,srcrefvars.vel_fine,xyzsrc,grd_fine)


    # ## REGULAR grid
    # if simtype==:cartesian
    #     #ijsrc_fine = sourceboxloctt!(fmmvars_fine,vel_fine,xysrc,grd_fine )
    #     sourceboxloctt_cart!(fmmvars_fine,srcrefvars.vel_fine,xyzsrc,grd_fine )
               
    # elseif simtype==:spherical
    #     #ijsrc_fine = sourceboxloctt_sph!(fmmvars_fine,vel_fine,xysrc,grd_fine )
    #     sourceboxloctt_sph!(fmmvars_fine,srcrefvars.vel_fine,xyzsrc,grd_fine )
       
    # end

    ##==================================================================
    ## Run forward simulation (FMM) within the FINE grid
    #ijsrc_coarse = ttFMM_core!(fmmvars_fine,vel_fine,grd_fine,ijsrc_fine,adjvars_fine,
    ttFMM_core!(fmmvars_fine,srcrefvars.vel_fine,grd_fine,adjvars_fine,
                fmmvars_coarse=fmmvars, adjvars_coarse=adjvars,
                srcrefvars=srcrefvars)
         

    ##==================================================================

    # if dogradsrcpos
    #     ##======================================================
    #     ## Derivatives w.r.t. source position.  !!!FINE GRID!!!
    #     ##======================================================
      #     ∂u_h∂x_s = calc_∂u_h∂x_s_finegrid(fmmvars_fine,adjvars_fine,xysrc,vel_fine,grd_fine)
    #     println("hello from runrefinementaroundsrc(): ∂u_h∂x_s = $∂u_h∂x_s")
    # end
  
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

    # if typeof(grd)==Grid2D
    #     simtype=:cartesian
    # elseif typeof(grd)==Grid2DSphere
    #     simtype=:spherical
    # end
    
    # ## source location, etc.      
    # ## REGULAR grid
    # if simtype==:cartesian
    #     #ijsrc = sourceboxloctt!(fmmvars,vel,src,grd )
    #     sourceboxloctt_cart!(fmmvars,vel,src,grd )
    # elseif simtype==:spherical
    #     #ijsrc = sourceboxloctt_sph!(fmmvars,vel,src,grd )
    #     sourceboxloctt_sph!(fmmvars,vel,src,grd )
    # end

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
        ijkcoarse = srcrefvars.ijkcoarse
        
        # outxmin = srcrefvars.outxyminmax[1]
        # outxmax = srcrefvars.outxyminmax[2]
        # outymin = srcrefvars.outxyminmax[3]
        # outymax = srcrefvars.outxyminmax[4]
    end

    if adjvars==nothing
        dodiscradj=false
    else
        dodiscradj=true
    end    

    # ## Sizes
    # if typeof(grd)==Grid2DCart || typeof(grd)==Grid3DCart 
    #     simtype = :cartesian
    # elseif typeof(grd)==Grid2DSphere || typeof(grd)==Grid3DSphere
    #     simtype = :spherical
    # end
    # if simtype==:cartesian
    #     n1,n2 = grd.nx,grd.ny 
    # elseif simtype==:spherical
    #     n1,n2 = grd.nr,grd.nθ
    # end
    # n12 = n1*n2

    ##----------------------------------------------------------------------
    ## fmmvars.srcboxpar.ijksrc can be a field of either
    ##  - a SrcRefinVars2D if we are in the coarse grid with no refinement,
    ##     or in the fine grid
    ##  - a SourcePtsFromFineGrid if we are in the coarse grid after the
    ##     FMM has been already run in the fine grid (multiple source points)
    ##
    ijksrc = fmmvars.srcboxpar.ijksrc

    ## number of accepted points       
    naccinit = size(ijksrc,1)  

    ## Establish a mapping between linear and Cartesian indexing 
    grsize = size(vel)
    vlen = prod(grsize)
    ND = ndims(vel)
    @assert ND==N

    
    ##======================================================
    # MVector size needs to be know at compile time, so...
    if N==2
        # a 2D point
        curptijk = MVector(0,0)
        accptijk = MVector(0,0)
        # derivative codes
        idD = MVector(0,0)
        # neighboring points
        neigh = SA[ 1  0;
                    0  1;
                   -1  0;
                    0 -1]
        
    elseif N==3
        # a 3D point
        curptijk = MVector(0,0,0)
        accptijk = MVector(0,0,0)
        # derivative codes
        idD = MVector(0,0,0)
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
        # init row stuff
        colindsonsrc = MVector(0)

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
    ## Only while running in the fine grid
    ##======================================================
    if isthisrefinementsrc
        # init source indices array and some counters
        ijksrc_coarse = Matrix{Int64}(undef,vlen,ND)
        counter_ijksrccoarse::Int64 = 1
        counter_adjcoarse::Int64 = 1

        for l=1:naccinit
            curptijk .= ijksrc[l,:]

            oncoa,ijk_coarse = isacoarsegridnode(curptijk,downscalefactor,ijkcoarse)

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
    ## construct initial narrow band
    for l=1:naccinit ##
        
        for ne=1:size(neigh,1) ## four potential neighbors
            # i = ijksrc[l,1] + neigh[ne,1]
            # j = ijksrc[l,2] + neigh[ne,2]
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
                # get handle
                #han = cart2lin(curptijk,grsize)
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
    totnpts = vlen
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if fmmvars.bheap.Nh[]<1
            break
        end
        # pop value from min-heap
        acchan,tmptt = pop_minheap!(fmmvars.bheap)
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
            oncoa,ijk_coarse = isacoarsegridnode(accptijk,downscalefactor,ijkcoarse)

            if oncoa
                # get the linear index/handle 
                pthan_coarse = cart2lin(ijk_coarse,size(fmmvars_coarse.status))
                # update stuff
                fmmvars_coarse.ttime[pthan_coarse]  = fmmvars.ttime[acchan]
                fmmvars_coarse.status[pthan_coarse] = fmmvars.status[acchan]
                ijksrc_coarse[counter_ijksrccoarse,:] .= ijk_coarse
                ##fmmvars_coarse.srcboxpar.ijksrc = vcat(fmmvars_coarse.srcboxpar.ijksrc,[ia_coarse ja_coarse])
                #@show counter_ijksrccoarse,fmmvars_coarse.srcboxpar.ijksrc
                ## update counter
                counter_ijksrccoarse += 1
                
                if dodiscradj
                    ## next line on FINE grid
                    adjvars.fmmord.onhpoints[node] = true ## adjvars of FINE grid
                    # Discrete adjoint on COARSE grid
                    # go from cartesian (i,j) to linear
                    adjvars_coarse.idxconv.lfmm2grid[counter_adjcoarse] = pthan_coarse
                                      # cart2lin(ijk_coarse,adjvars_coarse.idxconv.grsize)
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
            #  if (ia==n1) || (ia==1) || (ja==n2) || (ja==1)
            #if (ia==1 && !outxmin) || (ia==n1 && !outxmax) || (ja==1 && !outymin) || (ja==n2 && !outymax)
            if any(accptijk.==1) || any(accptijk.==grsize)
            
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

                #println("\n === Hello, quitting the refinement at node $node === ")
                return 
            end
            
        end # if isthisrefinementsrc      
        ##===================================
        

        ##===========================================
        ## try all neighbors of newly accepted point
        for ne=1:size(neigh,1) 
            # i = ia + neigh[ne,1]
            # j = ja + neigh[ne,2]
            curptijk .= accptijk .+ neigh[ne,:]
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
                # get handle
                #han = cart2lin(curptijk,grsize)
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

                # get handle
                #han = cart2lin(curptijk,n1)
                # update the traveltime for this point in the heap
                update_node_minheap!(fmmvars.bheap,tmptt,curpthan)

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    adjvars.codeDeriv[curpthan,:] .= idD
                end

            end
        end # for ne=1:4

    end # for node=naccinit+1:totnpts 

    ##======================================================
    if dodiscradj
        createsparsederivativematrices!(grd,adjvars,fmmvars.status)
    end # if dodiscradj
    ##======================================================
    
    return 
end







##################################################################
