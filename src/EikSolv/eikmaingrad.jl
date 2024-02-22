
######################################################################

"""
$(TYPEDSIGNATURES)

Calculate the gradient using the adjoint state method for 2D velocity models. 
Returns the gradient of the misfit function with respect to velocity calculated at the given point (velocity model). 
The gradient is calculated using the adjoint state method.
The computations are run in parallel depending on the value of `extraparams.parallelkind`.

# Arguments
- `vel`: the 2D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays 
- `pickobs`: observed traveltime picks
- `stdobs`: standard deviation of error on observed traveltime picks, an array with same shape than `pickobs`
    * `parallelkind`: serial, Threads or Distributed run? (:serial, :sharedmem, :distribmem)
    * `refinearoundsrc`: whether to perform a refinement of the grid around the source location
    * `radiussmoothgradsrc`: radius for smoothing the gradient around the source. Zero means no smoothing.
    * `allowfixsqarg`: brute-force fix negative saqarg. Don't use this.
    * `smoothgradkern`: smooth the gradient with a kernel of size (in pixels). Zero means no smoothing.
    * `manualGCtrigger`: trigger garbage collector (GC) manually at selected points.

# Returns
- `grad`: the gradient as a 2D array

"""
function eikgradient(vel::Array{Float64,N}, grd::AbstractGridEik,coordsrc::Array{Float64,2},
                     coordrec::Vector{Array{Float64,2}},pickobs::Vector{Vector{Float64}},
                     stdobs::Vector{Vector{Float64}},whichgrad::Symbol=:gradvel ;
                     extraparams::Union{ExtraParams,Nothing}=nothing) where N

    @assert (whichgrad in (:gradvel, :gradsrcloc, :gradvelandsrcloc)) "whichgrad not in (:vel, :srcloc, :velandsrcloc)"
    
    if extraparams==nothing
        extraparams =  ExtraParams()
    end

    # some checks
    @assert all(vel.>0.0)
    checksrcrecposition(grd,coordsrc,coordrec)
    
    nsrc = size(coordsrc,1)

    if extraparams.parallelkind==:distribmem
        ##====================================
        ## Distributed memory
        ##====================
        nw = nworkers()
        ## calculate how to subdivide the srcs among the workers
        grpsrc = distribsrcs(nsrc,nw)
        nchu = size(grpsrc,1)
        ## array of workers' ids
        wks = workers()
        ## do the calculations
        ∂χ∂vel_all = Vector{Array{Float64,ndims(vel)}}(undef,nchu)
        ∂χ∂xysrc = Vector{Vector{Float64}}(undef,nsrc)
        @sync begin 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async begin
                    ∂χ∂vel_all[s],∂χ∂xysrc[igrs] .= remotecall_fetch(calcgradsomesrc2D,wks[s],vel,
                                                                       coordsrc[igrs,:],coordrec[igrs],
                                                                       grd,stdobs[igrs],pickobs[igrs],
                                                                       whichgrad,extraparams )
                end
            end
            ∂χ∂vel = sum(∂χ∂vel_all)
        end


    elseif extraparams.parallelkind==:sharedmem
        ##====================================
        ## Shared memory        
        ##====================
        nth = Threads.nthreads()
        grpsrc = distribsrcs(nsrc,nth)
        nchu = size(grpsrc,1)            
        ##  do the calculations
        ∂χ∂vel_all = Vector{Array{Float64,ndims(vel)}}(undef,nchu)
        ∂χ∂xysrc = Vector{Vector{Float64}}(undef,nsrc)
        
        Threads.@threads for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @show igrs
            ∂χ∂vel_all[s],∂χ∂xysrc[igrs] = calcgradsomesrc2D(vel,view(coordsrc,igrs,:),view(coordrec,igrs),
                                                               grd,view(stdobs,igrs),view(pickobs,igrs),
                                                               whichgrad,extraparams )
        end
        ∂χ∂vel = sum(∂χ∂vel_all)


    elseif extraparams.parallelkind==:serial
        ##====================================
        ## Serial run
        ##====================
        ∂χ∂vel,∂χ∂xysrc = calcgradsomesrc2D(vel,coordsrc,coordrec,grd,stdobs,pickobs,
                                            whichgrad,extraparams )

    end

    ## smooth gradient
    if extraparams.smoothgradkern>0
        ∂χ∂vel = smoothgradient(extraparams.smoothgradkern,∂χ∂vel)
    end

    if extraparams.manualGCtrigger
        # trigger garbage collector
        #println("Triggering GC")
        GC.gc()
    end
    return ∂χ∂vel,∂χ∂xysrc 
end

######################################################################

"""
$(TYPEDSIGNATURES)


Calculate the gradient for some requested sources 
"""
function calcgradsomesrc2D(vel::Array{Float64,N},xyzsrc::AbstractArray{Float64,2},
                           coordrec::AbstractVector{Array{Float64,2}},grd::AbstractGridEik,
                           stdobs::AbstractVector{Vector{Float64}},pickobs1::AbstractVector{Vector{Float64}},
                           whichgrad::Symbol,extrapars::ExtraParams ) where N
                           
    grsize = size(vel)
    nsrc = size(xyzsrc,1)

    if whichgrad==:gradvel
        gradvel1 = zeros(grsize)
        gradvelall = zeros(grsize)
        gradsrcpos = nothing

    elseif whichgrad==:gradsrcloc
        gradvel1 = nothing
        gradvelall = nothing
        gradsrcpos = zeros(eltype(vel),nsrc,ndims(vel)) # ,2)-> 2D
        
    elseif whichgrad==:gradvelandsrcloc
        gradvel1 = zeros(grsize)
        gradvelall = zeros(grsize)
        gradsrcpos = zeros(eltype(vel),nsrc,ndims(vel)) # ,2)-> 2D

    end

    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars = createFMMvars(grd,amIcoarsegrid=true,refinearoundsrc=extrapars.refinearoundsrc,
                            allowfixsqarg=extrapars.allowfixsqarg)
    
    ## pre-allocate discrete adjoint variables for coarse grid
    adjvars = createAdjointVars(grsize)

    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        ###########################################
        ## Get the traveltime, forward and adjoint parameters, etc.
        if extrapars.refinearoundsrc
            ##
            ## Refinement around the source
            ## 
            # fmmvars_fine,adjvars_fine need to be *re-allocated* for each source
            #  because the size of the grid may change when hitting borders, etc.
            fmmvars_fine,adjvars_fine,grd_fine,srcrefvars = ttFMM_hiord!(fmmvars,vel,view(xyzsrc,s,:),
                                                                         grd,adjvars,extrapars)
        else
            ##
            ## NO refinement around the source
            ## 
            ttFMM_hiord!(fmmvars,vel,view(xyzsrc,s,:),grd,adjvars,extrapars)
            fmmvars_fine=nothing
            adjvars_fine=nothing
            grd_fine=nothing
            srcrefvars=nothing
        end

        ###############################################################
        # Solve the adjoint problem(s) and get the requested gradient(s)
        if gradsrcpos==nothing
            tmpgradsrcpos = gradsrcpos
        else
            # use a view so gradsrcpos gets automatically filled in this loop
            tmpgradsrcpos = view(gradsrcpos,s,:)
        end
                
        calcgrads_singlesrc!(gradvel1,tmpgradsrcpos,
                            fmmvars,adjvars,
                            xyzsrc[s,:],coordrec[s],pickobs1[s],stdobs[s],
                            vel,grd,whichgrad,extrapars.refinearoundsrc,
                            fmmvars_fine=fmmvars_fine,
                            adjvars_fine=adjvars_fine,
                            grd_fine=grd_fine,
                            srcrefvars=srcrefvars)

         
        if whichgrad==:gradvel || whichgrad==:gradvelandsrcloc
            ###########################################
            ## smooth gradient (with respect to velocity) around the source
            smoothgradaroundsrc2D!(gradvel1,view(xyzsrc,s,:),grd,
                                   radiuspx=extrapars.radiussmoothgradsrc)
            ##########################################
            ## add up gradients from different sources
            gradvelall .+= gradvel1
        end
        
    end # for s=1:nsrc
 
    if extrapars.manualGCtrigger
        # trigger garbage collector
        #println("Triggering GC")
        GC.gc()
    end
    
    return gradvelall,gradsrcpos
end 

##############################################################################

function createAdjointVars(grsize::Tuple)
    ndim = length(grsize)
    if ndim==2
        adjointvars = AdjointVars2D(grsize[1],grsize[2])
    elseif ndim==3
        adjointvars = AdjointVars3D(grsize[1],grsize[2],grsize[3])
    end
    return adjointvars
end

##############################################################################

function calcgrads_singlesrc!(gradvel1::Union{AbstractArray{Float64},Nothing},
                              gradsrcpos1::Union{AbstractArray{Float64},Nothing},
                              fmmvars::AbstractFMMVars,
                              adjvars::AbstractAdjointVars,
                              xyzsrc::AbstractArray{Float64},
                              xyzrecs::AbstractArray{Float64},
                              pickobs::AbstractVector{Float64},
                              stdobs::AbstractVector{Float64},
                              velcart::AbstractArray{Float64},
                              grd::AbstractGridEik,
                              whichgrad::Symbol,
                              refinearoundsrc::Bool;
                              fmmvars_fine::Union{AbstractFMMVars,Nothing}=nothing,
                              adjvars_fine::Union{AbstractAdjointVars,Nothing}=nothing,
                              grd_fine::Union{AbstractGridEik,Nothing}=nothing,
                              srcrefvars::Union{AbstractArray{Float64},AbstractSrcRefinVars,Nothing}=nothing
                              )
    
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    ##======================================================
    # Projection operator P ordered according to FMM order
    #  The order for P is:
    #    rows: according to coordrec, stdobs, pickobs
    #    columns: according to the FMM order
    P = calcprojttfmmord(fmmvars.ttime,grd,adjvars.idxconv,xyzrecs)
                                      
    ##==============================
    tt_fmmord = adjvars.fmmord.ttime
    Ndim = ndims(velcart)
    grsize = size(velcart)
    
    #######################################################
    ##  right hand side adj eq. == -(∂ψ/∂u_p)^T
    #######################################################
    #rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)
    fact2∂ψ∂u = ((P*tt_fmmord).-pickobs)./stdobs.^2
    # bool vector with false for src columns
    rq = .!adjvars.fmmord.onsrccols
    P_Breg = P[:,rq] # remove columns
    P_Areg = P[:,adjvars.fmmord.onsrccols] # only cols on source pts
    
    #################################################################
    ##  2 * diag(D_ij*tt_j)*D_ip ( for ∂f_i_∂u_p and ∂f_i_∂u_s )
    #################################################################
    # Derivatives (along x,y,z) matrices, row-deficient
    Deriv = adjvars.fmmord.Deriv
    ## compute the lhs terms
    twoDttDR = Vector{VecSPDerivMat}(undef,Ndim)
    twoDttDS = Vector{SparseMatrixCSC}(undef,Ndim)
    for d=1:Ndim
        twoDttDR[d],twoDttDS[d] = calc2diagDttD(Deriv[d],tt_fmmord,adjvars.fmmord.onsrccols)
    end

    #####################################
    ##   Solve the adjoint equation
    #####################################
    rhs = - transpose(P_Breg) * fact2∂ψ∂u
    lambda_fmmord_coarse = solveadjointeq(twoDttDR,rhs)

    ###############################################
    # deriv. of the implicit forw. mod. w.r.t. u_s
    #   ordered according to FMM order
    ∂fi_∂us = sum( twoDttDS )
    ## 
    ∂ψ_∂u_s = transpose(P_Areg) * fact2∂ψ∂u
    ∂ψ_∂u_p = transpose(∂fi_∂us) * lambda_fmmord_coarse


        
    ####################################################
    ##   Pre-compute stuff for grid refinement
    ####################################################
    if refinearoundsrc
        ############################################################
        #  Get adjoint variable, etc. from fine grid
        ############################################################
        lambda_fmmord_fine,∂u_h_dtau_s,H_Areg  = computestuff_finegrid!(fmmvars_fine,
                                                                        adjvars_fine,
                                                                        grd_fine,
                                                                        srcrefvars)
    end


    ##----------------------------------------------------------
    if whichgrad==:gradsrcloc || whichgrad==:gradvelandsrcloc
        
        ##############################################
        ##  Derivative with respect to the traveltime
        ##     at the "onsrc" points
        ##  compute gradient
        ##############################################
        if refinearoundsrc
            gradsrcpos1 .= ∂misfit∂srcpos(∂fi_∂us,
                                          lambda_fmmord_coarse,
                                          velcart,
                                          ∂ψ_∂u_s,
                                          ∂ψ_∂u_p,
                                          fmmvars.srcboxpar,
                                          grd,
                                          xyzsrc,
                                          lambda_fmmord_fine,
                                          grd_fine,
                                          ∂u_h_dtau_s,
                                          H_Areg,
                                          srcrefvars,
                                          fmmvars_fine.srcboxpar,
                                          refinearoundsrc=refinearoundsrc )
        else
            gradsrcpos1 .= ∂misfit∂srcpos(∂fi_∂us,
                                          lambda_fmmord_coarse,
                                          velcart,
                                          ∂ψ_∂u_s,
                                          ∂ψ_∂u_p,
                                          fmmvars.srcboxpar,
                                          grd,
                                          xyzsrc,
                                          refinearoundsrc=refinearoundsrc )
        end
    end

    
    ##----------------------------------------------------------
    if whichgrad==:gradvel || whichgrad==:gradvelandsrcloc
      
        ##===============================================
        # Gradient with respect to velocity, term T2Vb
        ##===============================================
        # Reorder lambda from fmmord to original grid!!
        idxconv = adjvars.idxconv
        Nacc = length(lambda_fmmord_coarse)
        nptsonsrc = count(adjvars.fmmord.onsrccols)

        if Ndim==2
            # a 2D point
            ijkpt = MVector(0,0)
        elseif Ndim==3
            ijkpt = MVector(0,0,0)
        end

        #Nall = length(idxconv.lfmm2grid)
        for p=1:Nacc
            # The following two lines (p+nptsonsrc) is a way to map the
            #  "full" indices to the "reduced" set of indices (no source region)
            iorig = idxconv.lfmm2grid[p+nptsonsrc] # skip pts on src
            # get the contribution of T2Vb to the gradient
            gradvel1[iorig] = 2.0 * lambda_fmmord_coarse[p] / velcart[iorig]^3
        end

        ##===============================================
        #   Derivative  du_s/dv_a
        ##===============================================
        if refinearoundsrc==false
            #-----------------------------------
            #  NO refinement around the source
            #-----------------------------------
            ## source box parameters
            velcorn  = fmmvars.srcboxpar.velcorn
            distcorn = fmmvars.srcboxpar.distcorn

            ## derivative of traveltime at source nodes w.r.t. velocity in region A
            ##   dus_dva must be in FMM order to agree with ∂fi_∂us
            dus_dva = - diagm(distcorn) .* (1.0 ./ velcorn.^2)  # Hadamard product!

        elseif refinearoundsrc
            #-----------------------------------
            #  REFINEMENT around the source
            #-----------------------------------
            ## Solve the adjoint equation in the fine grid
            ##   dus_dva must be in FMM order to agree with ∂fi_∂us
            dus_dva = calc_duh_dva_finegrid!(lambda_fmmord_fine,
                                             H_Areg,
                                             ∂u_h_dtau_s,
                                             fmmvars_fine,
                                             adjvars_fine,
                                             srcrefvars)
        end

        ##===============================================
        # Gradient with respect to velocity, term T2Va
        ##===============================================
        ## FMM ordering 
        T2Va = transpose( dus_dva ) * ∂ψ_∂u_p 
        #T2Va = transpose(lambda_fmmord_coarse) * tmpdfidva
        ## (i,j) indices of the source points
        ijksrcreg = fmmvars.srcboxpar.ijksrc

        ## add contribution to the gradient
        for i=1:length(T2Va)
            ijkpt .= ijksrcreg[i,:]
            igr = cart2lin(ijkpt,grsize)
            gradvel1[igr] += T2Va[i]
        end

        ##===============================================
        # Gradient with respect to velocity, term T1Va 
        ##===============================================
        ## FMM ordering
        tmpT1Va = transpose( dus_dva) * ∂ψ_∂u_s 
        for i=1:length(tmpT1Va)
            ijkpt .= ijksrcreg[i,:]
            igr = cart2lin(ijkpt,grsize)
            gradvel1[igr] += tmpT1Va[i]
        end

    end
    
    return 
end

##############################################################################

function solveadjointeq(twoDttDR::Vector{VecSPDerivMat},rhs::AbstractVecOrMat{Float64})

    #===========================================
    ###  !!! REMARK from SparseArrays:  !!! ####
    ============================================
    The row indices in every column NEED to be SORTED. If your 
    SparseMatrixCSC object contains unsorted row indices, one quick way 
    to sort them is by doing a double transpose.

    # struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    #     m::Int                  # Number of rows
    #     n::Int                  # Number of columns
    #     colptr::Vector{Ti}      # Column j is in colptr[j]:(colptr[j+1]-1)
    #     rowval::Vector{Ti}      # Row indices of stored values
    #     nzval::Vector{Tv}       # Stored values, typically nonzeros
    # end

    ============================================#
    
    Ni = twoDttDR[1].Nsize[1]
    Nj = twoDttDR[1].Nsize[2]
    tmplhs = spzeros(Ni,Nj)
    ## We have a CSR matrix as vectors, we need a *transpose* of it,
    ##  so we construct directly a CSC by *exchanging* i and j indices
    Ndim = length(twoDttDR)
    for d=1:Ndim
        Nnnz = twoDttDR[d].Nnnz[]
        # Ni = twoDttDR[d].Nsize[1]
        # Nj = twoDttDR[d].Nsize[2]
        # Transpose the matrices and add them
        tmplhs .+= SparseMatrixCSC(Nj,Ni,twoDttDR[d].iptr[1:Ni+1],
                                   twoDttDR[d].j[1:Nnnz],twoDttDR[d].v[1:Nnnz])
    end
        
    ## make sure it's recognised as upper triangular...
    lhs = UpperTriangular(tmplhs)

    ## OLD stuff...
    # ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
    # ##   the solver (\) does not use the correct sparse algo for matrix division
    # tmplhs = copy(transpose( (2.0.*Diagonal(Dx*tt)*Dx) .+ (2.0.*Diagonal(Dy*tt)*Dy) ))
    # lhs = UpperTriangular(tmplhs)

    ############################################################
    ##  solve the linear system to get the adjoint variable
    ############################################################
    lambda_fmmord_coarse = lhs\rhs

    return lambda_fmmord_coarse
end


##############################################################################

function computestuff_finegrid!(fmmvars,adjvars,grd,srcrefvars)

    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    
    ## find the last value of traveltime computed in the fine grid
    ##    (The fine grid stop when hitting a boundary...)
    ## number of model parameters (== computed traveltimes)
    Naccpts = adjvars.fmmord.lastcomputedtt[]
    @assert findfirst( adjvars.fmmord.ttime.==0.0)-1 == Naccpts
    ## extract computed traveltime values
    tt_fmmord = adjvars.fmmord.ttime[1:Naccpts]
    #@show size(tt_fmmord)
    onsrccols = adjvars.fmmord.onsrccols[1:Naccpts]
    nptsonsrc = count(onsrccols)
    @assert length(onsrccols)==length(tt_fmmord)
    ## bool array of grid points (in FMM order) on h (u_s) points
    onHpts = adjvars.fmmord.onhpoints[1:Naccpts]  #[node]

    ## indices of coinciding with u_s (h)
    idxonHpts = findall(onHpts) # get all "true" values
    Nhpts = length(idxonHpts)

    # To reorder lambda from fmmord to original grid...
    idxconv = adjvars.idxconv
    grsize = idxconv.grsize
    Ndim = length(grsize)

    ###########################################
    H_i = zeros(Int64,Nhpts)
    H_j = zeros(Int64,Nhpts)
    H_v = zeros(Float64,Nhpts)

    ############################################################
    ## Define the projection operator H in the fine grid
    ## 
    ## Since the "receivers" are all exactly on grid points (h),
    ##  conciding with the coarse grid, we don't need to
    ##  perform any interpolation.
    ############################################################
    q::Int64=1    
    for l=1:Nhpts
        # The "onhpoints" are already in fmm order
        jfmmord = idxonHpts[l]
        # the order for P is:
        #   rows: according to coordrec, stdobs, pickobs
        #   columns: according to the FMM order
        ## P[r,jfmmord] =
        H_i[q] = l
        H_j[q] = jfmmord
        H_v[q] = 1.0
        q+=1
    end
    H = sparse(H_i,H_j,H_v,Nhpts,Naccpts)

    
    ###########################################
    #  Derivatives (along x,y,z) matrices, row-deficient
    Deriv = adjvars.fmmord.Deriv

    ################################
    ##   left hand side adj eq.
    ################################
    ## compute the lhs terms
    twoDttDR = Vector{VecSPDerivMat}(undef,Ndim)
    twoDttDS = Vector{SparseMatrixCSC}(undef,Ndim)
    for d=1:Ndim
        twoDttDR[d],twoDttDS[d] = calc2diagDttD(Deriv[d],tt_fmmord,onsrccols)
    end

    #######################################################
    ##  right hand side adj eq. == -(adjoint source)^T
    #######################################################
    # @show size(H)
    # @show size(.!adjvars.fmmord.onsrccols[1:Naccpts])
    # bool vector with false for src columns
    #    only part of the rectangular fine grid has been used, so [1:Naccpts]
    rq = .!onsrccols
    H_Breg = H[:,rq] # remove columns
    H_Areg = H[:,onsrccols]
    rhs = - transpose(H_Breg)

  
    #####################################
    ##   Solve the adjoint equation
    #####################################
    lambda_fmmord_fine = solveadjointeq(twoDttDR,rhs)

    # deriv. of the implicit forw. mod. w.r.t. u_s, FMM ordering
    ∂gi_∂taus = sum( twoDttDS )
    # copy to materialize the lazy transpose
    ∂u_h_dtau_s =  transpose(lambda_fmmord_fine ) * ∂gi_∂taus
   
    return lambda_fmmord_fine,∂u_h_dtau_s,H_Areg 
end

####################################################################

function calc_duh_dva_finegrid!(lambda_fmmord_fine,
                                H_Areg,
                                ∂u_h_dtau_s,
                                fmmvars,
                                adjvars,
                                srcrefvars)

    Naccpts = adjvars.fmmord.lastcomputedtt[]
    onsrccols = adjvars.fmmord.onsrccols[1:Naccpts]
    nptsonsrc = count(onsrccols)
    Nptsoffsrc = Naccpts-nptsonsrc
    onHpts = adjvars.fmmord.onhpoints[1:Naccpts]  #[node]
    ## indices of coinciding with u_s (h)
    idxonHpts = findall(onHpts) # get all "true" values
    Nhpts = length(idxonHpts)
    idxconv = adjvars.idxconv

    @assert size(lambda_fmmord_fine,1)==Nptsoffsrc
    @assert size(lambda_fmmord_fine,2)==Nhpts


    ##========================================
    ##  Compute the T2Vd term
    ##========================================
    ## The gradient in the fine grid is not a rectangular grid but it
    ##    depends on when the FMM has stopped (so a vector with 'Naccpts'
    ##    entries, not a 2D array)
    duh_dwq = zeros(Nhpts,Naccpts)
    
    for p=1:Nptsoffsrc
        # The following two lines (p+nptsonsrc) is a way to map the
        #  "full" indices to the "reduced" set of indices (no source region)
        iorig = idxconv.lfmm2grid[p+nptsonsrc] # skip pts on src
        # # get the i,j indices for vel and grad
        # lin2cart!(iorig,grsize,ijkpt)
        # i,j = ijkpt
        p2 = p+nptsonsrc
        for h=1:Nhpts
            # get the contribution of T2Vd to the gradient
            duh_dwq[h,p2] = 2.0 * lambda_fmmord_fine[p,h] / srcrefvars.velcart_fine[iorig]^3
        end
    end

    ## source box parameters
    velcorn  = fmmvars.srcboxpar.velcorn
    distcorn = fmmvars.srcboxpar.distcorn
    ijsrcreg = fmmvars.srcboxpar.ijksrc

    ## derivative of \tau_s w.r.t. w_a
    dtaus_dwa = - diagm(distcorn) .* (1.0./velcorn.^2) # Hadamard product!

    ##========================================
    ## Compute the T2Vc term 
    ##======================================== 
    ## FMM ordering for ∂gi_∂taus!!!
    #tmpdgidwa = ∂gi_∂taus * dtaus_dwa  
    ## FMM ordering for lambda_fmmord!!!
    #T2Vc = transpose(lambda_fmmord_fine) * tmpdgidwa
    T2Vc = ∂u_h_dtau_s * dtaus_dwa

    ## add contribution to the gradient in the fine grid
    for p=1:nptsonsrc #size(T2Vc,2)
        for h=1:Nhpts #size(T2Vc,1)
            duh_dwq[h,p] += T2Vc[h,p]
        end
    end

    ##========================================
    ## Compute the T1Vc term 
    ##========================================
    T1Vc = H_Areg * dtaus_dwa
    for p=1:nptsonsrc
        for h=1:Nhpts
            duh_dwq[h,p] += T1Vc[h,p]
        end
    end
    
    ##========================================
    ## Compute dwq/dwa
    ##========================================
    # filter grid point which have been actually used 
    dwq_dva = zeros(Naccpts,Nhpts)
    nneigh = srcrefvars.nearneigh_oper
    ##
    for (j,q) in enumerate(idxonHpts)

        tmpqorig = idxconv.lfmm2grid[q]
        qorig = srcrefvars.nearneigh_idxcoarse[tmpqorig]        

        for p=1:Naccpts
            #i = p
            #  use the "full" indices (including points on source)
            porig = idxconv.lfmm2grid[p]

            #@show (p,j),q,(porig,qorig)
            ## extract the proper elements from the interpolation
            ##   matrix in FMM order
            dwq_dva[p,j] = nneigh[porig,qorig]
        end

    end

    ##========================================
    ## duh_dva
    ##========================================
    duh_dva = duh_dwq * dwq_dva

    return duh_dva
end

#######################################################################################

"""
$(TYPEDSIGNATURES)

Calculates the following maintaining the sparse structure:
     2 * diag(D * tt) * Dr
"""
function calc2diagDttD(vecD::VecSPDerivMat,tt::Vector{Float64},
                         onsrccols::Vector{Bool})
    # Remark: equivalent to CSR format 

    ##
    ## Calculates the following in place, maintaining the sparse structure:
    ##   2*diag(Dx*tt)*Dxr 
    ##

    # ## CSR matrix-vector product
    # tmp1 = zeros(nrows)
    # for i=1:nrows
    #     # pi=pointers to column indices
    #     for l=vecD.pi[i]:vecD.pi[i+1]-1
    #         j = vecD.j[l]
    #         # dot product
    #         tmp1[i] += vecD.v[l]*tt[j]
    #     end
    # end

    # size of D^x or D^y
    nrows,ncols = vecD.Nsize[1],vecD.Nsize[2]
    # number of rows of D^xr, D^yr, both row and col deficient
    nsrcpts = count(onsrccols)
    ncolsR = ncols - nsrcpts

    #############################################
    ## for grad w.r.t. velocity
    # Derivative (along x) matrix, row- *and* column-deficient
    ## ncolidx = vecD.Nnnz[] - nsrcpts #  <<--- OLD
    #ncolidx = ncols - nsrcpts  #  <<--- NEW
    Nnnz = vecD.Nnnz[]

    two_Dtt_DR = VecSPDerivMat( iptr=zeros(Int64,nrows+1), j=zeros(Int64,Nnnz),
                                v=zeros(Nnnz), Nsize=[nrows,ncolsR] )
    #############################################
    
    # mapping of column number from full to column reduced
    idxrowred = zeros(Int64,length(onsrccols))
    q = 0
    for p=1:length(onsrccols)
        if onsrccols[p]
            idxrowred[p] = q
        else
            q+=1
            idxrowred[p] = q
        end
    end
    
    #############################################
    ## for grad w.r.t. source loc calculations
    #twoDttDS = zeros(eltype(vecD.v),nrows,nsrcpts)
    DS_i = Vector{Int64}(undef,Nnnz)
    DS_j = Vector{Int64}(undef,Nnnz)
    DS_val = Vector{Float64}(undef,Nnnz)
    #############################################

    l3=0
    ## CSR matrix-vector product
    for i=1:nrows  # rows

        ## pre-compute Dx * tt 
        tmp1 = 0.0
        for l=vecD.iptr[i]:vecD.iptr[i+1]-1
            j = vecD.j[l]
            # dot product
            tmp1 += vecD.v[l]*tt[j]
        end

        ## init the next row pointer
        two_Dtt_DR.iptr[i+1] = two_Dtt_DR.iptr[i]
       
        ## l2 runs on the column-reduced matrix DR
        l2 = two_Dtt_DR.iptr[i]-1 

        for l=vecD.iptr[i]:vecD.iptr[i+1]-1 # columns
            j = vecD.j[l]    # column index

            ## scale all rows by 2*tmp1 excluding certain columns
            if onsrccols[j]==false
                ############################################################### 
                ##  Computation of 2*D*tt*DR for the adjoint variable lambda
                ###############################################################
                # perform the calculation only if we are not on a source point
                #   in order to remove the columns corresponding to the source
                #   points

                # populate the row-reduced matrix two_Dtt_DR
                two_Dtt_DR.iptr[i+1] += 1
                l2 += 1
                two_Dtt_DR.j[l2] = idxrowred[j]
                ## 2*diag(D*tt)*DR
                two_Dtt_DR.v[l2] = 2.0 * tmp1 * vecD.v[l]
                # update pointers
                two_Dtt_DR.Nnnz[] += 1
                two_Dtt_DR.lastrowupdated[] = i # this must stay here (this if and for loop)!

            elseif onsrccols[j]
                ################################################################## 
                ##  Computation of 2*D*tt*DS for the grad w.r.t. source position
                ##################################################################
                js = count(onsrccols[1:j])
                ## old way, dense matrix
                #twoDttDS[i,js] = 2.0 * tmp1 * vecD.v[l]
                ## create a sparse matrix
                l3 += 1
                DS_i[l3] = i
                DS_j[l3] = js
                DS_val[l3] = 2.0 * tmp1 * vecD.v[l]
            end
        end
    end

    ## create the sparse matrix
    twoDttDS = sparse(DS_i[1:l3],DS_j[1:l3],DS_val[1:l3],nrows,nsrcpts)

    return two_Dtt_DR,twoDttDS
end


##################################################

"""
$(TYPEDSIGNATURES)

 Projection operator P containing interpolation coefficients, ordered according to FMM order.
"""
function calcprojttfmmord(ttime::AbstractArray{Float64,N},grd::AbstractGridEik,idxconv::MapOrderGridFMM,
                          coordrec::AbstractArray{Float64}) where N

    if typeof(grd)<:AbstractGridEik2D
        simdim = :sim2D
        Ncoe = 4 
    elseif typeof(grd)<:AbstractGridEik3D
        simdim = :sim3D
        Ncoe = 8
    else
        error("calcprojttfmmord(): Wrong grd argument.")
    end

    # calculate the coefficients and their indices using bilinear interpolation
    nmodpar = length(ttime)
    nrec = size(coordrec,1)
    P_i = zeros(Int64,nrec*Ncoe)
    P_j = zeros(Int64,nrec*Ncoe)
    P_v = zeros(nrec*Ncoe)

    q::Int64=1
    for r=1:nrec

        if simdim==:sim2D 
            coeff,_,ijcoe = bilinear_interp( ttime,grd,view(coordrec,r,:),
                                             outputcoeff=true )
        elseif simdim==:sim3D 
            coeff,_,ijkcoe = trilinear_interp( ttime,grd,view(coordrec,r,:),
                                              outputcoeff=true )
        end

        #Ncoe = length(coeff)
        @assert Ncoe==length(coeff)
        for l=1:Ncoe

            # convert (i,j) from original grid to fmmord
            if simdim==:sim2D
                iorig = cart2lin(ijcoe[l,:],idxconv.grsize)
            elseif simdim==:sim3D
                iorig = cart2lin(ijkcoe[l,:],idxconv.grsize)
            end

            # get index in fmm order
            jfmmord = idxconv.lgrid2fmm[iorig] 

            # the order for P is:
            #   rows: according to coordrec, stdobs, pickobs
            #   columns: according to the FMM order
            ## P[r,jfmmord] =
            P_i[q] = r
            P_j[q] = jfmmord
            P_v[q] = coeff[l]
            q+=1
        end

    end
    
    P = sparse(P_i,P_j,P_v,nrec,nmodpar)
    return P
end

#############################################

"""
$(TYPEDSIGNATURES)

 Set the coefficients (elements) of the derivative matrices in the x and y directions.
"""
function setcoeffderiv!(D::VecSPDerivMat,status::Array,irow::Integer,idxconv::MapOrderGridFMM,
                        codeDeriv_orig::Array{Int64,2},allcoeff::CoeffDerivatives,ijkpt::AbstractVector,
                        colinds::AbstractVector{Int64},colvals::AbstractVector{Float64},
                        idxperm::AbstractVector{Int64},
                        nptsfixedtt::Integer;
                        axis::Symbol, simtype::Symbol)
    
    # get the linear index in the original grid
    iptorig = idxconv.lfmm2grid[irow+nptsfixedtt]
    # get the (i,j) indices in the original grid
    grsize = idxconv.grsize
    ijkpt = MVector(0,0)
    lin2cart!(iptorig,grsize,ijkpt)

    ##=======================
    # ## OLD version
    # # get the codes of the derivatives (+1,-2,...)
    # # extract codes for X and Y
    # codex,codey = codeDxy_orig[iptorig,1],codeDxy_orig[iptorig,2]

    ##=======================
    ## NEW version
    ## If and only if the point has been accepted, then get the codes
    if status[iptorig]==2
        # get the codes of the derivatives (+1,-2,...)
        # extract codes for X and Y
        #codex,codey = codeDeriv_orig[iptorig,1],codeDeriv_orig[iptorig,2]
        codexyz = codeDeriv_orig[iptorig,:]
    else
        codexyz = SVector(0,0,0)
    end
    
    ## row stuff
    ## they must to be zeroed/filled every time!
    tmi = typemax(eltype(colinds)) # highest possible value for given type
    colinds .= tmi # for permsort!() to work with only 2 out of 3 elements
    colvals .= 0.0
    idxperm .= 0
    
    # whX::Int=0
    # whY::Int=0
    if length(grsize)==2
        whXYZ = MVector(0,0)
        ijkcoe = MVector(0,0)
    elseif length(grsize)==3
        whXYZ = MVector(0,0,0)
        ijkcoe = MVector(0,0,0)
    end

    if axis==:X
        if codexyz[1]==0
            ## no derivatives have been used, so no element will
            ##  be added, but the row counter will increase...
            nnzcol = 0
            addrowCSRmat!(D,irow,colinds,colvals,nnzcol)
            return
        else
            code = codexyz[1]
            whXYZ[1] = 1
        end

    elseif axis==:Y
        if codexyz[2]==0
            ## no derivatives have been used, so no element will
            ##  be added, but the row counter will increase...
            nnzcol = 0
            addrowCSRmat!(D,irow,colinds,colvals,nnzcol)
            return
        else        
            code = codexyz[2]
            whXYZ[2] = 1
        end
        
    elseif axis==:Z
        if codexyz[3]==0
            ## no derivatives have been used, so no element will
            ##  be added, but the row counter will increase...
            nnzcol = 0
            addrowCSRmat!(D,irow,colinds,colvals,nnzcol)
            return
        else        
            code = codexyz[3]
            whXYZ[3] = 1
        end
 
    else
        error("setcoeffderiv2D(): if axis==:X ...")
    end
    
    # we skip points on source, so there must always be derivatives
    if all(code.==0)
        error("setcoeffderiv2D: code==[0,0,0]")
    end

    # at this point abs(code) can only be 1 or 2
    abscode = abs(code)
    nnzcol = abscode+1 # max 3
    signcod = sign(code)
    ## select first or second order coefficients
    if abscode==1
        coeff = allcoeff.firstord
    elseif abscode==2
        coeff = allcoeff.secondord
    end   

    if simtype==:cartesian
        ## store coefficients in the struct for sparse matrices
        for p=1:nnzcol 
            # i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
            # j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            ijkcoe .= ijkpt .+ (whXYZ .* signcod*(p-1))
            mycoeff = signcod * coeff[p]
            ##
            #iorig = cart2lin2D(i,j,nx)
            iorig = cart2lin(ijkcoe,grsize)
            ifmmord = idxconv.lgrid2fmm[iorig] 
            ##
            colinds[p] = ifmmord
            colvals[p] = mycoeff
        end

    elseif simtype==:spherical
        ## store coefficients in the struct for sparse matrices
        for p=1:nnzcol 
            # i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
            # j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            ijkcoe .= ijkpt .+ (whXYZ .* signcod*(p-1))
            mycoeff = signcod * coeff[p]
            
            if axis==:X
                mycoeff = signcod * coeff[p]
            elseif axis==:Y
                # in this case coeff depends on radius (index i)
                mycoeff = signcod * coeff[i,p]
            elseif axis==:Z
                error("setcoeffderiv!(): :Z axis not yet implemented for spherical stuff.")
                # # in this case coeff depends on radius (index i)
                # mycoeff = signcod * coeff[i,p]
            end
            ##
            #iorig = cart2lin2D(i,j,nx)
            iorig = cart2lin(ijkcoe,grsize)
            ifmmord = idxconv.lgrid2fmm[iorig]
            ##
            colinds[p] = ifmmord
            colvals[p] = mycoeff
        end       

    end
    
    ##########################################
    ## Add one entire row at a time!
    ##########################################
    ## the following is needed because in Julia's CSR format the
    ##    row indices in every column NEED to be SORTED!
    sortperm!(idxperm,colinds)
    colinds .= colinds[idxperm]
    colvals .= colvals[idxperm]
    addrowCSRmat!(D,irow,colinds,colvals,nnzcol)

    return
end

##############################################################################

function addrowCSRmat!(D::VecSPDerivMat,irow::Integer,colinds::AbstractVector{<:Integer},
                       colvals::AbstractVector{<:Float64},nnzcol::Integer)

    @assert D.lastrowupdated[]==irow-1
    # set 1 to first entry
    if irow==1
        D.iptr[irow] = 1
    end
    # count total nnz values
    D.Nnnz[] += nnzcol 
    # update col pointer
    D.iptr[irow+1] = D.iptr[irow] + nnzcol

    # if there are values, then add them
    for p=1:nnzcol
        j = D.iptr[irow]-1+p
        D.j[j] = colinds[p]
        D.v[j] = colvals[p]
    end
    D.lastrowupdated[] = irow

    return
end

##############################################################################

"""
$(TYPEDSIGNATURES)

Calculates the derivative of the misfit function with respect to the traveltime at the initial points around the source ("onsrc").
    """
function ∂misfit∂srcpos(∂fi_∂us::AbstractMatrix{Float64},
                        lambda_coarse::AbstractArray{Float64},
                        velcart_coarse::Array{Float64,N},
                        ∂ψ_∂u_s::AbstractVector{Float64},
                        ∂ψ_∂u_p::AbstractVector{Float64},
                        srcboxpar_coarse,
                        grd::AbstractGridEik,
                        xyzsrc::Vector{Float64},
                        lambda_fmmord_fine=nothing,
                        grd_fine=nothing,
                        ∂u_h_dtau_s=nothing,
                        H_Areg=nothing,
                        srcrefvars=nothing,
                        srcboxpar_fine=nothing;
                        refinearoundsrc::Bool ) where N

    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    ###################################################################
    ## Derivative of the misfit w.r.t. the traveltime at source nodes
    ###################################################################
    npts = length(∂ψ_∂u_s)
    Ndim = ndims(velcart_coarse)

    if Ndim==2
        ijkpt = MVector(0,0)
        xyzpt = MVector(0.0,0.0)
        curijksrc = MVector(0,0)
    elseif Ndim==2
        ijkpt = MVector(0,0,0)
        xyzpt = MVector(0.0,0.0,0.0)
        curijksrc = MVector(0,0,0)
    end

    dus_dsr = zeros(npts,Ndim)

    if refinearoundsrc
        ###################################################################
        ## WITH REFINEMENT of around the source
        ###################################################################
        velcorn  = srcboxpar_fine.velcorn
        distcorn = srcboxpar_fine.distcorn
        ijksrc = srcboxpar_fine.ijksrc

        npts_fine = size(∂u_h_dtau_s,2)
        dtaus_dsr = zeros(npts_fine,Ndim)
        @show npts_fine,size(∂u_h_dtau_s)
        for p=1:npts_fine
            ## coordinates of point
            if Ndim==2
                xyzpt .= (grd_fine.x[ijksrc[p,1]],grd_fine.y[ijksrc[p,2]])
                ## velocity associated with the above point
                velsrc = srcrefvars.velcart_fine[ijksrc[p,1],ijksrc[p,2]]
            elseif Ndim==3
                xyzpt .= (grd_fine.x[ijksrc[p,1]],grd_fine.y[ijksrc[p,2]],grd_fine.z[ijksrc[p,3]])
                ## velocity associated with the above point
                velsrc = srcrefvars.velcart_fine[ijksrc[p,1],ijksrc[p,2],ijksrc[p,3]]
            end
            ## get the derivative
            dtaus_dsr[p,:] .= partderivttsrcpos(xyzpt,xyzsrc,velsrc)
        end

        trm1 = ∂u_h_dtau_s * dtaus_dsr
        trm2 = H_Areg * dtaus_dsr
        
        dus_dsr .= trm1 .+ trm2
    else
        ###################################################################
        ## NO REFINEMENT around the source
        ## Derivative of the misfit w.r.t. the source position (chain rule)
        ###################################################################
        velcorn  = srcboxpar_coarse.velcorn
        distcorn = srcboxpar_coarse.distcorn
        ijksrc = srcboxpar_coarse.ijksrc

        for p=1:npts 
            ## coordinates of point
            if Ndim==2
                xyzpt .= (grd.x[ijksrc[p,1]],grd.y[ijksrc[p,2]])
                ## velocity associated with the above point
                velsrc = velcart_coarse[ijksrc[p,1],ijksrc[p,2]]
            elseif Ndim==3
                xyzpt .= (grd.x[ijksrc[p,1]],grd.y[ijksrc[p,2]],grd.z[ijksrc[p,3]])
                ## velocity associated with the above point
                velsrc = velcart_coarse[ijksrc[p,1],ijksrc[p,2],ijksrc[p,3]]
            end
            ## get the derivative
            dus_dsr[p,:] .= partderivttsrcpos(xyzpt,xyzsrc,velsrc)
        end
    end

    dψ_ds_r = transpose(dus_dsr) * ∂ψ_∂u_p + transpose(dus_dsr) * ∂ψ_∂u_s
    return dψ_ds_r
end

###########################################################################

function partderivttsrcpos(xyzpt::AbstractVector,xyzsrc::AbstractVector,vel::Real)

    #denom = vel * sqrt((xpt - xsrc)^2 + (ypt - ysrc)^2)
    denom = vel .* sqrt.( sum((xyzpt.-xyzsrc).^2) )
    @assert denom!=0.0 "partderivttsrcpos2D(): Source position and grid node position concide."
    # deriv_x = -(xpt - xsrc) / denom
    # deriv_y = -(ypt - ysrc) / denom
    Ndim = length(xyzpt)
    deriv_xyz = zeros(Ndim)
    for d=1:Ndim
        deriv_xyz[d] = - (xyzpt[d]-xyzsrc[d]) / denom
    end

    return deriv_xyz
end

###########################################################################
