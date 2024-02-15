

#######################################################
##     Misfit of gradient using adjoint              ## 
#######################################################


###############################################################################

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
function gradttime2D(vel::Array{Float64,2}, grd::GridEik2D,coordsrc::Array{Float64,2},
                     coordrec::Vector{Array{Float64,2}},pickobs::Vector{Vector{Float64}},
                     stdobs::Vector{Vector{Float64}},whichgrad::Symbol=:gradvel ;
                     extraparams::Union{ExtraParams,Nothing}=nothing)

    @assert (whichgrad in (:gradvel, :gradsrcloc, :gradvelandsrcloc)) "whichgrad not in (:vel, :srcloc, :velandsrcloc)"
    
    if extraparams==nothing
        extraparams =  ExtraParams()
    end

    if typeof(grd)==Grid2D
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
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

    nsrc = size(coordsrc,1)
    ∂χ∂vel = zeros(n1,n2)
    ∂χ∂xysrc = zeros(nsrc,2)

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
        ∂χ∂vel_all = Vector{Array{Float64,2}}(undef,nchu)
        @sync begin 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async begin
                    ∂χ∂vel_all[s],∂χ∂xysrc[igrs,:] .= remotecall_fetch(calcgradsomesrc2D,wks[s],vel,
                                                                       coordsrc[igrs,:],coordrec[igrs],
                                                                       grd,stdobs[igrs],pickobs[igrs],
                                                                       whichgrad,extraparams )
                end
            end
            ∂χ∂vel .= sum(∂χ∂vel_all)
        end


    elseif extraparams.parallelkind==:sharedmem
        ##====================================
        ## Shared memory        
        ##====================
        nth = Threads.nthreads()
        grpsrc = distribsrcs(nsrc,nth)
        nchu = size(grpsrc,1)            
        ##  do the calculations
        ∂χ∂vel_all = Vector{Array{Float64,2}}(undef,nchu)
        
        Threads.@threads for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @show igrs
            ∂χ∂vel_all[s],∂χ∂xysrc[igrs,:] = calcgradsomesrc2D(vel,view(coordsrc,igrs,:),view(coordrec,igrs),
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

###############################################################################

"""
$(TYPEDSIGNATURES)


Calculate the gradient for some requested sources 
"""
function calcgradsomesrc2D(vel::Array{Float64,2},xysrc::AbstractArray{Float64,2},
                           coordrec::AbstractVector{Array{Float64,2}},grd::GridEik2D,
                           stdobs::AbstractVector{Vector{Float64}},pickobs1::AbstractVector{Vector{Float64}},
                           whichgrad::Symbol,extrapars::ExtraParams )
                           
    nx,ny=size(vel)
    nsrc = size(xysrc,1)

    if whichgrad==:gradvel
        gradvel1 = zeros(nx,ny)
        gradvelall = zeros(nx,ny)
        gradsrcpos = nothing

    elseif whichgrad==:gradsrcloc
        gradvel1 = nothing
        gradvelall = nothing
        gradsrcpos = zeros(eltype(vel),nsrc,2) # ,2)-> 2D
        
    elseif whichgrad==:gradvelandsrcloc
        gradvel1 = zeros(nx,ny)
        gradvelall = zeros(nx,ny)
        gradsrcpos = zeros(eltype(vel),nsrc,2) # ,2)-> 2D

    end

    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars = FMMvars2D(nx,ny,amIcoarsegrid=true,refinearoundsrc=extrapars.refinearoundsrc,
                        allowfixsqarg=extrapars.allowfixsqarg)
    
    ## pre-allocate discrete adjoint variables for coarse grid
    adjvars = AdjointVars2D(nx,ny)

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
            fmmvars_fine,adjvars_fine,grd_fine,srcrefvars = ttFMM_hiord!(fmmvars,vel,view(xysrc,s,:),
                                                                       grd,adjvars,extrapars)
        else
            ##
            ## NO refinement around the source
            ## 
            ttFMM_hiord!(fmmvars,vel,view(xysrc,s,:),grd,adjvars,extrapars)
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
                
        solveadjointgetgrads_singlesrc!(gradvel1,tmpgradsrcpos,
                                        fmmvars,adjvars,
                                        xysrc[s,:],coordrec[s],pickobs1[s],stdobs[s],
                                        vel,grd,whichgrad,extrapars.refinearoundsrc,
                                        fmmvars_fine=fmmvars_fine,
                                        adjvars_fine=adjvars_fine,
                                        grd_fine=grd_fine,
                                        srcrefvars=srcrefvars)


        ###########################################
        # projection operator P ordered according to FMM order        
        # if extrapars.refinearoundsrc
        #     # P from fine grid = 
        #     H_fmmord_fine = calcprojttfmmord(fmmvars.ttime,grd,adjvars.idxconv,coordrec[s])
        # end
        # # P from coarse grid
        # P_fmmord = calcprojttfmmord(fmmvars.ttime,grd,adjvars.idxconv,coordrec[s])

        ###########################################
        # discrete adjoint formulation for gradient(s) with respect to velocity and/or source position
        # gradvel1,gradsrcpos1[s,:] .= discradjoint2D_FMM_SINGLESRC(adjvars,P_fmmord1,pickobs1[s],
        #                                                      stdobs[s],vel,xysrc[s,:],grd)
        # if gradsrcpos==nothing
        #     tmpgradsrcpos = gradsrcpos
        # else
        #     tmpgradsrcpos = view(gradsrcpos,s,:)
        # end
        # discradjoint2D_FMM_SINGLESRC!(gradvel1,tmpgradsrcpos,adjvars,P_fmmord,
        #                               pickobs1[s],stdobs[s],vel,xysrc[s,:],grd,whichgrad,
        #                               refinearoundsrc=extrapars.refinearoundsrc ) #,∂u_h∂x_s=∂u_h∂x_s)

        
        if whichgrad==:gradvel || whichgrad==:gradvelandsrcloc
            ###########################################
            ## smooth gradient (with respect to velocity) around the source
            smoothgradaroundsrc2D!(gradvel1,view(xysrc,s,:),grd,
                                   radiuspx=extrapars.radiussmoothgradsrc)
            ##########################################
            ## add up gradients from different sources
            gradvelall .+= gradvel1
        end
        
    end
 
    if extrapars.manualGCtrigger
        # trigger garbage collector
        #println("Triggering GC")
        GC.gc()
    end
    
    return gradvelall,gradsrcpos
end 

##############################################################################

"""
$(TYPEDSIGNATURES)

 Set the coefficients (elements) of the derivative matrices in the x and y directions.
    """
function setcoeffderiv2D!(D::VecSPDerivMat,irow::Integer,idxconv::MapOrderGridFMM2D,
                          codeDxy_orig::Array{Int64,2},allcoeff::CoeffDerivatives,ijpt::AbstractVector,
                          colinds::AbstractVector{Int64},colvals::AbstractVector{Float64},
                          idxperm::AbstractVector{Int64},
                          nptsfixedtt::Integer;
                          axis::Symbol, simtype::Symbol)
    
    # get the linear index in the original grid
    iptorig = idxconv.lfmm2grid[irow+nptsfixedtt]
    # get the (i,j) indices in the original grid
    nx = idxconv.nx
    lin2cart2D!(iptorig,nx,ijpt)
    igrid = ijpt[1]
    jgrid = ijpt[2]
    # get the codes of the derivatives (+1,-2,...)
    # extract codes for X and Y
    codex,codey = codeDxy_orig[iptorig,1],codeDxy_orig[iptorig,2]

    ## row stuff
    ## they must to be zeroed/filled every time!
    tmi = typemax(eltype(colinds)) # highest possible value for given type
    colinds[:] .= tmi # for permsort!() to work with only 2 out of 3 elements
    colvals[:] .= 0.0
    idxperm[:] .= 0
    
    whX::Int=0
    whY::Int=0

    if axis==:X
        if codex==0
            ## no derivatives have been used, so no element will
            ##  be added, but the row counter will increase...
            nnzcol = 0
            addrowCSRmat!(D,irow,colinds,colvals,nnzcol)
            return
        else
            code = codex
            whX=1
            whY=0
        end

    elseif axis==:Y
        if codey==0
            ## no derivatives have been used, so no element will
            ##  be added, but the row counter will increase...
            nnzcol = 0
            addrowCSRmat!(D,irow,colinds,colvals,nnzcol)
            return
        else        
            code = codey
            whX=0
            whY=1
        end
        
    else
        error("setcoeffderiv2D(): if axis==:X ...")
    end
       
    # we skip points on source, so there must always be derivatives
    if code==[0,0]
        error("setcoeffderiv2D: code==[0,0]")
    end


    # at this point abs(code) can only be 1 or 2
    abscode = abs(code)
    nnzcol = abscode+1
    signcod = sign(code)
    ## select first or second order coefficients
    if abscode==1
        coeff = allcoeff.firstord
    else
        coeff = allcoeff.secondord
    end   

    if simtype==:cartesian
        ## store coefficients in the struct for sparse matrices
        for p=1:nnzcol 
            i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
            j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            mycoeff = signcod * coeff[p]
            ##
            iorig = cart2lin2D(i,j,nx)
            ifmmord = idxconv.lgrid2fmm[iorig] 
            ##
            colinds[p] = ifmmord
            colvals[p] = mycoeff
        end

    elseif simtype==:spherical
        ## store coefficients in the struct for sparse matrices
        for p=1:nnzcol 
            i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
            j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            if axis==:X
                mycoeff = signcod * coeff[p]
            elseif axis==:Y
                # in this case coeff depends on radius (index i)
                mycoeff = signcod * coeff[i,p]
            end
            ##
            iorig = cart2lin2D(i,j,nx)
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
    colinds[:] .= colinds[idxperm]
    colvals[:] .= colvals[idxperm]
    addrowCSRmat!(D,irow,colinds,colvals,nnzcol)

    return
end

##############################################################################

"""
$(TYPEDSIGNATURES)

 Projection operator P containing interpolation coefficients, ordered according to FMM order.
"""
function calcprojttfmmord(ttime::AbstractArray{Float64},grd::GridEik,idxconv::MapOrderGridFMM,
                          coordrec::AbstractArray{Float64})

    if typeof(grd)==Grid2D
        simdim = :sim2D
    elseif typeof(grd)==Grid3D
        simdim = :sim3D
    else
        error("calcprojttfmmord(): Wrong grd argument.")
    end

    # calculate the coefficients and their indices using bilinear interpolation
    nmodpar = length(ttime)
    nrec = size(coordrec,1)
    Ncoe = 4 
    P_i = zeros(Int64,nrec*Ncoe)
    P_j = zeros(Int64,nrec*Ncoe)
    P_v = zeros(nrec*Ncoe)

    q::Int64=1
    for r=1:nrec

        if simdim==:sim2D 
            coeff,_,ijcoe = bilinear_interp( ttime,grd,view(coordrec,r,:),
                                           outputcoeff=true )
        elseif simdim==:sim3D 
            coeff,_,ijcoe = trilinear_interp( ttime,grd,view(coordrec,r,:),
                                            outputcoeff=true )
        end

        #Ncoe = length(coeff)
        @assert Ncoe==length(coeff)
        for l=1:Ncoe

            # convert (i,j) from original grid to fmmord
            if simdim==:sim2D
                i,j = ijcoe[l,1],ijcoe[l,2]
                iorig = cart2lin2D(i,j,idxconv.nx)
            elseif simdim==:sim3D
                i,j,k = ijcoe[l,1],ijcoe[l,2],ijcoe[l,3]
                iorig = cart2lin3D(i,j,k,idxconv.nx,idxconv.ny)
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

##############################################################################

function solveadjointgetgrads_singlesrc!(gradvel1::Union{AbstractArray{Float64},Nothing},
                                         gradsrcpos1::Union{AbstractArray{Float64},Nothing},
                                         fmmvars::FMMvars2D,
                                         adjvars::AdjointVars2D,
                                         xysrc::AbstractArray{Float64},
                                         xyrecs::AbstractArray{Float64},
                                         pickobs::AbstractVector{Float64},
                                         stdobs::AbstractVector{Float64},
                                         vel2d::AbstractArray{Float64},
                                         grd::GridEik2D,
                                         whichgrad::Symbol,
                                         refinearoundsrc::Bool;
                                         fmmvars_fine::Union{FMMvars2D,Nothing}=nothing,
                                         adjvars_fine::Union{AdjointVars2D,Nothing}=nothing,
                                         grd_fine::Union{GridEik2D,Nothing}=nothing,
                                         srcrefvars::Union{AbstractArray{Float64},SrcRefinVars2D,Nothing}=nothing
                                         )
    
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    println("=== START gradient coarse grid ===" )
    ##======================================================
    # Projection operator P ordered according to FMM order
    #  The order for P is:
    #    rows: according to coordrec, stdobs, pickobs
    #    columns: according to the FMM order
    P = calcprojttfmmord(fmmvars.ttime,grd,adjvars.idxconv,xyrecs)
                                      
    ##==============================
    tt_fmmord = adjvars.fmmord.ttime
    
    ###########################################
    # Derivative (along x) matrix, row-deficient
    vecDx = adjvars.fmmord.vecDx
    ###########################################
    # Derivative (along y) matrix, row-deficient
    vecDy = adjvars.fmmord.vecDy

    #######################################################
    ##  right hand side adj eq. == -(∂ψ/∂u_p)^T
    #######################################################
    #rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)
    fact2∂ψ∂u = ((P*tt_fmmord).-pickobs)./stdobs.^2
    # bool vector with false for src columns
    rq = .!adjvars.fmmord.onsrccols
    P_Breg = P[:,rq] # remove columns
    rhs = - transpose(P_Breg) * fact2∂ψ∂u

       
    ################################
    ##   left hand side adj eq.
    ################################
    ## compute the lhs terms
    twoDttDRx,twoDttDSx = calcadjlhsterms(vecDx,tt_fmmord,adjvars.fmmord.onsrccols)
    twoDttDRy,twoDttDSy = calcadjlhsterms(vecDy,tt_fmmord,adjvars.fmmord.onsrccols) 

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

    ## We have a CSR matrix as vectors, we need a *transpose* of it,
    ##  so we construct directly a CSC by *exchanging* i and j indices
    Nxnnz = twoDttDRx.Nnnz[]
    Nix = twoDttDRx.Nsize[1]
    Njx = twoDttDRx.Nsize[2]
    lhsterm1 = SparseMatrixCSC(Njx,Nix,twoDttDRx.iptr[1:Nix+1],
                               twoDttDRx.j[1:Nxnnz],twoDttDRx.v[1:Nxnnz])
 
    ## We have a CSR matrix as vectors, we need a *transpose* of it,
    ##  so we construct directly a CSC by *exchanging* i and j indices
    Nynnz = twoDttDRy.Nnnz[]
    Niy = twoDttDRy.Nsize[1]
    Njy = twoDttDRy.Nsize[2]
    lhsterm2 = SparseMatrixCSC(Njy,Niy,twoDttDRy.iptr[1:Niy+1],
                               twoDttDRy.j[1:Nynnz],twoDttDRy.v[1:Nynnz])

    # They are already transposed, so only add them
    tmplhs = lhsterm1 .+ lhsterm2

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
    lambda_fmmord = lhs\rhs

    
    ##----------------------------------------------------------
    if whichgrad==:gradsrcloc || whichgrad==:gradvelandsrcloc
        
        if refinearoundsrc
            error("Gradient with respect to source location with refinement of the grid currently broken... Aborting.")
        end

        ##############################################
        ##  Derivative with respect to the traveltime
        ##     at the "onsrc" points
        ##  compute gradient
        ##############################################
        gradsrcpos1 .= ∂misfit∂initsrcpos2D(twoDttDSx,twoDttDSy,tt_fmmord,
                                            fmmvars.srcboxpar,
                                            adjvars.idxconv,lambda_fmmord,vel2d,
                                            adjvars.fmmord.onsrccols,xysrc,grd,
                                            refinearoundsrc=refinearoundsrc ) 
    end

    
    if whichgrad==:gradvel || whichgrad==:gradvelandsrcloc
      
        ##===============================================
        # Gradient with respect to velocity, term T2Vb
        ##===============================================
        # Reorder lambda from fmmord to original grid!!
        idxconv = adjvars.idxconv
        N = length(lambda_fmmord)
        nptsonsrc = count(adjvars.fmmord.onsrccols)

        ijpt = MVector(0,0)
        #Nall = length(idxconv.lfmm2grid)
        for p=1:N
            # The following two lines (p+nptsonsrc) is a way to map the
            #  "full" indices to the "reduced" set of indices (no source region)
            iorig = idxconv.lfmm2grid[p+nptsonsrc] # skip pts on src
            # get the i,j indices for vel and grad
            lin2cart2D!(iorig,idxconv.nx,ijpt)
            i,j = ijpt
            # get the contribution of T2Vb to the gradient
            gradvel1[i,j] = 2.0 * lambda_fmmord[p] / vel2d[i,j]^3
        end


        ##########################################
        
        if refinearoundsrc==false
            #################################################
            #  NO refinement around the source
            #################################################

            ## source box parameters
            velcorn  = fmmvars.srcboxpar.velcorn
            distcorn = fmmvars.srcboxpar.distcorn
            ijsrcreg = fmmvars.srcboxpar.ijsrc

            ## derivative of traveltime at source nodes w.r.t. velocity in region A
            dus_dva = - diagm(distcorn) * 1.0 ./ velcorn.^2
            
            ##===============================================p
            # Gradient with respect to velocity, term T2Va
            ##===============================================
            # deriv. of the implicit forw. mod. w.r.t. u_s
            ∂fi_∂us = twoDttDSx .+ twoDttDSy
            ## FMM ordering for ∂fi_∂us!!!
            tmpdfidva = ∂fi_∂us * dus_dva        
            ## FMM ordering for lambda_fmmord!!!
            T2Va = transpose(lambda_fmmord) * tmpdfidva
            ## add contribution to the gradient
            for i=1:length(T2Va)
                m,n = ijsrcreg[i,:]
                gradvel1[m,n] += T2Va[i]
            end

            ##===============================================
            # Gradient with respect to velocity, term T1Va 
            ##===============================================
            # T1v = ∂ψ/∂u_s
            ## using fact2∂ψ∂u from the calculations above!
            ## fact2∂ψ∂u = ((P*tt_fmmord).-pickobs)./stdobs.^2
            # pick only columns belonging to the source region
            P_Areg = P[:,adjvars.fmmord.onsrccols]
            ∂ψ_∂u_s = transpose(P_Areg) * fact2∂ψ∂u

            ## add contribution to the gradient       
            tmpT1Va = transpose(∂ψ_∂u_s) * dus_dva
            for i=1:length(tmpT1Va)
                m,n = ijsrcreg[i,:]
                #@show i,m,n,tmpgr1[i]
                gradvel1[m,n] += tmpT1Va[i]
            end

            println("=== END gradient coarse grid ===")

        
        elseif refinearoundsrc
            
            println("=== END gradient coarse grid ===")

            #############################################
            ##   REFINEMENT around the source
            #############################################
            # #error("Gradient with respect to velocity with refinement of the grid currently missing a piece... Aborting.")
            ## Solve the adjoint equation in the fine grid
            duh_dva = solveadjointfinegrid!(fmmvars_fine,adjvars_fine,
                                            grd_fine,srcrefvars)

        end

    end
    
    return 
end

##############################################################################

function solveadjointfinegrid!(fmmvars,adjvars,grd,srcrefvars)

    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    println("=== START adjoint fine grid ===")
    
    ## find the last value of traveltime computed in the fine grid
    ##    (The fine grid stop when hitting a boundary...)
    lasttt = findfirst( adjvars.fmmord.ttime.==0.0)-1
    ## extract computed traveltime values
    tt_fmmord = adjvars.fmmord.ttime[1:lasttt]
    @show size(tt_fmmord)

    ## bool array of grid points (in FMM order) on h (u_s) points
    onHpts = adjvars.fmmord.onhpoints#[node]
    ## number of model parameters (== computed traveltimes)
    Naccpts = length(tt_fmmord)
    ## indices of coinciding with u_s (h)
    idxonHpts = findall(onHpts) # get all "true" values
    Nhpts = length(idxonHpts)
    @show Nhpts

    ###########################################
    H_i = zeros(Int64,Nhpts)
    H_j = zeros(Int64,Nhpts)
    H_v = zeros(Float64,Nhpts)
    ##H = ??? #idxonHpts


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
        H_v[q] = 1
        q+=1
    end
    H = sparse(H_i,H_j,H_v,Nhpts,Naccpts)
    
    ###########################################
    # Derivative (along x) matrix, row-deficien
    vecDx = adjvars.fmmord.vecDx
    ###########################################
    # Derivative (along y) matrix, row-deficien
    vecDy = adjvars.fmmord.vecDy

    #######################################################
    ##  right hand side adj eq. == -(adjoint source)^T
    #######################################################
    # @show size(H)
    # @show size(.!adjvars.fmmord.onsrccols[1:lasttt])
    # bool vector with false for src columns
    #    only part of the rectangular fine grid has been used, so [1:lasttt]
    rq = .!adjvars.fmmord.onsrccols[1:lasttt]  
    HD = H[:,rq] # remove columns
    rhs = - transpose(HD)

    ################################
    ##   left hand side adj eq.
    ################################
    ## compute the lhs terms
    twoDttDRx,twoDttDSx = calcadjlhsterms(vecDx,tt_fmmord,adjvars.fmmord.onsrccols)
    twoDttDRy,twoDttDSy = calcadjlhsterms(vecDy,tt_fmmord,adjvars.fmmord.onsrccols) 

    ## We have a CSR matrix as vectors, we need a *transpose* of it,
    ##  so we construct directly a CSC by *exchanging* i and j indices
    Nxnnz = twoDttDRx.Nnnz[]
    Nix = twoDttDRx.Nsize[1]
    Njx = twoDttDRx.Nsize[2]
    lhsterm1 = SparseMatrixCSC(Njx,Nix,twoDttDRx.iptr[1:Nix+1],
                               twoDttDRx.j[1:Nxnnz],twoDttDRx.v[1:Nxnnz])
 
    ## We have a CSR matrix as vectors, we need a *transpose* of it,
    ##  so we construct directly a CSC by *exchanging* i and j indices
    Nynnz = twoDttDRy.Nnnz[]
    Niy = twoDttDRy.Nsize[1]
    Njy = twoDttDRy.Nsize[2]
    lhsterm2 = SparseMatrixCSC(Njy,Niy,twoDttDRy.iptr[1:Niy+1],
                               twoDttDRy.j[1:Nynnz],twoDttDRy.v[1:Nynnz])

    # They are already transposed, so only add them
    tmplhs = lhsterm1 .+ lhsterm2

    ## make sure it's recognised as upper triangular...
    lhs = UpperTriangular(tmplhs)
    
    # @show Nix,Njx
    # @show Niy,Njy
    # @show prod(size(vel2d_fine))
    # @show count(adjvars.fmmord.onsrccols)

    ## OLD stuff...
    # ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
    # ##   the solver (\) does not use the correct sparse algo for matrix division
    # tmplhs = copy(transpose( (2.0.*Diagonal(Dx*tt)*Dx) .+ (2.0.*Diagonal(Dy*tt)*Dy) ))
    # lhs = UpperTriangular(tmplhs)

    ############################################################
    ##  solve the linear system to get the adjoint variable
    ############################################################
    # Remark: "lambda_fmmord" excludes the points on the source (in the fine grid)
    lambda_fmmord = lhs\rhs  
    #@show size(lambda_fmmord)

    # Reorder lambda from fmmord to original grid!!
    idxconv = adjvars.idxconv
    N = size(lambda_fmmord,1) # lambda is 2D for grid refinement
    nptsonsrc = count(adjvars.fmmord.onsrccols)
    #@show nptsonsrc

    ##-------------------------------------------------------
    #gradve1_fine = zeros(srcrefvals.grd.nx,srcrefvals.grd.ny)

    ##========================================
    ##  Compute the T2Vd term
    ##========================================
    #T2Vd = zeros(N,Nhpts)
    ## The gradient in the fine grid is not a rectangular grid but it
    ##    depends on when the FMM has stopped (so a vector with 'Naccpts'
    ##    entries, not a 2D array)
    #duh_dwq = zeros(Naccpts,Nhpts)
    duh_dwq = zeros(Nhpts,grd.nx*grd.ny)

    @show N
    @show size(srcrefvars.vel2d_fine)
    @show size(lambda_fmmord)
    @show size(duh_dwq)

    ijpt = MVector(0,0)
    for p=1:N
        # The following two lines (p+nptsonsrc) is a way to map the
        #  "full" indices to the "reduced" set of indices (no source region)
        iorig = idxconv.lfmm2grid[p+nptsonsrc] # skip pts on src
        # get the i,j indices for vel and grad
        lin2cart2D!(iorig,idxconv.nx,ijpt)
        i,j = ijpt
        for h=1:Nhpts   
            # get the contribution of T2Vb to the gradient
            #T2Vd[p,h] = 2.0 * lambda_fmmord[p,h] / vel2d_fine[i,j]^3
            # T2Vd
            duh_dwq[h,iorig] = 2.0 * lambda_fmmord[p,h] / srcrefvars.vel2d_fine[i,j]^3
        end
    end
    @show N,Nhpts
    #@show size(T2Vd)
    
    ## source box parameters
    velcorn  = fmmvars.srcboxpar.velcorn
    distcorn = fmmvars.srcboxpar.distcorn
    ijsrcreg = fmmvars.srcboxpar.ijsrc

    ## derivative of \tau_s w.r.t. w_a
    dtaus_dwa = - diagm(distcorn) * 1.0 ./ velcorn.^2


    ##========================================
    ## Compute the T2Vc term 
    ##========================================
    # deriv. of the implicit forw. mod. w.r.t. u_s
    ∂gi_∂taus = twoDttDSx .+ twoDttDSy
    ## FMM ordering for ∂gi_∂taus!!!
    tmpdgidwa = ∂gi_∂taus * dtaus_dwa        
    ## FMM ordering for lambda_fmmord!!!
    T2Vc = transpose(lambda_fmmord) * tmpdgidwa
    @show size(T2Vc)
    ## add contribution to the gradient in the fine grid
    for a=1:size(T2Vc,2)
        m,n = ijsrcreg[a,:]
        iorig = cart2lin2D(m,n,grd.nx)
        for h=1:Nhpts
            duh_dwq[h,iorig] += T2Vc[h,a]
        end
    end

    ##========================================
    ## Compute the T1Vc term 
    ##========================================
    H_Areg = H[:,adjvars.fmmord.onsrccols[1:lasttt]]
    T1Vc = H_Areg * dtaus_dwa
    @show size(T1Vc)
    for a=1:size(T1Vc,2)
        m,n = ijsrcreg[a,:]
        iorig = cart2lin2D(m,n,grd.nx)
        for h=1:Nhpts
            duh_dwq[h,iorig] += T1Vc[h,a]
        end
    end
    #println("duh_dwq:")
    #display(duh_dwq)
    @show extrema(duh_dwq)
    
    ##========================================
    ## Compute dw1/dwa
    ##========================================
    # filter grid point which have been actually used 
    #dwq_dva = srcrefvars.nearneigh_oper#[ , ]
    @show size(srcrefvars.nearneigh_oper)
    dwq_dva = zeros(grd.nx*grd.ny,Nhpts)

    nx_window_coarse = srcrefvars.nxny_window_coarse[1]
    @show nptsonsrc
    @show size(idxconv.lfmm2grid)
    @show length(idxonHpts)
    @show length(onHpts)
    for p=1:Naccpts
        iorig = idxconv.lfmm2grid[p]
        for (a,a_fine) in enumerate(idxonHpts)
            a_coarse = srcrefvars.nearneigh_idxcoarse[a_fine]
            dwq_dva[p,a] = srcrefvars.nearneigh_oper[iorig,a_coarse]
        end
    end


    # println("dwq_dva:")
    # display(dwq_dva)
    @show extrema(dwq_dva)
    
   #  ##========================================
   #  ## duh_dva
   #  ##========================================
   #  duh_dva = duh_dwq * dwq_dva

     println("=== END adjoint fine grid ===")    
    return 
end

#######################################################################################

"""
$(TYPEDSIGNATURES)

Calculates the following maintaining the sparse structure:
     2 * diag(D * tt) * Dr
"""
function calcadjlhsterms(vecD::VecSPDerivMat,tt::Vector{Float64},
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
    ncolidx = vecD.Nnnz[] - nsrcpts
    two_Dtt_DR = VecSPDerivMat( iptr=zeros(Int64,nrows+1), j=zeros(Int64,ncolidx),
                                v=zeros(ncolidx), Nsize=[nrows,ncolsR] )
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
    DS_i = Vector{Int64}(undef,ncolidx)
    DS_j = Vector{Int64}(undef,ncolidx)
    DS_val = Vector{Float64}(undef,ncolidx)
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

#######################################################################################


#######################################################################################

# """
# $(TYPEDSIGNATURES)

#  Attempt to reconstruct how derivatives have been calculated in the area of 
#   the refinement of the source. Updates the codes for derivatives used to construct
#    Dx and Dy.
# """
# function derivaroundsrcfmm2D!(lseq::Integer,idxconv::MapOrderGridFMM2D,codeD::MVector)
#     # lipt = index in fmmord in sequential order (1,2,3,4,...)
    
#     nx = idxconv.nx
#     ny = idxconv.ny

#     ijpt = MVector(0,0)
#     idxpt = idxconv.lfmm2grid[lseq]
#     lin2cart2D!(idxpt,nx,ijpt)
#     ipt,jpt = ijpt[1],ijpt[2]
    
#     codeD[:] .= 0
#     for axis=1:2

#         chosenidx = lseq
#         for dir=1:2

#             ## map the 4 cases to an integer as in linear indexing...
#             lax = dir + 2*(axis-1)
#             if lax==1 # axis==1
#                 ish = 1
#                 jsh = 0
#             elseif lax==2 # axis==1
#                 ish = -1
#                 jsh = 0
#             elseif lax==3 # axis==2
#                 ish = 0
#                 jsh = 1
#             elseif lax==4 # axis==2
#                 ish = 0
#                 jsh = -1
#             end
            
#             ##=========================
#             ## first order
#             i = ipt+ish
#             j = jpt+jsh

#             ## check if on boundaries
#             isonb1st,isonb2nd = isonbord(i,j,nx,ny)

#             if !isonb1st
#                 ## calculate the index of the neighbor in the fmmord
#                 l = cart2lin2D(i,j,nx)
#                 #idxne1 = findfirst(idxconv.lfmm2grid.==l)
#                 idxne1 = idxconv.lgrid2fmm[l]
                
#                 if idxne1!=nothing && idxne1<chosenidx                   

#                     # to make sure we chose correctly the direction
#                     chosenidx = idxne1
#                     # save derivative choices [first order]
#                     axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

#                     if !isonb2nd
#                         ##=========================
#                         ## second order
#                         i = ipt + 2*ish
#                         j = jpt + 2*jsh
                        
#                         ## calculate the index of the neighbor in the fmmord
#                         l = cart2lin2D(i,j,nx)
#                         idxne2 = idxconv.lgrid2fmm[l]
                        
#                         ##===========================================================
#                         ## WARNING! The traveltime for the second order point must
#                         ##   be smaller than the one for the first order for selecting
#                         ##   second order. Otherwise first order only.
#                         ##  Therefore we compare idxne2<idxne1 instead of idxne2<idxpt
#                         ##===========================================================
#                         if idxne2<idxne1
#                             # save derivative choices [second order]
#                             axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)
#                         end

#                     end
#                 end
#             end            
#         end
#     end

#     return 
# end

#######################################################################################

"""
$(TYPEDSIGNATURES)

Calculates the derivative of the misfit function with respect to the traveltime at the initial points around the source ("onsrc").
"""
function ∂misfit∂initsrcpos2D(twoDttDSx,twoDttDSy,tt,
                              srcboxpar,
                              idxconv,
                              lambda::AbstractArray{Float64},
                              vel2d::Array{Float64,2},
                              sourcerows,xysrc,grd::GridEik2D;
                              refinearoundsrc::Bool ) #,
                              #∂u_h∂x_s)
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    nx,ny = size(vel2d)


    ###################################################################
    ## Derivative of the misfit w.r.t. the traveltime at source nodes
    ###################################################################
    tmp1 = twoDttDSx .+ twoDttDSy
    
    npts = size(twoDttDSx,2)
    ∂χ∂t_src = zeros(npts)
    for p=1:npts
        ∂χ∂t_src[p] = dot(lambda,tmp1[:,p])
    end
    ########################################
   
    if refinearoundsrc
        ###################################################################
        ## WITH REFINEMENT of around the source
        ###################################################################
        # # If there is refinement around the source, an extra factor needs
        # #  to be computed in the chain rule 
        # #   ∂ψ/∂u_h * ∂u_h/∂u_s * ∂u_s/∂x_s
        # ∂u_h∂u_s = calc∂u_h∂u_s_finegrid(fmmvars_fine,adjvars_coarse,xysrc,grd)
        dχdx_src = dot( ∂χ∂t_src, ∂u_h∂x_s) # along x
        dχdy_src = dot( ∂χ∂t_src, ∂u_h∂y_s) # along y

        
    else
        ###################################################################
        ## NO REFINEMENT of around the source
        ## Derivative of the misfit w.r.t. the source position (chain rule)
        ###################################################################
        ijpt = MVector(0,0)
        xypt = MVector(0.0,0.0)
        curijsrc = MVector(0,0)
        derpos = zeros(npts,2)

        ##
        velcorn  = srcboxpar.velcorn
        distcorn = srcboxpar.distcorn
        ijsrc = srcboxpar.ijsrc

        for p=1:npts 
            ## coordinates of point
            xypt .= (grd.x[ijsrc[p,1]],grd.y[ijsrc[p,2]])
            ## velocity associated with the above point
            velsrc = vel2d[ijsrc[p,1],ijsrc[p,2]]
            ## get the derivative
            derpos[p,:] .= partderivttsrcpos2D(xypt,xysrc,velsrc)
        end
        
        dχdx_src = dot( ∂χ∂t_src, derpos[:,1] ) # along x
        dχdy_src = dot( ∂χ∂t_src, derpos[:,2] ) # along y
    end

    return dχdx_src,dχdy_src
end

###########################################################################

function partderivttsrcpos2D(xypt::AbstractVector,xysrc::AbstractVector,vel::Real)

    xpt = xypt[1]
    ypt = xypt[2]
    xsrc = xysrc[1]
    ysrc = xysrc[2]

    denom = vel * sqrt((xpt - xsrc)^2 + (ypt - ysrc)^2)
    @assert denom!=0.0 "partderivttsrcpos2D(): Source position and grid node position concide."
    deriv_x = -(xpt - xsrc) / denom
    deriv_y = -(ypt - ysrc) / denom
    
    return deriv_x,deriv_y
end

###########################################################################


# function calc_∂u_h∂x_s_finegrid(fmmvars,adjvars,xysrc,vel2d,grd)

#     # Derivative (along x) matrix, row-deficien
#     vecDx = adjvars.fmmord.vecDx
#     # Derivative (along y) matrix, row-deficien
#     vecDy = adjvars.fmmord.vecDy

#     onsrccols = adjvars.fmmord.onsrccols
#     onhpoints = adjvars.fmmord.onhpoints
#     tt = adjvars.fmmord.ttime

#     ######################################
#     ## First part: compute ∂u_h/∂u_s
#     ######################################

#     ##
#     ##  ∂f_h/∂u_h * ∂u_h/∂u_s = - ∂f_h/∂u_s
#     ##
#     ##  "h" points
#     ##  A = ∂f_h/∂u_h
#     ##  A = 2 ( Dx_ij * tt_j ) * Dx_ih + 2 ( Dy_ij * tt_j ) * Dy_ih
#     ##   size(A) => N_i x N_h  with i != s
#     ##
#     ##  "s" points
#     ##  B = ∂f_h/∂u_s
#     ##  B = 2 ( Dx_ij * tt_j ) * Dx_is + 2 ( Dy_ij * tt_j ) * Dy_is
#     ##   size(A) => N_i x N_h  with i != s
#     ##
#     ##  X = ∂u_h/∂u_s
#     ##  A * X = - B
#     ##

#     twoDttDH_x,twoDttDS_x = calcABtermsfinegrid(vecDx,tt,onsrccols,onhpoints)
#     twoDttDH_y,twoDttDS_y = calcABtermsfinegrid(vecDy,tt,onsrccols,onhpoints)
#     A = twoDttDH_x .+ twoDttDH_y
#     B = twoDttDS_x .+ twoDttDS_y

    
#     # solve the linear systemS
#     ∂u_h∂u_s = A \ (-B)

#     println("A")
#     display(A)
#     println("B")
#     display(B)
#     println("∂u_h∂u_s")
#     display(∂u_h∂u_s)

#     ######################################
#     ## END First part: compute ∂u_h/∂u_s
#     ######################################

    
#     ######################################
#     ## Second part part: compute ∂u_s/∂x_s
#     ######################################
#     nx,ny = size(vel2d)
#     npts = size(B,2)
#     ijpt = MVector(0,0)
#     xypt = MVector(0.0,0.0)
#     curijsrc = MVector(0,0)
#     derpos = zeros(npts,2)

#     for p=1:npts 
#         # ifmmord_srcpts = count(sourcerows[1:p])
#         # iorig_srcpts = idxconv.lfmm2grid[ifmmord_srcpts]
#         iorig_srcpts = adjvars.idxconv.lfmm2grid[p] 

#         ##################################
#         ## Source velocity stuff
#         # Get the correct velocity for all 4 vertices
#         #  The velocity should be the same (same ii,jj) for all 4 points
#         #velpts[p] = vel2d[iorig_srcpts]
#         lin2cart2D!(iorig_srcpts,grd.nx,curijsrc)
#         xps = grd.x[curijsrc[1]] 
#         yps = grd.y[curijsrc[2]]
#         ii = Int(floor((xysrc[1]-grd.xinit)/grd.hgrid)) +1
#         jj = Int(floor((xysrc[2]-grd.yinit)/grd.hgrid)) +1 
#         velpt = vel2d[ii,jj]
        
#         ##################################
#         ## Derivatives for each source point
#         ## CONVERT linear to cartesian 2D to get point position
#         lin2cart2D!(iorig_srcpts,nx,ijpt)
#         xypt .= (grd.x[ijpt[1]], grd.y[ijpt[2]])

#         derpos[p,:] .= partderivttsrcpos2D(xypt,xysrc,velpt)
#     end
    
#     ######################################
#     ## END Second part part: compute ∂u_s/∂x_s
#     ######################################


#     ######################################
#     ## Third part: combine results
#     ##             ∂u_h/∂u_s * ∂u_s/∂x_s
#     ######################################
#     ∂u_h∂x_s = ∂u_h∂u_s * derpos

#     return ∂u_h∂x_s
# end

###########################################################################

# function calcABtermsfinegrid(vecD::VecSPDerivMat,tt::Vector{Float64},
#                              onsrccols::Vector{Bool},onhpoints::Vector{Bool})
    
#     # Remark: equivalent to CSR format 
#     # ## CSR matrix-vector product
#     # tmp1 = zeros(nrows)
#     # for i=1:nrows
#     #     # pi=pointers to column indices
#     #     for l=vecD.pi[i]:vecD.pi[i+1]-1
#     #         j = vecD.j[l]
#     #         # dot product
#     #         tmp1[i] += vecD.v[l]*tt[j]
#     #     end
#     # end

#     # size of D^x or D^y
#     nrows,ncols = vecD.Nsize[1],vecD.Nsize[2]
#     @show vecD.Nsize
#     @show size(tt)
#     # number of rows of D^xr, D^yr, both row and col deficient
#     nsrcpts = count(onsrccols)
#     nhpts = count(onhpoints)

#     @show nsrcpts,nhpts
#     @show findall(onsrccols)
#     @show findall(onhpoints)

#     #################################
#     ## for grad w.r.t. source loc calculations
#     twoDttDH = zeros(eltype(vecD.v),nhpts,nhpts)
#     twoDttDS = zeros(eltype(vecD.v),nhpts,nsrcpts)
#     #################################

#     ## CSR matrix-vector product
#     for i=1:nrows

#         ## pre-compute Dx * tt 
#         tmp1 = 0.0
#         for l=vecD.iptr[i]:vecD.iptr[i+1]-1
#             j = vecD.j[l]
#             # dot product
#             tmp1 += vecD.v[l]*tt[j]
#         end
    
#         for l=vecD.iptr[i]:vecD.iptr[i+1]-1
#             j = vecD.j[l]  # column index

#             # +nsrcpts-> onhpoints runs on all columns, while rows are reduced...
#             if onhpoints[j] #&& onhpoints[i+nsrcpts] 
#                 ############################################################### 
#                 ##  ∂f_h/∂u_h = twoDttDH = 2 ( D_ij * tt_j ) * D_ih
#                 ###############################################################
#                 # perform the calculation only if we are not on a source point
#                 #   in order to remove the columns corresponding to the source
#                 #   points
#                 jh = count(onhpoints[1:j]) 
#                 ih = count(onhpoints[1:i+nsrcpts]) 
#                 twoDttDH[ih,jh] = 2.0 * tmp1 * vecD.v[l]

#                 #@show i,j,ih,jh
#             end

#             # +nsrcpts-> onhpoints runs on all columns, while rows are reduced...
#             if onsrccols[j] #&& onhpoints[i+nsrcpts]
#                 # Remark: onsrc in the fine grid
#                 ################################################################### 
#                 ##  ∂f_h/∂u_s = twoDttDS = 2 ( D_ij * tt_j ) * D_is
#                 ###################################################################
#                 jss = count(onsrccols[1:j])
#                 iss = count(onhpoints[1:i+nsrcpts])
#                 #twoDttDS[iss,jss] = 2.0 * tmp1 * vecD.v[l]
#                 @show i,j,iss,jss,onhpoints[i+nsrcpts]
#                 # println()
#                 # @show iss,jss,
#                 # println()
#             end

#         end # for l=vecD.iptr[i]:vecD.iptr[i+1]-1
#     end # i=1:nrows

#     println("twoDttDH")
#     display(twoDttDH)
#     println("twoDttDS")
#     display(twoDttDS)

#     return twoDttDH,twoDttDS
# end

###########################################################################

