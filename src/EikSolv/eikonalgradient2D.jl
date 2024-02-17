

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
function setcoeffderiv2D!(D::VecSPDerivMat,status::Array,irow::Integer,idxconv::MapOrderGridFMM2D,
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

    ##=======================
    # ## OLD version
    # # get the codes of the derivatives (+1,-2,...)
    # # extract codes for X and Y
    # codex,codey = codeDxy_orig[iptorig,1],codeDxy_orig[iptorig,2]

    ##=======================
    ## NEW version
    ## If and only if the point has been accepted, then get the codes
    if status[igrid,jgrid]==2
        # get the codes of the derivatives (+1,-2,...)
        # extract codes for X and Y
        codex,codey = codeDxy_orig[iptorig,1],codeDxy_orig[iptorig,2]
    else
        codex,codey = 0,0
    end
    
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
    elseif abscode==2
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

# println("\nlhs 1 coarse grid")
# display(lhsterm1)
# println("\nlhs 2 coarse grid")
# display(lhsterm2)

# println("lhs coarse grid")
# display(tmplhs)    

   
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
            # #error("Gradient with respect to velocity with refinement of the grid currently missing a piece... Aborting.")
            
            ## Solve the adjoint equation in the fine grid
            ##   dus_dva must be in FMM order to agree with ∂fi_∂us
            dus_dva = solveadjointfinegrid!(fmmvars_fine,adjvars_fine,
                                            grd_fine,srcrefvars,
                                            fmmvars.srcboxpar.ijsrc)
        end


        ##===============================================
        # Gradient with respect to velocity, term T2Va
        ##===============================================
        # deriv. of the implicit forw. mod. w.r.t. u_s
        #   ordered according to FMM order
        ∂fi_∂us = twoDttDSx .+ twoDttDSy
        ## FMM ordering for ∂fi_∂us!!!
        tmpdfidva = ∂fi_∂us * dus_dva        
        ## FMM ordering for lambda_fmmord!!!
        T2Va = transpose(lambda_fmmord) * tmpdfidva

        ## (i,j) indices of the source points
        ijsrcreg = fmmvars.srcboxpar.ijsrc

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

    end
    
    return 
end

##############################################################################

function solveadjointfinegrid!(fmmvars,adjvars,grd,srcrefvars,ijsrc_coarse)

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
    #@show Nhpts

@show grd.nx,grd.ny,grd.nx*grd.ny
@show Naccpts,Nhpts,count(onsrccols)
#@show onHpts


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
        H_v[q] = 1.0
        q+=1
    end
    H = sparse(H_i,H_j,H_v,Nhpts,Naccpts)

# @show Nhpts
# @show size(H)
#     display(H)
# #    display(Array(H))
# @show length(H.nzval)
    
    ###########################################
    # Derivative (along x) matrix, row-deficient
    vecDx = adjvars.fmmord.vecDx
    ###########################################
    # Derivative (along y) matrix, row-deficient
    vecDy = adjvars.fmmord.vecDy

#@show vecDx.Nsize,vecDy.Nsize    

    #######################################################
    ##  right hand side adj eq. == -(adjoint source)^T
    #######################################################
    # @show size(H)
    # @show size(.!adjvars.fmmord.onsrccols[1:Naccpts])
    # bool vector with false for src columns
    #    only part of the rectangular fine grid has been used, so [1:Naccpts]
    rq = .!onsrccols
    HD = H[:,rq] # remove columns
    rhs = - transpose(HD)


    ################################
    ##   left hand side adj eq.
    ################################
    ## compute the lhs terms
    twoDttDRx,twoDttDSx = calcadjlhsterms(vecDx,tt_fmmord,onsrccols)
    twoDttDRy,twoDttDSy = calcadjlhsterms(vecDy,tt_fmmord,onsrccols) 

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
    

    ############################################################
    ##  solve the linear system to get the adjoint variable
    ############################################################
    # Remark: "lambda_fmmord" excludes the points on the source (in the fine grid)
    #  lambda_fmmord is in FMM order
    lambda_fmmord = lhs\rhs

    Nptsoffsrc = Naccpts-nptsonsrc
    @assert size(lambda_fmmord,1)==Nptsoffsrc
    @assert size(lambda_fmmord,2)==Nhpts


    @show size(lambda_fmmord)
    @show extrema(lambda_fmmord)
    #@show Array(lambda_fmmord)
    # println("\n lambda_fmmord:")
    # display(lambda_fmmord)
    for h=1:Nhpts
        nnz_lambda_h = count(abs.(lambda_fmmord[:,h]).>0.0)
        @show h,nnz_lambda_h
    end

    ##========================================
    ##  Compute the T2Vd term
    ##========================================
    #T2Vd = zeros(N,Nhpts)
    ## The gradient in the fine grid is not a rectangular grid but it
    ##    depends on when the FMM has stopped (so a vector with 'Naccpts'
    ##    entries, not a 2D array)
    #duh_dwq = zeros(Nhpts,grd.nx*grd.ny)
    duh_dwq = zeros(Nhpts,Naccpts)
    @show size(duh_dwq),prod(size(duh_dwq))

    # To reorder lambda from fmmord to original grid...
    idxconv = adjvars.idxconv
    ijpt = MVector(0,0)
    for p=1:Nptsoffsrc
        # The following two lines (p+nptsonsrc) is a way to map the
        #  "full" indices to the "reduced" set of indices (no source region)
        iorig = idxconv.lfmm2grid[p+nptsonsrc] # skip pts on src
        # # get the i,j indices for vel and grad
        lin2cart2D!(iorig,idxconv.nx,ijpt)
        i,j = ijpt
        p2 = p+nptsonsrc
        for h=1:Nhpts
            # get the contribution of T2Vd to the gradient
            duh_dwq[h,p2] = 2.0 * lambda_fmmord[p,h] / srcrefvars.vel2d_fine[i,j]^3
        end
    end

    println("\n duh_dwq [T2Vd] ")
    display(duh_dwq)
    @show count(abs.(duh_dwq).>0.0)


    #@show N,Nhpts
    #@show size(T2Vd)
    # println("write duh_dwq to h5")
    # h5open("duh_dwq.h5","w") do fl
    #     fl["duh_dwq"] = duh_dwq
    # end

    ## source box parameters
    velcorn  = fmmvars.srcboxpar.velcorn
    distcorn = fmmvars.srcboxpar.distcorn
    ijsrcreg = fmmvars.srcboxpar.ijsrc

    ## derivative of \tau_s w.r.t. w_a
    dtaus_dwa = - diagm(distcorn) .* (1.0./velcorn.^2) # Hadamard product!

    println("dtaus_dwa")
    display(dtaus_dwa)
    #display(diagm(distcorn))


    ##========================================
    ## Compute the T2Vc term 
    ##========================================
    # deriv. of the implicit forw. mod. w.r.t. u_s, FMM ordering
    ∂gi_∂taus = twoDttDSx .+ twoDttDSy
    ## FMM ordering for ∂gi_∂taus!!!
    tmpdgidwa = ∂gi_∂taus * dtaus_dwa  
    ## FMM ordering for lambda_fmmord!!!
    T2Vc = transpose(lambda_fmmord) * tmpdgidwa

    println("\n ∂gi_∂taus ")
    display(∂gi_∂taus)    
    println("\n tmpdgidwa ")
    display(tmpdgidwa)


    @show size(T2Vc)
    ## add contribution to the gradient in the fine grid
    for p=1:nptsonsrc #size(T2Vc,2)
        for h=1:Nhpts #size(T2Vc,1)
            duh_dwq[h,p] += T2Vc[h,p]
        end
    end

    println("\n T2Vc ")
    display(T2Vc)
    @show count(abs.(T2Vc).>0.0)

    println("\n duh_dwq [T2Vc] ")
    display(duh_dwq)
    @show count(abs.(duh_dwq).>0.0)
    

    ##========================================
    ## Compute the T1Vc term 
    ##========================================
    H_Areg = H[:,onsrccols]
    T1Vc = H_Areg * dtaus_dwa
    @show size(T1Vc)
    for p=1:nptsonsrc
        for h=1:Nhpts
            duh_dwq[h,p] += T1Vc[h,p]
        end
    end
    println("\n T1Vc ")
    display(T1Vc)

    println("\n duh_dwq [T1Vc] ")
    display(duh_dwq)
    @show count(abs.(duh_dwq).>0.0)
    
    ##========================================
    ## Compute dwq/dwa
    ##========================================
    # filter grid point which have been actually used 
    #dwq_dva = srcrefvars.nearneigh_oper#[ , ]
    #@show size(srcrefvars.nearneigh_oper)
    dwq_dva = zeros(Nhpts,Nhpts)

    nx_window_coarse = srcrefvars.nxny_window_coarse[1]
    # @show nptsonsrc
    # @show size(idxconv.lfmm2grid)
    # @show length(idxonHpts)
    # @show length(onHpts)
    # for p=1:Naccpts
    #     iorig = idxconv.lfmm2grid[p]
    #     for (a,a_fine) in enumerate(idxonHpts)
    #         a_coarse = srcrefvars.nearneigh_idxcoarse[a_fine]
    #         dwq_dva[p,a] = srcrefvars.nearneigh_oper[iorig,a_coarse]
    #     end
    # end

    # @show size(srcrefvars.nearneigh_oper)
    
    # lsiorig = Int64[]
    # for p=1:size(ijsrc_coarse,1)
    #     i,j = ijsrc_coarse[p,:]
    #     ii = i - srcrefvars.ijcoarse[1]+1
    #     jj = j - srcrefvars.ijcoarse[2]+1
    #     iorig = cart2lin2D(ii,jj,nx_window_coarse)
    #     push!(lsiorig,iorig)
    # end
    # @show lsiorig
    dwq_dva = srcrefvars.nearneigh_oper[1:Nhpts,1:Nhpts] 


                                        
    # println("nneigh:")
    # display(nneigh)
    # @show extrema(nneigh)

    #println("dwq_dva:")
    #display(dwq_dva)
    @show size(dwq_dva)
    @show extrema(dwq_dva)
    
    ##========================================
    ## duh_dva
    ##========================================
    duh_dva = duh_dwq * dwq_dva

    @show size( duh_dwq),size( dwq_dva)
    @show size(duh_dva)
    println("duh_dva:")
    display(duh_dva)
    @show extrema(duh_dva)
    @show count(abs.(duh_dva).>0.0)

    return duh_dva
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



