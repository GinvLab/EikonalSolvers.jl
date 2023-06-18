

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
                     stdobs::Vector{Vector{Float64}} ;
                     extraparams::Union{ExtraParams,Nothing}=nothing)

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
    grad = zeros(n1,n2)
    
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
        @sync begin 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async grad .+= remotecall_fetch(calcgradsomesrc2D,wks[s],vel,
                                                 coordsrc[igrs,:],coordrec[igrs],
                                                 grd,stdobs[igrs],pickobs[igrs],
                                                 extraparams )
            end
        end


    elseif extraparams.parallelkind==:sharedmem
        ##====================================
        ## Shared memory        
        ##====================
        nth = Threads.nthreads()
        grpsrc = distribsrcs(nsrc,nth)
        nchu = size(grpsrc,1)            
        ##  do the calculations
        grads = Vector{Array{Float64,2}}(undef,nchu)
        Threads.@threads for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            grads[s] = calcgradsomesrc2D(vel,view(coordsrc,igrs,:),view(coordrec,igrs),
                                         grd,view(stdobs,igrs),view(pickobs,igrs),
                                         extraparams )
        end
        grad = sum(grads)


    elseif extraparams.parallelkind==:serial
        ##====================================
        ## Serial run
        ##====================
        grad[:,:] .= calcgradsomesrc2D(vel,coordsrc,coordrec,grd,stdobs,
                                       pickobs,extraparams )

    end


    ## smooth gradient
    if extraparams.smoothgradkern>0
        grad = smoothgradient(extraparams.smoothgradkern,grad)
    end

    if extraparams.manualGCtrigger
        # trigger garbage collector
        #println("Triggering GC")
        GC.gc()
    end
    return grad
end

###############################################################################

"""
$(TYPEDSIGNATURES)


Calculate the gradient for some requested sources 
"""
function calcgradsomesrc2D(vel::Array{Float64,2},xysrc::AbstractArray{Float64,2},
                           coordrec::AbstractVector{Array{Float64,2}},grd::GridEik2D,
                           stdobs::AbstractVector{Vector{Float64}},pickobs1::AbstractVector{Vector{Float64}},
                           extrapars::ExtraParams )
                           
    nx,ny=size(vel)
    nsrc = size(xysrc,1)
    grad1 = zeros(nx,ny)
  
    ## pre-allocate ttime and status arrays plus the binary heap
    fmmvars = FMMvars2D(nx,ny,refinearoundsrc=extrapars.refinearoundsrc,
                        allowfixsqarg=extrapars.allowfixsqarg)
    
    ## pre-allocate discrete adjoint variables
    adjvars = AdjointVars2D(nx,ny)

    
    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        ###########################################
        ## calc ttime, etc.
        ttFMM_hiord!(fmmvars,vel,view(xysrc,s,:),grd,adjvars)

        ###########################################
        # projection operator P ordered according to FMM order
        P_fmmord1 = calcprojttfmmord(fmmvars.ttime,grd,adjvars.idxconv,coordrec[s])

        ###########################################
        # discrete adjoint formulation for gradient(s) with respect to velocity and/or source position
        tmpgrad,∂χ∂xy_src = discradjoint2D_FMM_SINGLESRC(adjvars,P_fmmord1,pickobs1[s],
                                                         stdobs[s],
                                                         vel,xysrc[s,:],grd)
        grad1 .+= tmpgrad

        ###########################################
        ## smooth gradient (with respect to velocity) around the source
        smoothgradaroundsrc2D!(grad1,view(xysrc,s,:),grd,radiuspx=extrapars.radiussmoothgradsrc)

    end
 
    if extrapars.manualGCtrigger
        # trigger garbage collector
        #println("Triggering GC")
        GC.gc()
    end
    
    return grad1
end 

##############################################################################

"""
$(TYPEDSIGNATURES)

 Set the coefficients (elements) of the derivative matrices in the x and y directions.
    """
function setcoeffderiv2D!(D::VecSPDerivMat,irow::Integer,idxconv::MapOrderGridFMM2D,
                          codeDxy_orig::Array{Int64,2},allcoeff::CoeffDerivatives,ijpt::AbstractVector,
                          colinds::AbstractVector{Int64},colvals::AbstractVector{Float64},idxperm::AbstractVector{Int64},
                          nptsfixedtt::Integer;
                          axis::Symbol, simtype::Symbol)
    
    # get the linear index in the original grid
    iptorig = idxconv.lfmm2grid[irow+nptsfixedtt]
    # get the (i,j) indices in the original grid
    nx = idxconv.nx
    lin2cart2D!(iptorig,nx,ijpt)
    igrid = ijpt[1]
    jgrid = ijpt[2]
    #@show irow,iptorig,ijpt,nx
    # get the codes of the derivatives (+1,-2,...)
    # extract codes for X and Y
    codex,codey = codeDxy_orig[iptorig,1],codeDxy_orig[iptorig,2]

    ## row stuff
    ## they must to be zeroed/filled every time!
    tmi = typemax(eltype(colinds))
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
        error("if axis==:X ...")
    end
       
    # we skip points on source, so there must always be derivatives
    if code==[0,0]
        error("code==[0,0]")
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
    ## the following is needed because in Julia the
    ##    row indices in every column NEED to be SORTED!
    sortperm!(idxperm,colinds)
    colinds[:] .= colinds[idxperm]
    colvals[:] .= colvals[idxperm]
    addrowCSRmat!(D,irow,colinds,colvals,nnzcol)

    # if irow<=5
    #     @show irow,colinds[1:nnzcol],colvals[1:nnzcol]
    # end
    return
end

##############################################################################

"""
$(TYPEDSIGNATURES)

 Projection operator P containing interpolation coefficients, ordered according to FMM order.
"""
function calcprojttfmmord(ttime::AbstractArray{Float64},grd::GridEik,idxconv::MapOrderGridFMM,
                          coordrec::AbstractArray{Float64})

    if typeof(idxconv)==MapOrderGridFMM2D
        simdim = :sim2D
    elseif typeof(idxconv)==MapOrderGridFMM3D
        simdim = :sim3D
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
            coeff,ijcoe = bilinear_interp( ttime,grd,view(coordrec,r,:),
                                           return_coeffonly=true )

        elseif simdim==:sim3D 
            coeff,ijcoe = trilinear_interp( ttime,grd,view(coordrec,r,:),
                                            return_coeffonly=true )

        end

        #@assert size(ijcoe,1)==4
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
            jfmmord = idxconv.lgrid2fmm[iorig]  # jfmmord = findfirst(idxconv.lfmm2grid.==iorig)

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

"""
$(TYPEDSIGNATURES)

 Solve the discrete adjoint equations and return the gradient of the misfit.
"""
function discradjoint2D_FMM_SINGLESRC(adjvars::AdjointVars2D,P::AbstractArray{Float64},
                                      pickobs::AbstractVector{Float64},
                                      stdobs::AbstractVector{Float64},
                                      vel2d::AbstractArray{Float64},
                                      xysrc::AbstractArray{Float64},
                                      grd::GridEik2D)
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #
    
    idxconv = adjvars.idxconv
    tt_fmmord = adjvars.fmmord.ttime

    # Derivative (along x) matrix, row-deficien
    vecDx = adjvars.fmmord.vecDx
    # Derivative (along y) matrix, row-deficien
    vecDy = adjvars.fmmord.vecDy

    # # sizes of D^x, D^y
    # Ni,Nj = vecDx.Nsize
    # # number of rows of D^xr, D^yr, both row and col deficient
    # nsrcpts = count(adjvars.fmmord.onsrcrows)
    # Njr = Nj - nsrcpts
    
    # # Derivative (along x) matrix, row- *and* column-deficient
    # ncolidx = length(adjvars.fmmord.vecDx.j) - nsrcpts
    # vecDxr = VecSPDerivMat( iptr=zeros(Int64,Ni+1), j=zeros(Int64,ncolidx),
    #                         v=zeros(ncolidx), Nsize=[Ni,Njr] )

    # for i=1:Ni
    #     l2 = vecDx.iptr[i]
    #     for l=vecDx.iptr[i]:vecD.iptr[i+1]-1
    #         j = vecD.j[l]    # column
    #         # if we are not on a source point then copy current element
    #         if sourceptsindex[j]==false
    #             # add 1 to row pointer i+1
    #             vecDxr.iptr[i+1] += 1
    #             l2 += 1
    #             vecDxr.j[l2] = j
    #             vecDxr.v[l2] = vecDx.v[l]
                
    #         end
    #     end
    # end
    

    # # Derivative (along x) matrix, row- *and* column-deficient
    # ncolidy = length(adjvars.fmmord.vecDy.j) - nsrcpts
    # vecDyr = VecSPDerivMat( iptr=zeros(Int64,nxy+1), j=zeros(Int64,ncolidy),
    #                         v=zeros(ncolidy), Nsize=[Ni,Njr] )

    
    # ################################
    # ##  Derivative with respect to the traveltime
    # ##     at the "onsrc" points
    # ## precompute some terms
    # ################################
    # sourcerows = findall(adjvars.fmmord.sourcerows) # Bool array to indices
    # twodiagDuDh_x = calctwodiagDuDh(vecDx,tt,sourcerows)
    # twodiagDuDh_y = calctwodiagDuDh(vecDy,tt,sourcerows)

   
    ###########################################
    ##  right hand side adj eq. == -adj_src^T
    ###########################################
    #rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)

    fact2 = ((P*tt_fmmord).-pickobs)./stdobs.^2
    # @show typeof(fact2)
    # @show typeof(P)
    # bool vector with false for src columns
    rq = .!adjvars.fmmord.onsrcrows
    PD = P[:,rq] # remove columns
    # @show size(PD),size(fact2)
    rhs = - transpose(PD) * fact2


    ################################
    ##   left hand side adj eq.
    ################################
    ## compute the lhs terms
    # println("\n\n")
    # @show "1st term"
    vecDxDR,twoDttDSx = calcadjlhsterms(vecDx,tt_fmmord,adjvars.fmmord.onsrcrows)
    # println("\n\n")
    # @show "2nd term"
    vecDyDR,twoDttDSy = calcadjlhsterms(vecDy,tt_fmmord,adjvars.fmmord.onsrcrows) 


    # function showD(D)
    #     for i=1:D.Nsize[1]
    #         for l=D.iptr[i]:D.iptr[i+1]-1
    #             j = D.j[l]
    #             v = D.v[l]
    #             #@show l,i,j
    #             if i<=2
    #                 @show i,j,v
    #             end
    #             # if i==0 || j==0
    #             #     println(">>>>>>>> zero: i=$i, j=$j <<<<<<<<<")
    #             #     @show D.iptr[i:i+10],D.j[l:l+2]
    #             # end
    #         end
    #     end
    #     return
    # end

    # @show "vecDxDR"
    # showD(vecDxDR)
    # @show "vecDyDR"
    # showD(vecDyDR)



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
    Nxnnz = vecDxDR.Nnnz[]
    Nix = vecDxDR.Nsize[1]
    Njx = vecDxDR.Nsize[2]
    lhs1term = SparseMatrixCSC(Njx,Nix,vecDxDR.iptr[1:Nix+1],
                               vecDxDR.j[1:Nxnnz],vecDxDR.v[1:Nxnnz])
 
    ## We have a CSR matrix as vectors, we need a *transpose* of it,
    ##  so we construct directly a CSC by *exchanging* i and j indices
    Nynnz = vecDyDR.Nnnz[]
    Niy = vecDyDR.Nsize[1]
    Njy = vecDyDR.Nsize[2]
    lhs2term = SparseMatrixCSC(Njy,Niy,vecDyDR.iptr[1:Niy+1],
                               vecDyDR.j[1:Nynnz],vecDyDR.v[1:Nynnz])


    # They are already transposed, so only add them
    tmplhs = lhs1term .+ lhs2term

    # display(tmplhs)
    # @show diag(tmplhs)
    # display( (Array(tmplhs[1:10,1:10])) )
    # @show size(tmplhs)

    ## make sure it's recognised as upper triangular...
    lhs = UpperTriangular(tmplhs)

    ## OLD stuff...
    # ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
    # ##   the solver (\) does not use the correct sparse algo for matrix division
    # tmplhs = copy(transpose( (2.0.*Diagonal(Dx*tt)*Dx) .+ (2.0.*Diagonal(Dy*tt)*Dy) ))
    # lhs = UpperTriangular(tmplhs)

    ################################
    ##  solve the linear system
    ################################
    lambda_fmmord = lhs\rhs

    ################################
    ##  Derivative with respect to the traveltime
    ##     at the "onsrc" points
    ##  compute gradient
    ################################
    ∂χ∂xy_src = ∂misfit∂initsrcpos(twoDttDSx,twoDttDSy,tt_fmmord,
                                   adjvars.idxconv,lambda_fmmord,vel2d,
                                   adjvars.fmmord.onsrcrows,xysrc,grd)
    
    #@timeit to "reorder lambda" begin
    ##--------------------------------------
    # reorder lambda from fmmord to original grid!!
    N = length(lambda_fmmord)
    nptsonsrc = count(adjvars.fmmord.onsrcrows)

    # lambda must be zeroed here!
    lambda = zeros(eltype(lambda_fmmord),N+nptsonsrc)
    Nall = length(idxconv.lfmm2grid)
    for p=1:N
        # The following two lines is a complex way to map the
        #  "full" indices to the "reduced" set of indices
        curnptsonsrc_fmmord = count(adjvars.fmmord.onsrcrows[1:p])
        iorig = idxconv.lfmm2grid[p+nptsonsrc]
        # map lambda in fmmord into lambda in the original grid
        lambda[iorig] = lambda_fmmord[p]
    end

    # lambda = Vector{Float64}(undef,N)
    # Nall = length(idxconv.lfmm2grid)
    # onsrcrows_orig = zeros(Bool,Nall)
    # @assert length(adjvars.fmmord.onsrcrows)==Nall
    # # The following stuff is a complex way to map the
    # #  "full" indices to the "reduced" set of indices
    # onsrcrows_orig = zeros(Bool,Nall)
    # for q=1:Nall
    #     if adjvars.fmmord.onsrcrows[q]
    #         iorig = idxconv.lfmm2grid[q]
    #         onsrcrows_orig[iorig] = true
    #     end
    # end
    # for p=1:N
    #     curnptsonsrc_fmmord = count(adjvars.fmmord.onsrcrows[1:p])
    #     iorig = idxconv.lfmm2grid[p+nptsonsrc] 
    #     curnptsonsrc_orig = count(onsrcrows_orig[1:iorig])
    #     lambda[iorig-curnptsonsrc_orig] = lambda_fmmord[p]
    # end

    #gradvec = 2.0 .* transpose(lambda) * Diagonal(1.0./ (vec(vel2d).^2) )
    gradvec = 2.0 .* lambda ./ vec(vel2d).^3

    grad2d = reshape(gradvec,idxconv.nx,idxconv.ny)
    #end

    #show(to)
    #println()
    return grad2d,∂χ∂xy_src
end

#######################################################################################

"""
$(TYPEDSIGNATURES)

Calculates the following maintaining the sparse structure:
     2 * diag(D * tt) * Dr
"""
function calcadjlhsterms(vecD::VecSPDerivMat,tt::Vector{Float64},
                         onsrcrows::Vector{Bool})
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

    # # number of rows
    # nrows = vecD.Nsize[1]
    # ## CSR matrix-vector product
    # for i=1:nrows
    #     # pi=pointers to column indices
    #     tmp1 = 0.0
    #     for l=vecD.iptr[i]:vecD.iptr[i+1]-1
    #         j = vecD.j[l]j
    #         # dot product
    #         tmp1 += vecD.v[l]*tt[j]
    #     end
    #     ## scale all rows by 2*tmp1
    #     for l=vecD.iptr[i]:vecD.iptr[i+1]-1
    #         vecD.v[l] = 2.0*tmp1*vecD.v[l]
    #     end
    # end

    # size of D^x or D^y
    nrows,ncols = vecD.Nsize[1],vecD.Nsize[2]
    # number of rows of D^xr, D^yr, both row and col deficient
    nsrcpts = count(onsrcrows)
    ncolsR = ncols - nsrcpts

    # Derivative (along x) matrix, row- *and* column-deficient
    ncolidx = vecD.Nnnz[] - nsrcpts
    two_Dtt_DR = VecSPDerivMat( iptr=zeros(Int64,nrows+1), j=zeros(Int64,ncolidx),
                              v=zeros(ncolidx), Nsize=[nrows,ncolsR] )
   
    # mapping of column number from full to column reduced
    idxrowred = zeros(Int64,length(onsrcrows))
    q = 0
    for p=2:length(onsrcrows)
        if onsrcrows[p]
            idxrowred[p] = q
        else
            q+=1
            idxrowred[p] = q
        end
    end
    

    #################################
    ## for grad w.r.t. source loc calculations
    twoDttDS = zeros(eltype(vecD.v),nrows,nsrcpts)
    #################################

    ## CSR matrix-vector product
    for i=1:nrows

        ## pre-compute Dx * tt 
        tmp1 = 0.0
        for l=vecD.iptr[i]:vecD.iptr[i+1]-1
            j = vecD.j[l]
            # dot product
            tmp1 += vecD.v[l]*tt[j]
        end

        ## init the next row pointer
        two_Dtt_DR.iptr[i+1] = two_Dtt_DR.iptr[i]
       
        ## scale all rows by 2*tmp1 excluding certain columns
        l2 = two_Dtt_DR.iptr[i]-1 # l2 runs on the column-reduced matrix DR
        for l=vecD.iptr[i]:vecD.iptr[i+1]-1
            j = vecD.j[l]    # column index

            # if i<=6
            #     @show "D",i,j
            # end

            if onsrcrows[j]==false
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
                two_Dtt_DR.v[l2] = 2.0*tmp1*vecD.v[l]
                # update pointers
                two_Dtt_DR.Nnnz[] += 1
                two_Dtt_DR.lastrowupdated[] = i # this must stay here (this if and for loop)!

                # if i<=10
                #     @show i,ii,idxrowred[j],two_Dtt_DR.j[l2]
                # end

            elseif onsrcrows[j]
                ################################################################### 
                ##  Computation of 2*D*tt*DS for the grad w.r.t. source position
                ###################################################################
                js = count(onsrcrows[1:j])
                twoDttDS[i,js] = 2.0 * tmp1 * vecD.v[l]
                    
            end
        end
    end

    return two_Dtt_DR,twoDttDS
end

#######################################################################################

"""
$(TYPEDSIGNATURES)

 Attempt to reconstruct how derivatives have been calculated in the area of 
  the refinement of the source. Updates the codes for derivatives used to construct
   Dx and Dy.
"""
function derivaroundsrcfmm2D!(lseq::Integer,idxconv::MapOrderGridFMM2D,codeD::MVector)
    # lipt = index in fmmord in sequential order (1,2,3,4,...)
    
    nx = idxconv.nx
    ny = idxconv.ny

    ijpt = MVector(0,0)
    idxpt = idxconv.lfmm2grid[lseq]
    lin2cart2D!(idxpt,nx,ijpt)
    ipt,jpt = ijpt[1],ijpt[2]
    
    codeD[:] .= 0
    for axis=1:2

        chosenidx = lseq
        for dir=1:2

            ## map the 4 cases to an integer as in linear indexing...
            lax = dir + 2*(axis-1)
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
            
            ##=========================
            ## first order
            i = ipt+ish
            j = jpt+jsh

            ## check if on boundaries
            isonb1st,isonb2nd = isonbord(i,j,nx,ny)

            if !isonb1st
                ## calculate the index of the neighbor in the fmmord
                l = cart2lin2D(i,j,nx)
                #idxne1 = findfirst(idxconv.lfmm2grid.==l)
                idxne1 = idxconv.lgrid2fmm[l]
                
                #@show lseq,l,idxne1,chosenidx
            
                if idxne1!=nothing && idxne1<chosenidx                   

                    # to make sure we chose correctly the direction
                    chosenidx = idxne1
                    # save derivative choices [first order]
                    axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

                    if !isonb2nd
                        ##=========================
                        ## second order
                        i = ipt + 2*ish
                        j = jpt + 2*jsh
                        
                        ## calculate the index of the neighbor in the fmmord
                        l = cart2lin2D(i,j,nx)
                        idxne2 = idxconv.lgrid2fmm[l]
                        
                        ##===========================================================
                        ## WARNING! The traveltime for the second order point must
                        ##   be smaller than the one for the first order for selecting
                        ##   second order. Otherwise first order only.
                        ##  Therefore we compare idxne2<idxne1 instead of idxne2<idxpt
                        ##===========================================================
                        if idxne2<idxne1
                            # save derivative choices [second order]
                            axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)
                        end

                    end
                end
            end            
        end
    end

    return 
end

#######################################################################################

"""
$(TYPEDSIGNATURES)

Calculates the derivative of the misfit function with respect to the traveltime at the initial points around the source ("onsrc").
"""
function ∂misfit∂initsrcpos(twoDttDSx,twoDttDSy,tt,
                            idxconv,
                            lambda::AbstractArray{Float64},
                            vel2d,sourcerows,xysrc,grd)
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
        # # element-wise product
        # ∂χ∂t_src[p] = lambda .* tmp1[:,p]

        idx = [(tmp1[:,p].!=0.0 .&& lambda.!=0.0)]
        # println()
        # @show tmp1[idx...,p]
        # @show lambda[idx...]
        # @show sum(tmp1[idx...,p] .* lambda[idx...])
        # @show ∂χ∂t_src[p]
    end

    ###################################################################
    ## Derivative of the misfit w.r.t. the source position (chain rule)
    ###################################################################
    velpts = zeros(npts)
    ijpt = zeros(Int64,2)
    xypt = zeros(npts,2)
    for p=1:npts 
        ifmmord = count(sourcerows[1:p])
        iorig = idxconv.lfmm2grid[ifmmord]
        velpts[p] = vel2d[iorig]

        ## CONVERT linear to cartesian 2D to get point position
        lin2cart2D!(iorig,nx,ijpt)
        xypt[p,:] .= (grd.x[ijpt[1]], grd.y[ijpt[2]])
    end

    # compute the derivative 
    ∂χ∂xy_src = derivttsrcposition2D(∂χ∂t_src,xypt,xysrc,velpts)
    
    # @show extrema(∂χ∂t_src)
    display( ∂χ∂xy_src )

    return ∂χ∂xy_src
end

###########################################################################

function derivttsrcposition2D(∂χ∂t_src,xypt,xysrc,velpts)

    npts = length(∂χ∂t_src)

    deriv = zeros(npts,2)
    for p=1:npts
        ## Apply the chain rule:
        ##   ∂χ/∂t_src * ∂t_src/∂x, etc.
        derpos = partderivttsrcpos2D(xypt[p,:],xysrc,velpts[p])
        #deriv[p,:] .= ∂χ∂t_src[p] .* derpos
        deriv[p,:] .= derpos

        #println()
        #@show ∂χ∂t_src[p]
        #@show derpos
    end

    dχdx_src = dot( ∂χ∂t_src, deriv[:,1] )
    dχdy_src = dot( ∂χ∂t_src, deriv[:,2] )

    return dχdx_src,dχdy_src
end

###########################################################################

function partderivttsrcpos2D(xypt::Vector,xysrc::Vector,vel::Real)

    xpt = xypt[1]
    ypt = xypt[2]
    xsrc = xysrc[1]
    ysrc = xysrc[2]

    denom = vel * sqrt((xpt - xsrc)^2 + (ypt - ysrc)^2)
    deriv_x = -(xpt - xsrc) / denom
    deriv_y = -(ypt - ysrc) / denom

    # println()
    # @show xpt,ypt
    # @show xsrc,ysrc
    # @show denom
    # @show deriv_x,deriv_y

    return deriv_x,deriv_y
end

###########################################################################
