

#######################################################
##     Misfit of gradient using adjoint              ## 
#######################################################

###############################################################################

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
- `smoothgrad`: smooth the gradient? true or false

# Returns
- `grad`: the gradient as a 2D array

"""
function gradttime2D(vel::Array{Float64,2}, grd::GridEik2D,coordsrc::Array{Float64,2},
                     coordrec::Vector{Array{Float64,2}},pickobs::Vector{Vector{Float64}},
                     stdobs::Vector{Vector{Float64}} ;
                     smoothgradsourceradius::Integer=3,smoothgrad::Bool=false,
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
                                                 smoothgradsourceradius,
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
        Threads.@threads for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            grad .+= calcgradsomesrc2D(vel,view(coordsrc,igrs,:),view(coordrec,igrs),
                                       grd,view(stdobs,igrs),view(pickobs,igrs),
                                       smoothgradsourceradius,
                                       extraparams )
        end
        


    elseif extraparams.parallelkind==:serial
        ##====================================
        ## Serial run
        ##====================
        grad[:,:] .= calcgradsomesrc2D(vel,coordsrc,coordrec,grd,stdobs,pickobs,
                                       smoothgradsourceradius,extraparams )

    end


    ## smooth gradient
    if smoothgrad
        l = 5  # 5 pixels kernel
        grad = smoothgradient(l,grad)
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
                           smoothgradsourceradius::Integer, extrapars::ExtraParams )
                           
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

        # projection operator P ordered according to FMM order
        P_fmmord1 = calcprojttfmmord(fmmvars.ttime,grd,adjvars.idxconv,coordrec[s])

        ###########################################
        # discrete adjoint formulation
        grad1 .+= discradjoint2D_FMM_SINGLESRC(adjvars,P_fmmord1,pickobs1[s],stdobs[s],vel)

        ###########################################
        ## smooth gradient around the source
        smoothgradaroundsrc2D!(grad1,xysrc[s,:],grd,radiuspx=smoothgradsourceradius)

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
                          codeDxy_orig::Array{Int64,2},allcoeff::CoeffDerivatives,ijpt::AbstractVector;
                          axis::Symbol, simtype::Symbol)
    
    # get the linear index in the original grid
    iptorig = idxconv.lfmm2grid[irow]
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
    ## they must to be zeroed every time!
    colinds = MVector(0,0,0)
    colvals = MVector(0.0,0.0,0.0)

    whX::Int=0
    whY::Int=0

    if axis==:X
        if codex==0
            # no derivatives have been used...
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
            # no derivatives have been used...
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
            ## the following is needed because in Julia the
            ##    row indices in every column NEED to be SORTED!
            if signcod<0.0
                colinds[p] = ifmmord
                colvals[p] = mycoeff
            else
                # reverse the order
                q = nnzcol-p+1
                colinds[q] = ifmmord
                colvals[q] = mycoeff
            end
            ##addentry!(D, irow, ifmmord, mycoeff )
        end
        ##########################################
        ## Add one entire row at a time!
        ##########################################
        addrowCSRmat!(D,irow,colinds,colvals,nnzcol)  


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
             ##
            ## the following is needed because in Julia the
            ##    row indices in every column NEED to be SORTED!
            if signcod<0.0
                colinds[p] = ifmmord
                colvals[p] = mycoeff
            else
                # reverse the order 
                q = nnzcol-p+1
                colinds[q] = ifmmord
                colvals[q] = mycoeff
            end
            ##addentry!(D, irow, ifmmord, mycoeff )
        end
        ##########################################
        ## Add one entire row at a time!
        ##########################################
        addrowCSRmat!(D,irow,colinds,colvals,nnzcol)  

    end

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
                                      vel2d::AbstractArray{Float64})
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #

   # @time begin
    
    idxconv = adjvars.idxconv
    tt = adjvars.fmmord.ttime
    vecDx = adjvars.fmmord.vecDx
    vecDy = adjvars.fmmord.vecDy
    Ni,Nj = vecDx.Nsize

    ################################
    ##  right hand side
    ################################
    rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)


    ################################
    ##   left hand side
    ################################

    # for i=1:size(tmplhs,1)
    #     h = nnz(tmplhs[i,:])
    #     println("$i  nnz: $h")
    # end

    ## compute the lhs terms
    calclhsterms!(vecDx,tt)
    calclhsterms!(vecDy,tt)


    # vecD = vecDy
    # for i=1:vecD.Nsize[1]
    #     aa = vecD.iptr[i]:vecD.iptr[i+1]-1
    #     @show i,vecD.j[aa],vecD.v[aa]
    # end

    #===========================================
    ###  !!! REMARK from SparseArrays:  !!! ####
    ============================================
    The row indices in every column NEED to be SORTED. If your 
    SparseMatrixCSC object contains unsorted row indices, one quick way 
    to sort them is by doing a double transpose.
    ============================================#

    ## We have a CSR matrix as vectors, we need a traspose of it,
    ##  so we construct directly a CSC by exchanging i and j indices
    Nxnnz = adjvars.fmmord.vecDx.Nnnz[]
    lhs1term = SparseMatrixCSC(Nj,Ni,vecDx.iptr,vecDx.j[1:Nxnnz],vecDx.v[1:Nxnnz])
    # see remark above
    lhs1term = copy(transpose(copy(transpose(lhs1term))))

    Nynnz = adjvars.fmmord.vecDy.Nnnz[]
    lhs2term = SparseMatrixCSC(Nj,Ni,vecDy.iptr,vecDy.j[1:Nynnz],vecDy.v[1:Nynnz])
    # see remark above
    lhs2term = copy(transpose(copy(transpose(lhs2term))))


    
    @show lhs1term.colptr-vecDx.iptr
    @show lhs1term.rowval-vecDx.j[1:Nxnnz]
    @show lhs1term.nzval-vecDx.v[1:Nxnnz]
    @show size(lhs1term.colptr),size(vecDx.iptr)
    @show size(lhs1term.rowval),size(vecDx.j[1:Nxnnz])
    @show size(lhs1term.nzval),size(vecDx.v)

    for i=1:length(lhs1term.rowval)
        a = lhs1term.rowval[i]
        b = vecDx.j[1:Nxnnz][i]
        if a!= b
            @show i,a,b
        end
    end
            

    # they are already transposed, so only add them
    tmplhs = lhs1term .+ lhs2term
    #end

    # @show nnz(lhs1term)

    # arr = tmplhs
    # for i=1:size(arr,2)
        
    #     tmp = Array(arr)[:,i]
    #     co = count(tmp.!=0.0)
    #     if co==0
    #         @show i,tmp
    #     else
    #         @show i,co
    #     end
    #     # h = nnz(lhs1term[i,:])
    #     # println("$i  nnz: $h")
    # end
    # @show rank(Array(tmplhs))

    # #@show display(copy(transpose(Array(tmplhs))))
    # # display(diag(Array(tmplhs)))
    # # @show prod(diag(Array(tmplhs)))

    # @show typeof(tmplhs)
    # display(tmplhs)


    @time begin
        
        ## make sure it's recognised as upper triangular...
        lhs = UpperTriangular(tmplhs)
    end
        # struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
        #     m::Int                  # Number of rows
        #     n::Int                  # Number of columns
        #     colptr::Vector{Ti}      # Column j is in colptr[j]:(colptr[j+1]-1)
        #     rowval::Vector{Ti}      # Row indices of stored values
        #     nzval::Vector{Tv}       # Stored values, typically nonzeros
        # end
        
        # ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
        # ##   the solver (\) does not use the correct sparse algo for matrix division
        # tmplhs = copy(transpose( (2.0.*Diagonal(Dx*tt)*Dx) .+ (2.0.*Diagonal(Dy*tt)*Dy) ))
        # lhs = UpperTriangular(tmplhs)

        ################################
        ##  solve the linear system
        ################################
        lambda_fmmord = lhs\rhs
    

    ##--------------------------------------
    # reorder lambda from fmmord to original grid!!
    N = length(lambda_fmmord)
    lambda = Vector{Float64}(undef,N)
    for p=1:N
        iorig = idxconv.lfmm2grid[p]
        lambda[iorig] = lambda_fmmord[p]
    end

    #gradvec = 2.0 .* transpose(lambda) * Diagonal(1.0./ (vec(vel2d).^2) )
    gradvec = 2.0 .* lambda ./ vec(vel2d).^3

    grad2d = reshape(gradvec,idxconv.nx,idxconv.ny)


    return grad2d
end

#######################################################################################

function calclhsterms!(vecD::VecSPDerivMat,tt::Vector{Float64})
    # Remark: equivalent to CSR format 

    # tmp1 = zeros(nrows)
    # ## CSR matrix-vector product
    # for i=1:nrows
    #     # pi=pointers to column indices
    #     for l=vecD.pi[i]:vecD.pi[i+1]-1
    #         j = vecD.j[l]
    #         # dot product
    #         tmp1[i] += vecD.v[l]*tt[j]
    #     end
    # end

    # number of columns
    nrows = vecD.Nsize[1]

    ## CSR matrix-vector product
    for i=1:nrows
        # pi=pointers to column indices
        tmp1 = 0.0
        for l=vecD.iptr[i]:vecD.iptr[i+1]-1
            j = vecD.j[l]
            # dot product
            tmp1 += vecD.v[l]*tt[j]
        end
        ## scale all rows by 2*tmp1
        for l=vecD.iptr[i]:vecD.iptr[i+1]-1
            vecD.v[l] = 2.0*tmp1*vecD.v[l]
        end
    end
    
    # ## CSR matrix-vector product
    # tmp1 = zeros(nrows)
    # for i=1:nrows
    #     # pi=pointers to column indices
    #     for l=vecD.iptr[i]:vecD.iptr[i+1]-1
    #         j = vecD.j[l]
    #         # dot product
    #         tmp1[i] += vecD.v[l]*tt[j]
    #     end
    # end
    
    # for i=1:nrows
    #     ## scale all rows by 2*tmp1
    #     for l=vecD.iptr[i]:vecD.iptr[i+1]-1
    #         vecD.v[l] = 2.0*tmp1[i]*vecD.v[l]
    #     end
    # end
    
    return 
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
