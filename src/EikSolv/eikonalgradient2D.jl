

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
                     smoothgradsourceradius::Integer=3,smoothgrad::Bool=false )

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
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(n1,n2,nchu)
    ## do the calculations
    @sync begin 
        for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @async tmpgrad[:,:,s] = remotecall_fetch(calcgradsomesrc2D,wks[s],vel,
                                                     coordsrc[igrs,:],coordrec[igrs],
                                                     grd,stdobs[igrs],pickobs[igrs],
                                                     smoothgradsourceradius )
        end
    end
    grad = dropdims(sum(tmpgrad,dims=3),dims=3)
    #@assert !any(isnan.(grad))

    ## smooth gradient
    if smoothgrad
        l = 5  # 5 pixels kernel
        grad = smoothgradient(l,grad)
    end
 
    return grad
end

###############################################################################

"""
$(TYPEDSIGNATURES)


Calculate the gradient for some requested sources 
"""
function calcgradsomesrc2D(vel::Array{Float64,2},xysrc::Array{Float64,2},
                           coordrec::Vector{Array{Float64,2}},grd::GridEik2D,
                           stdobs::Vector{Vector{Float64}},pickobs1::Vector{Vector{Float64}},
                           smoothgradsourceradius::Integer )
                           
    nx,ny=size(vel)
    nsrc = size(xysrc,1)
    grad1 = zeros(nx,ny)
  
    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        ###########################################
        ## calc ttime, etc.
        ttgrdonesrc,idxconv,tt_fmmord1,Dx_fmmord1,Dy_fmmord1 = ttFMM_hiord(vel,xysrc[s,:],
                                                                           grd,dodiscradj=true)

        # projection operator P ordered according to FMM order
        P_fmmord1 = calcprojttfmmord(ttgrdonesrc,grd,idxconv,coordrec[s])

        ###########################################
        # discrete adjoint formulation
        grad1 .+= discradjoint2D_FMM_SINGLESRC(idxconv,tt_fmmord1,Dx_fmmord1,Dy_fmmord1,
                                                 P_fmmord1,pickobs1[s],stdobs[s],vel)

        ###########################################
        ## smooth gradient around the source
        smoothgradaroundsrc2D!(grad1,xysrc[s,:],grd,radiuspx=smoothgradsourceradius)

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

    whX::Int=0
    whY::Int=0

    if axis==:X
        if codex==0
            # no derivatives have been used...
            return
        else
            code = codex
            whX=1
            whY=0
        end

    elseif axis==:Y
        if codey==0
            # no derivatives have been used...
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
    signcod = sign(code)
    ## select first or second order coefficients
    if abscode==1
        coeff = allcoeff.firstord
    else
        coeff = allcoeff.secondord
    end

    if simtype==:cartesian
        ## store coefficients in the struct for sparse matrices
        for p=1:abscode+1 
            i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
            j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            mycoeff = signcod * coeff[p]
            ##
            iorig = cart2lin2D(i,j,nx)
            ifmmord = idxconv.lgrid2fmm[iorig]
            ##
            addentry!(D, irow, ifmmord, mycoeff )
        end

    elseif simtype==:spherical
        ## store coefficients in the struct for sparse matrices
        for p=1:abscode+1 
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
            addentry!(D, irow, ifmmord, mycoeff )
        end

    end

    return
end

##############################################################################

"""
$(TYPEDSIGNATURES)

 Projection operator P containing interpolation coefficients, ordered according to FMM order.
"""
function calcprojttfmmord(ttime,grd,idxconv,coordrec)

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
            coeff,ijcoe = bilinear_interp( ttime,grd,coordrec[r,1],coordrec[r,2],
                                           return_coeffonly=true )

        elseif simdim==:sim3D 
            coeff,ijcoe = trilinear_interp( ttime,grd,coordrec[r,1],coordrec[r,2],coordrec[r,3],
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
function discradjoint2D_FMM_SINGLESRC(idxconv,tt,Dx,Dy,P,pickobs,stdobs,vel2d)
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #

    rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)
    ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
    ##   the solver (\) does not use the correct sparse algo for matrix division
    tmplhs = copy(transpose( (2.0.*Diagonal(Dx*tt)*Dx) .+ (2.0.*Diagonal(Dy*tt)*Dy) ))
    lhs = UpperTriangular(tmplhs)
    # solve the linear system
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
