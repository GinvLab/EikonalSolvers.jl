

#######################################################
##     Misfit of gradient using adjoint              ## 
#######################################################


###############################################################################
########################################################################################
""" 
$(TYPEDSIGNATURES)

Calculate the gradient using the adjoint state method for 3D velocity models. 
Returns the gradient of the misfit function with respect to velocity calculated at the given point (velocity model).
The gradient is calculated using the adjoint state method. 
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 3D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y,z), a 3-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y,z), a 3-column array 
- `pickobs`: observed traveltime picks
- `stdobs`: standard deviation of error on observed traveltime picks, an array with same shape than `pickobs`
- `smoothgrad`: smooth the gradient? true or false

# Returns
- `grad`: the gradient as a 3D array

"""
function gradttime3D(vel::Array{Float64,3},grd::GridEik3D,coordsrc::Array{Float64,2},coordrec::Vector{Matrix{Float64}},
                     pickobs::Vector{Vector{Float64}},stdobs::Vector{Vector{Float64}} ;
                     smoothgradsourceradius::Integer=3,smoothgrad::Bool=false )
   
    if typeof(grd)==Grid3D
        simtype = :cartesian
    elseif typeof(grd)==Grid3DSphere
        simtype = :spherical
    end
    if simtype==:cartesian
        n1,n2,n3 = grd.nx,grd.ny,grd.nz #size(vel)  ## NOT A STAGGERED GRID!!!
        ax1min,ax1max = grd.x[1],grd.x[end]
        ax2min,ax2max = grd.y[1],grd.y[end]
        ax3min,ax3max = grd.z[1],grd.z[end]
    elseif simtype==:spherical
        n1,n2,n3 = grd.nr,grd.nθ,grd.nφ
        ax1min,ax1max = grd.r[1],grd.r[end]
        ax2min,ax2max = grd.θ[1],grd.θ[end]
        ax3min,ax3max = grd.φ[1],grd.φ[end]
    end

    @assert size(coordsrc,2)==3
    @assert all(vel.>0.0)
    @assert size(coordsrc,1)==length(coordrec)
    @assert all(ax1min.<=coordsrc[:,1].<=ax1max)
    @assert all(ax2min.<=coordsrc[:,2].<=ax2max)
    @assert all(ax3min.<=coordsrc[:,3].<=ax3max)

    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)

    ## array of workers' ids
    wks = workers()
    
    grad = zeros(n1,n2,n3)
    ## do the calculations
    @sync for s=1:nchu
        igrs = grpsrc[s,1]:grpsrc[s,2]
        @async grad .+= remotecall_fetch(calcgradsomesrc3D,wks[s],vel,
                                            coordsrc[igrs,:],coordrec[igrs],
                                            grd,stdobs[igrs],pickobs[igrs],
                                            smoothgradsourceradius )
    end    

    ## smooth gradient
    if smoothgrad
        l = 5  # 5 pixels kernel
        grad = smoothgradient(l,grad)
    end
    return grad
end

################################################################3

"""
$(TYPEDSIGNATURES)

Calculate the gradient for some requested sources 
"""
function calcgradsomesrc3D(vel::Array{Float64,3},xyzsrc::Array{Float64,2},coordrec::Vector{Matrix{Float64}},
                           grd::GridEik3D,stdobs::Vector{Vector{Float64}},pickobs1::Vector{Vector{Float64}},
                           smoothgradsourceradius::Integer )

    nx,ny,nz=size(vel)
    nsrc = size(xyzsrc,1)
    grad1 = zeros(nx,ny,nz)


    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc
      
        ###########################################
        ## calc ttime, etc.
       # al = @allocated
        ttgrdonesrc,idxconv,tt_fmmord1,Dx_fmmord1,Dy_fmmord1,Dz_fmmord1 = ttFMM_hiord(vel,xyzsrc[s,:],
                                                                                      grd,dodiscradj=true)

        # println("$s after ttFMM allocated: $(al/1e6)")

        # projection operator P ordered according to FMM order
        # al = @allocated
        P_fmmord1 = calcprojttfmmord(ttgrdonesrc,grd,idxconv,coordrec[s])

        # println("$s after P allocated: $(al/1e6)")

        ###########################################
        # discrete adjoint formulation
        # al = @allocated
        grad1 .+= discradjoint3D_FMM_SINGLESRC(idxconv,tt_fmmord1,Dx_fmmord1,Dy_fmmord1,Dz_fmmord1,
                                               P_fmmord1,pickobs1[s],stdobs[s],vel)

        #println("$s after discr. adj. allocated: $(al/1e6)")

        ###########################################
        ## smooth gradient around the source
        # al = @allocated
        smoothgradaroundsrc3D!(grad1,xyzsrc[s,:],grd,radiuspx=smoothgradsourceradius)
        # println("$s after smoothing allocated: $(al/1e6)")
        
    end

    return grad1
end



##############################################################################

"""
$(TYPEDSIGNATURES)

 Set the coefficients (elements) of the derivative matrices in the x and y directions.
    """
function setcoeffderiv3D!(D::VecSPDerivMat,irow::Integer,idxconv::MapOrderGridFMM3D,
                          codeDxy_orig::Array{Int64,2},allcoeff::CoeffDerivatives,ijpt::AbstractVector;
                          axis::Symbol,simtype::Symbol)
    
    # get the linear index in the original grid
    iptorig = idxconv.lfmm2grid[irow]
    # get the (i,j) indices in the original grid
    nx = idxconv.nx
    ny = idxconv.ny
    lin2cart3D!(iptorig,nx,ny,ijpt)
    igrid = ijpt[1]
    jgrid = ijpt[2]
    kgrid = ijpt[3]
    # get the codes of the derivatives (+1,-2,...)
    # extract codes for X and Y
    codex,codey,codez = codeDxy_orig[iptorig,1],codeDxy_orig[iptorig,2],codeDxy_orig[iptorig,3]

    whX::Int=0
    whY::Int=0
    whZ::Int=0

    if axis==:X
        if codex==0
            # no derivatives have been used...
            return
        else
            code = codex
            whX=1
            whY=0
            whZ=0
        end

    elseif axis==:Y
        if codey==0
            # no derivatives have been used...
            return
        else
            code = codey
            whX=0
            whY=1
            whZ=0
        end
        
    elseif axis==:Z
        if codez==0
            # no derivatives have been used...
            return
        else
            code = codez
            whX=0
            whY=0
            whZ=1
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
            k = kgrid + whZ*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            mycoeff = signcod * coeff[p]
            ##
            iorig = cart2lin3D(i,j,k,nx,ny)
            ifmmord = idxconv.lgrid2fmm[iorig]
            ##
            addentry!(D, irow, ifmmord, mycoeff )
        end

    elseif simtype==:spherical
        ## store coefficients in the struct for sparse matrices
        for p=1:abscode+1 
            i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
            j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            k = kgrid + whZ*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            if axis==:X
                mycoeff = signcod * coeff[p]
            else
                # in this case coeff depends on radius (index i)
                mycoeff = signcod * coeff[i,p]
            end
            ##
            iorig = cart2lin3D(i,j,k,nx,ny)
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

 Solve the discrete adjoint equations and return the gradient of the misfit.
"""
function discradjoint3D_FMM_SINGLESRC(idxconv,tt,Dx,Dy,Dz,P,pickobs,stdobs,vel3d)
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #

    rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)
    ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
    ##   the solver (\) does not use the correct sparse algo for matrix division
    tmplhs = copy(transpose( (2.0.*Diagonal(Dx*tt)*Dx) .+ (2.0.*Diagonal(Dy*tt)*Dy) .+ (2.0.*Diagonal(Dz*tt)*Dz) ))
    lhs = UpperTriangular(tmplhs)

    # @show rank(tmplhs),size(tt)
    # @show nnz(Dx),nnz(Dy),nnz(Dz)
    # @show rank(Dx),rank(Dy),rank(Dz)
    # @show rank(Dx.+Dy.+Dz)

    # DD = Dx.+Dy.+Dz
    # for i=1:size(DD,1)
    #     nonzero = nnz(DD[i,:])
    #     if nonzero<=0
    #         @show i,nonzero
    #     end
    # end
    
    # solve the linear system
    lambda_fmmord = lhs\rhs

    # println("reorder lambda, calc grad")
    # @time begin

    ##--------------------------------------
    # reorder lambda from fmmord to original grid!!
    N = length(lambda_fmmord)
    lambda = Vector{Float64}(undef,N)
    for p=1:N
        iorig = idxconv.lfmm2grid[p]
        lambda[iorig] = lambda_fmmord[p]
    end

    #gradvec = 2.0 .* transpose(lambda) * Diagonal(1.0./ (vec(vel2d).^2) )
    gradvec = 2.0 .* lambda ./ vec(vel3d).^3

    grad3d = reshape(gradvec,idxconv.nx,idxconv.ny,idxconv.nz)

    return grad3d
end

##############################################################################

"""
$(TYPEDSIGNATURES)

 Attempt to reconstruct how derivatives have been calculated in the area of 
  the refinement of the source. Updates the codes for derivatives used to construct
   Dx and Dy.
"""
function derivaroundsrcfmm3D!(lseq::Integer,idxconv::MapOrderGridFMM3D,codeD::MVector)
    # lipt = index in fmmord in sequential order (1,2,3,4,...)
    
    nx = idxconv.nx
    ny = idxconv.ny
    nz = idxconv.nz

    ijpt = MVector(0,0,0)
    idxpt = idxconv.lfmm2grid[lseq]
    lin2cart3D!(idxpt,nx,ny,ijpt)
    ipt,jpt,kpt = ijpt[1],ijpt[2],ijpt[3]
    
    codeD[:] .= 0
    for axis=1:3

        chosenidx = lseq
        for dir=1:2

            ## map the 6 cases to an integer as in linear indexing...
            lax = dir + 2*(axis-1)
            if lax==1 # axis==1
                ish = 1
                jsh = 0
                ksh = 0
            elseif lax==2 # axis==1
                ish = -1
                jsh = 0
                ksh = 0
            elseif lax==3 # axis==2
                ish = 0
                jsh = 1
                ksh = 0
            elseif lax==4 # axis==2
                ish = 0
                jsh = -1
                ksh = 0
            elseif lax==5 # axis==3
                ish = 0
                jsh = 0
                ksh = 1
            elseif lax==6 # axis==3
                ish = 0
                jsh = 0
                ksh = -1
            end
            
            ##=========================
            ## first order
            i = ipt+ish
            j = jpt+jsh
            k = kpt+ksh

            ## check if on boundaries
            isonb1st,isonb2nd = isonbord(i,j,k,nx,ny,nz)

            if !isonb1st
                ## calculate the index of the neighbor in the fmmord
                l = cart2lin3D(i,j,k,nx,ny)
                #idxne1 = findfirst(idxconv.lfmm2grid.==l)
                idxne1 = idxconv.lgrid2fmm[l]
                
                #@show lseq,l,idxne1,chosenidx
            
                if idxne1!=nothing && idxne1<chosenidx                   

                    # to make sure we chose correctly the direction
                    chosenidx = idxne1
                    # save derivative choices [first order]
                    if axis==1
                        codeD[axis]=ish
                    elseif axis==2
                        codeD[axis]=jsh
                    else
                        codeD[axis]=ksh
                    end

                    if !isonb2nd
                        ##=========================
                        ## second order
                        i = ipt + 2*ish
                        j = jpt + 2*jsh
                        k = kpt + 2*ksh
                        
                        ## calculate the index of the neighbor in the fmmord
                        l = cart2lin3D(i,j,k,nx,ny)
                        idxne2 = idxconv.lgrid2fmm[l]
                        
                        ##===========================================================
                        ## WARNING! The traveltime for the second order point must
                        ##   be smaller than the one for the first order for selecting
                        ##   second order. Otherwise first order only.
                        ##  Therefore we compare idxne2<idxne1 instead of idxne2<idxpt
                        ##===========================================================
                        if idxne2<idxne1
                            # save derivative choices [second order]
                            if axis==1
                                codeD[axis]=2*ish
                            elseif axis==2
                                codeD[axis]=2*jsh
                            else
                                codeD[axis]=2*ksh
                            end
                        end

                    end
                end
            end            
        end
    end

    return 
end

#######################################################################################


##################################################
#end # end module
##################################################
