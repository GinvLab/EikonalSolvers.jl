
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
            whY=0
        end
        
    elseif axis==:Z
        if codez==0
            # no derivatives have been used...
            return
        else
            code = codez
            whX=0
            whY=0
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
            k = kgrid + whZ*signcod*(p-1)  # start from 0  (e.g., 0,-1)
            mycoeff = signcod * coeff[p]
            ##
            iorig = cart2lin3D(i,j,nx,ny)
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
            elseif axis==:Y
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
            lax = l + 2*(axis-1)
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
                    axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

                    if !isonb2nd
                        ##=========================
                        ## second order
                        i = ipt + 2*ish
                        j = jpt + 2*jsh
                        k = kpt + 2*jsh
                        
                        ## calculate the index of the neighbor in the fmmord
                        l = cart2lin(i,j,nx,ny)
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



