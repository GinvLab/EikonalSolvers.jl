
##############################################################################

# structs for sparse matrices to represent derivatives

struct VecSPDerivMat
    i::Vector{Int64}
    j::Vector{Int64}
    v::Vector{Float64}
    Nnnz::Base.RefValue{Int64}
    Nsize::Vector{Int64}

    function VecSPDerivMat(; i,j,v,Nsize)
        Nnnz = Ref(0)
        new(i,j,v,Nnnz,Nsize) 
    end
end

function addentry!(D::VecSPDerivMat,i::Integer,j::Integer,v::Float64)
    p = D.Nnnz[]+1
    #@show D.Nsize,p,i,j
    D.i[p] = i
    D.j[p] = j
    D.v[p] = v
    D.Nnnz[] = p 
    return
end



##############################################################################

function setcoeffderiv!(D::VecSPDerivMat,irow,idx_fmmord,codeDxy_orig,
                        cartid_nxny,linid_nxny,dh; axis)

    # get the linear index in the original grid
    iptorig = idx_fmmord[irow]
    # get the (i,j) indices in the original grid
    cij = cartid_nxny[iptorig]
    i,j = cij[1],cij[2]
    # get the codes of the derivatives (+1,-2,...)
    codes_orig = codeDxy_orig[iptorig,:]

    if codes_orig == [0,0]
        # no derivatives have been used...
        return
    end

    ##--------------------------
    # closure over idx_fmmord
    function orig2fmmord(i,j)
        iorig = linid_nxny[i,j]
        ifmmord = findfirst(idx_fmmord.==iorig)
        return ifmmord
    end
    ##--------------------------

    #@show i,j,irow,code

    if axis=="X"
        code = codes_orig[1]

        if code==0
            nothing

        elseif code==-2
            addentry!(D, irow, orig2fmmord(i,j),    3.0/(2.0*dh) ) #  0   
            addentry!(D, irow, orig2fmmord(i-1,j), -4.0/(2.0*dh) ) # -1
            addentry!(D, irow, orig2fmmord(i-2,j),  1.0/(2.0*dh) ) # -2

        elseif code==-1
            addentry!(D, irow, orig2fmmord(i,j),    1.0/dh ) #  0
            addentry!(D, irow, orig2fmmord(i-1,j), -1.0/dh ) # -1

        elseif code==1
            addentry!(D, irow, orig2fmmord(i,j),  -1.0/dh ) #  0
            addentry!(D, irow, orig2fmmord(i+1,j), 1.0/dh ) # +1

        elseif code==2
            addentry!(D, irow, orig2fmmord(i,j),   -3.0/(2.0*dh) ) #  0
            addentry!(D, irow, orig2fmmord(i+1,j),  4.0/(2.0*dh) ) # +1
            addentry!(D, irow, orig2fmmord(i+2,j), -1.0/(2.0*dh) ) # +2

        else
            error("setcoeffderiv(): Wrong code...")

        end

    elseif axis=="Y"
        code = codes_orig[2]

        if code==0
            nothing

        elseif code==-2
            addentry!(D, irow, orig2fmmord(i,j),   -3.0/(2.0*dh) ) #  0                        
            addentry!(D, irow, orig2fmmord(i,j-1),  4.0/(2.0*dh) ) # -1
            addentry!(D, irow, orig2fmmord(i,j-2), -1.0/(2.0*dh) ) # -2

        elseif code==-1
            addentry!(D, irow, orig2fmmord(i,j),    1.0/dh ) #  0
            addentry!(D, irow, orig2fmmord(i,j-1), -1.0/dh ) # -1

        elseif code==1
            addentry!(D, irow, orig2fmmord(i,j),  -1.0/dh ) #  0
            addentry!(D, irow, orig2fmmord(i,j+1), 1.0/dh ) # +1

        elseif code==2
            addentry!(D, irow, orig2fmmord(i,j),   -3.0/(2.0*dh) ) #  0
            addentry!(D, irow, orig2fmmord(i,j+1),  4.0/(2.0*dh) ) # +1
            addentry!(D, irow, orig2fmmord(i,j+2), -1.0/(2.0*dh) ) # +2

        else
            error("setcoeffderiv(): Wrong code...")

        end

    else
        error("setcoeffderiv(): Wrong axis...")
    end

    return
end

##############################################################################
# """
# $(TYPEDSIGNATURES)

#  Find the corresponding linear index in FMM order given (i,j) in the original grid.
# """
# function orig2fmmord(linid_nxny::LinearIndices,idx_fmmord::Vector{<:Integer},i::Integer,j::Integer)
#     # convert to linear index
#     lidx_orig = linid_nxny[i,j]
#     # find corresponding index in FMM order
#     idx_fmmord = findfirst(idx_fmmord.==lidx_orig)    
#     return idx_fmmord
# end

##############################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 2D using traditional stencils on regular grid. 
"""
function ttFMM_hiord_discradj(vel::Array{Float64,2},src::Vector{Float64},grd::Grid2D)
                                     
    ## Sizes
    nx,ny = grd.nx,grd.ny #size(vel)  ## NOT A STAGGERED GRID!!!
    hgrid = grd.hgrid
    nxXny = nx*ny
    epsilon = 1e-6

    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nx,ny)
    ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nx,ny)
    status[:,:] .= 0   ## set all to far
    
    ##########################################
    ##refinearoundsrc=true
    
    if extrapars.refinearoundsrc
        ##---------------------------------
        ## 
        ## Refinement around the source      
        ##
        ttaroundsrc!(status,ttime,vel,src,grd,inittt)
 
    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        println("\nttFMM_hiord(): NO refinement around the source! \n")

        ## source location, etc.      
        ## REGULAR grid
        onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=false )

        ##
        ## Status of nodes
        status[onsrc] .= 2 ## set to accepted on src
        
    end # if refinearoundsrc
    ###################################################### 

    #-------------------------------
    ## init FMM 
    neigh = [1  0;
             0  1;
            -1  0;
             0 -1]
    
    ## get all i,j accepted
    ijss = findall(status.==2) 
    is = [l[1] for l in ijss]
    js = [l[2] for l in ijss]
    naccinit = length(ijss)

    ########################
    ## conversion cart to lin indices, old sub2ind
    linid_nxny = LinearIndices((nx,ny))
    ## conversion lin to cart indices, old ind2sub
    cartid_nxny = CartesianIndices((nx,ny))


    ########################
    # discrete adjoint: init stuff
    idx_fmmord = zeros(Int64,nxXny)
    tt_fmmord  = zeros(nxXny)
    idD = zeros(Int64,2)
    codeDxy = zeros(Int64,nxXny,2)
    vecDx_fmmord = VecSPDerivMat( i=zeros(Int64,nxXny*3), j=zeros(Int64,nxXny*3), v=zeros(nxXny*3), Nsize=[nxXny,nxXny] )
    vecDy_fmmord = VecSPDerivMat( i=zeros(Int64,nxXny*3), j=zeros(Int64,nxXny*3), v=zeros(nxXny*3), Nsize=[nxXny,nxXny] )                 
    
    
    ########################
    # discrete adjoint: first visited points in FMM order
    ttfirstpts = Vector{Float64}(undef,naccinit)
    for l=1:naccinit
        i,j = is[l],js[l]
        ttij = ttime[i,j]
        # store initial points' FMM order
        ttfirstpts[l] = ttij
    end
    # sort according to arrival time
    spidx = sortperm(ttfirstpts)
    # store the first accepted points' order
    for l=1:naccinit
        # ordered index
        p = spidx[l]
        # go from cartesian (i,j) to linear
        idx_fmmord[l] = linid_nxny[is[p],js[p]]

        # store arrival time for first points in FMM order
        tt_fmmord[l] = ttime[is[p],js[p]]

        #################################################################
        # Here we store a 1 in the diagonal because we are on a source node...
        #  store arrival time for first points in FMM order
        #
        addentry!(vecDx_fmmord,l,l,1.0)                   ## <<<<<<<<<<<<<========= CHECK this! =============#####
        addentry!(vecDy_fmmord,l,l,1.0)                   ## <<<<<<<<<<<<<========= CHECK this! =============#####
        #
        #################################################################

        #@show is[p],js[p],ttime[is[p],js[p]],idx_fmmord[l]
    end
    

    ## Init the min binary heap with void arrays but max size
    Nmax=nxXny
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0     

    ## construct initial narrow band
    for l=1:naccinit ##
        
        for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                continue
            end

            if status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord_discradj!(ttime,vel,grd,status,i,j,idD)

                # get handle
                # han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j]
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1

                # codes of chosen derivatives for adjoint
                codeDxy[han,:] .= idD
            end
        end
    end

    #-------------------------------
    ## main FMM loop
    totnpts = nxXny
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nx,ny),han)
        cija = cartid_nxny[han]
        ia,ja = cija[1],cija[2]
        #ja = div(han,nx) +1
        #ia = han - nx*(ja-1)
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt

        # store the linear index of FMM order 
        idx_fmmord[node] = linid_nxny[ia,ja]
        # store arrival time for first points in FMM order
        tt_fmmord[node] = tmptt


        ## try all neighbors of newly accepted point
        for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord_discradj!(ttime,vel,grd,status,i,j,idD)
                han = linid_nxny[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

                # codes of chosen derivatives for adjoint
                codeDxy[han,:] .= idD
                
            elseif status[i,j]==1 ## narrow band                
                # update the traveltime for this point
                tmptt = calcttpt_2ndord_discradj!(ttime,vel,grd,status,i,j,idD)

                # get handle
                han = linid_nxny[i,j]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

                # codes of chosen derivatives for adjoint 
                codeDxy[han,:] .= idD
            end
        end
        ##-------------------------------
    end

    
    for irow=1:nxXny
        
        # reorder the derivative codes (+-1,+-2) according to the FMM order from idx_fmmord
        icol = irow

        #@show irow,icol,cartid_nxny[icol]

        # compute the coefficients for X  derivatives
        setcoeffderiv!(vecDx_fmmord,irow,idx_fmmord,codeDxy,
                       cartid_nxny,linid_nxny,hgrid,axis="X")

        # compute the coefficients for Y derivatives
        setcoeffderiv!(vecDy_fmmord,irow,idx_fmmord,codeDxy,
                       cartid_nxny,linid_nxny,hgrid,axis="Y")

    end

    # create the actual sparse arrays from the vectors
    Nxnnz = vecDx_fmmord.Nnnz[]

    Dx_fmmord = sparse(vecDx_fmmord.i[1:Nxnnz],
                       vecDx_fmmord.j[1:Nxnnz],
                       vecDx_fmmord.v[1:Nxnnz],
                       vecDx_fmmord.Nsize[1], vecDx_fmmord.Nsize[2] )

    Nynnz = vecDy_fmmord.Nnnz[]
    Dy_fmmord = sparse(vecDy_fmmord.i[1:Nynnz],
                       vecDy_fmmord.j[1:Nynnz],
                       vecDy_fmmord.v[1:Nynnz],
                       vecDy_fmmord.Nsize[1], vecDy_fmmord.Nsize[2] ) 

    #########################
    # display(Dx_fmmord)
    # display(Dy_fmmord)

    # for i=1:size(Dx_fmmord,1)
    #     cnnz_X = count(Dx_fmmord[i,:].!=0.0)
    #     cnnz_Y = count(Dy_fmmord[i,:].!=0.0)
    #     if cnnz_X==0
    #         @show i,cnnz_X,cnnz_Y
    #     end
    #     if cnnz_Y==0
    #         @show i,cnnz_X,cnnz_Y
    #     end
    # end

    return ttime,idx_fmmord,tt_fmmord,Dx_fmmord,Dy_fmmord
end

##====================================================================##

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order.
"""
function calcttpt_2ndord_discradj!(ttime::Array{Float64,2},vel::Array{Float64,2},
                                  grd::Grid2D, status::Array{Int64,2},i::Int64,j::Int64,
                                  idD::Vector{<:Integer})
    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    deltah = grd.hgrid 
    # dx2 = dx^2
    # dy2 = dy^2
    nx = grd.nx
    ny = grd.ny
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
    HUGE = 1.0e30

    idD[:] .= 0.0 #zeros(Int64,2)
    
    ## 2 directions
    for axis=1:2
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## two sides for each direction
        for l=1:2

            ## map the 4 cases to an integer as in linear indexing...
            lax = l + 2*(axis-1)
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

            ## check if on boundaries
            isonb1st,isonb2nd = isonbord(i+ish,j+jsh,nx,ny)
                                    
            ## 1st order
            if !isonb1st && status[i+ish,j+jsh]==2 ## 2==accepted
                testval1 = ttime[i+ish,j+jsh]
                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only
                    chosenval1 = testval1
                    use1stord = true

                    # save derivative choices
                    axis==1 ? (idD[axis]=ish) : (idD[axis]=jsh)
                    
                    ## 2nd order
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    if !isonb2nd && status[i+ish2,j+jsh2]==2 ## 2==accepted
                        testval2 = ttime[i+ish2,j+jsh2]
                        ## pick the lowest value of the two
                        ## <=, compare to chosenval 1, *not* 2!!
                        if testval2<=chosenval1 
                            chosenval2=testval2
                            use2ndord=true

                            # save derivative choices
                            axis==1 ? (idD[axis]=2*ish) : (idD[axis]=2*jsh)

                        else
                            chosenval2=HUGE
                            use2ndord=false # this is needed!

                            # save derivative choices
                            axis==1 ? (idD[axis]=ish) : (idD[axis]=jsh)
                            
                        end
                    end
                    
                end
            end
        end # end two sides

        # if axis==1
        #     deltah = dx
        # elseif axis==2
        #     deltah = dy
        # end
        if use2ndord && use1stord # second order
            tmpa2 = 1.0/3.0 * (4.0*chosenval1-chosenval2)
            ## curalpha: make sure you multiply only times the
            ##   current alpha for beta and gamma...
            curalpha = 9.0/(4.0 * deltah^2)
            alpha += curalpha
            beta  += ( -2.0*curalpha * tmpa2 )
            gamma += curalpha * tmpa2^2

        elseif use1stord # first order
            ## curalpha: make sure you multiply only times the
            ##   current alpha for beta and gamma...
            curalpha = 1.0/deltah^2 
            alpha += curalpha
            beta  += ( -2.0*curalpha * chosenval1 )
            gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
        end

    end

    ## compute discriminant 
    sqarg = beta^2-4.0*alpha*gamma

    ## To get a non-negative discriminant, need to fulfil:
    ##    (tx-ty)^2 - 2*s^2/curalpha <= 0
    ##    where tx,ty can be
    ##     t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order
    ##     t? = chosenval1  if 1st order 
    ##  
    ## If discriminant is negative (probably because of sharp contrasts in
    ##  velocity) revert to 1st order for both x and y
    if sqarg<0.0

        begin
            alpha = 0.0
            beta  = 0.0
            gamma = - slowcurpt^2 ## !!!!
            
            idD[:] .= 0.0

            ## 2 directions
            for axis=1:2
                
                use1stord = false
                chosenval1 = HUGE
                
                ## two sides for each direction
                for l=1:2

                    ## map the 4 cases to an integer as in linear indexing...
                    lax = l + 2*(axis-1)
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

                    ## check if on boundaries
                    isonb1st,isonb2nd = isonbord(i+ish,j+jsh,nx,ny)
                    
                    ## 1st order
                    if !isonb1st && status[i+ish,j+jsh]==2 ## 2==accepted
                        testval1 = ttime[i+ish,j+jsh]
                        ## pick the lowest value of the two
                        if testval1<chosenval1 ## < only
                            chosenval1 = testval1
                            use1stord = true

                            # save derivative choices
                            axis==1 ? (idD[axis]=ish) : (idD[axis]=jsh)

                        end
                    end
                end # end two sides

                # if axis==1
                #     deltah = dx
                # elseif axis==2
                #     deltah = dy
                # end
                
                if use1stord # first order
                    ## curalpha: make sure you multiply only times the
                    ##   current alpha for beta and gamma...
                    curalpha = 1.0/deltah^2 
                    alpha += curalpha
                    beta  += ( -2.0*curalpha * chosenval1 )
                    gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
                end
            end
            
            ## recompute sqarg
            sqarg = beta^2-4.0*alpha*gamma

        end ## begin...

        if sqarg<0.0

            if extrapars.allowfixsqarg==true
            
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
    end ## if sqarg<0.0

    ### roots of the quadratic equation
    tmpsq = sqrt(sqarg)
    soughtt1 =  (-beta + tmpsq)/(2.0*alpha)
    soughtt2 =  (-beta - tmpsq)/(2.0*alpha)
    ## choose the largest solution
    soughtt = max(soughtt1,soughtt2)

    return soughtt

end

##############################################################################


# projection operator P ordered according to FMM order
function calcprojttfmmord(ttime,grd,idx_fmmord,coordrec)

    linid_nxny = LinearIndices((grd.nx,grd.ny))
    nxXny = grd.nx*grd.ny

    # calculate the coefficients and their indices using bilinear interpolation
    nrec = size(coordrec,1)
    Ncoe = 4
    ##P = zeros(nrec,nxXny)
    P_i = zeros(Int64,nrec*Ncoe)
    P_j = zeros(Int64,nrec*Ncoe)
    P_v = zeros(nrec*Ncoe)

    q::Int64=1
    for r=1:nrec
        coeff,ijcoe = bilinear_interp( ttime,grd.hgrid,grd.xinit,
                                       grd.yinit,coordrec[r,1],coordrec[r,2],
                                       return_coeffonly=true )
        @assert size(ijcoe,1)==4

        # convert (i,j) from original grid to fmmord
        for l=1:Ncoe
            i,j = ijcoe[l,1],ijcoe[l,2]
            iorig = linid_nxny[i,j]
            jfmmord = findfirst(idx_fmmord.==iorig)
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

    P = sparse(P_i,P_j,P_v,nrec,nxXny)
    return P
end

##############################################################################

function discradjoint_hiord_SINGLESRC(idx_fmmord,tt,Dx,Dy,P,pickobs,stdobs,grd,vel2d)
    #
    # all stuff must be in FMM order (e.g., tt_fmmord)
    #

    #@show size(P), size(tt),size(pickobs),size(stdobs)
    rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)

    tmplhs = transpose( (2.0 .* Diagonal(Dx*tt) * Dx) .+ (2.0 .* Diagonal(Dy*tt) * Dy) )

    lhs = UpperTriangular(tmplhs)

    # println("2.0 .* Diagonal(Dx*tt) * Dx) + [...]:")
    # display((2.0 .* Diagonal(Dx*tt) * Dx) .+ (2.0 .* Diagonal(Dy*tt) * Dy) )

    lambda_fmmord = lhs\rhs

    ##--------------------------------------
    # reorder lambda from fmmord to original grid!!
    N = length(lambda_fmmord)
    lambda = Vector{Float64}(undef,N)
    for p=1:N
        iorig = idx_fmmord[p]
        lambda[iorig] = lambda_fmmord[p]
    end

    #gradvec = -2.0 .* transpose(lambda) * Diagonal(1.0./vec(vel2d))
    gradvec = -2.0 .* lambda ./ vec(vel2d)

    grad2d = reshape(gradvec,grd.nx,grd.ny)
    
    return grad2d
end

##############################################################################
