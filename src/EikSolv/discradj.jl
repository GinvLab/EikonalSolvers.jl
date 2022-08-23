

##############################################################################

##=======================================================
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
    #@show i,j,D.Nsize,p
    D.i[p] = i
    D.j[p] = j
    D.v[p] = v
    D.Nnnz[] = p 
    return
end

##############################################################################

@inline function cart2lin(i::Integer,j::Integer,ni::Integer)
    l = (j-1)*ni + i
    # tmpl = LinearIndices((ni,140))[i,j]
    # @assert l==tmpl
    return l
end

# function lin2cart(l::Integer,ni::Integer)
#     i = div(l,ni) + 1
#     j = l-(i-1)*ni
#     return (i,j)
# end

# mutable struct IJPoint
#     i::Integer
#     j::Integer
# end

@inline function lin2cart!(l::Integer,ni::Integer,point::MVector) #point::IJPoint)
    point[2] = div(l-1,ni) + 1
    point[1] = l-(point[2]-1)*ni
    # point.j = div(l-1,ni) + 1
    # point.i = l-(point.j-1)*ni
    # tmpi,tmpj = Tuple(CartesianIndices((ni,140))[l])
    # @show l,point,(tmpi,tmpj)
    # @assert point.i==tmpi
    # @assert point.j==tmpj
    return 
end

##############################################################################

struct MapOrderGridFMM
    "Linear grid indices in FMM order (as visited by FMM)"
    lfmm2grid::Vector{Int64} # idx_fmmord
    "Linear FMM indices in grid order"
    lgrid2fmm::Vector{Int64} # idx_gridord
    # cart2lin::LinearIndices
    # lin2cart::CartesianIndices
    nx::Int64
    ny::Int64

    function MapOrderGridFMM(nx,ny)
        nxXny = nx*ny
        lfmm2grid = zeros(Int64,nxXny)
        lgrid2fmm = zeros(Int64,nxXny)
        # cart2lin = LinearIndices((nx,ny))
        # lin2cart = CartesianIndices((nx,ny))
        #return new(lfmm2grid,lgrid2fmm,cart2lin,lin2cart,nx,ny)
        return new(lfmm2grid,lgrid2fmm,nx,ny)
    end
end


struct VarsFMMOrder
    ttime::Vector{Float64}
    vecDx::VecSPDerivMat
    vecDy::VecSPDerivMat

    function VarsFMMOrder(nx,ny)
        nxXny =  nx*ny
        ttime = zeros(nxXny)
        vecDx = VecSPDerivMat( i=zeros(Int64,nxXny*3), j=zeros(Int64,nxXny*3),
                                v=zeros(nxXny*3), Nsize=[nxXny,nxXny] )
        vecDy = VecSPDerivMat( i=zeros(Int64,nxXny*3), j=zeros(Int64,nxXny*3),
                               v=zeros(nxXny*3), Nsize=[nxXny,nxXny] )
        return new(ttime,vecDx,vecDy)
    end
end

##############################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 2D using traditional stencils on regular grid 
  for discrete adjoint calculations. 
"""
function ttFMM_hiord_discradj(vel::Array{Float64,2},src::Vector{Float64},grd::Grid2D)
                                     
    # println(" -> init")
    # @time begin

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
    
    ########################
    # discrete adjoint: init stuff
    idxconv = MapOrderGridFMM(nx,ny)
    fmmord = VarsFMMOrder(nx,ny)
    idD = MVector(0,0)
    codeDxy = zeros(Int64,nxXny,2)
    ijpt = MVector(0,0) #IJPoint(0,0)
      
    ###########################################
    
    if extrapars.refinearoundsrc
        ##---------------------------------
        ## 
        ## Refinement around the source      
        ##
        #
        ttaroundsrc_discradj!(status,ttime,vel,src,grd,inittt,idxconv,fmmord )

        ##-----------------------------------------
        ## 
        ## DISCRETE ADJOINT WORKAROUND FOR DERIVATIVES
        ##
        ## get all i,j accepted
        ijss = findall(status.==2) 
        is = [l[1] for l in ijss]
        js = [l[2] for l in ijss]
        naccinit = length(ijss)

        #@show count(ttime.>=0.0),count(fmmord.ttime.>=0.0)

        # How many initial points to skip, considering them as "onsrc"?
        skipnptsDxy = 4
        
        ## pre-compute some of the mapping between fmm and orig order
        for i=1:naccinit
            ifm = idxconv.lfmm2grid[i]
            idxconv.lgrid2fmm[ifm] = i
        end

        for l=1:naccinit

            if l<=skipnptsDxy
                #################################################################
                # Here we store a 1 in the diagonal because we are on a source node...
                #  store arrival time for first points in FMM order
                addentry!(fmmord.vecDx,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                addentry!(fmmord.vecDy,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                #################################################################
            end

            ## "reconstruct" derivative stencils from known FMM order and arrival times
            derivaroundsrcfmm!(l,idxconv,idD)

            if idD==[0,0]
                #################################################################
                # Here we store a 1 in the diagonal because we are on a source node...
                #  store arrival time for first points in FMM order
                addentry!(fmmord.vecDx,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                addentry!(fmmord.vecDy,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                #################################################################
            else
                l_fmmord = idxconv.lfmm2grid[l]
                codeDxy[l_fmmord,:] .= idD
            end
        end

        
    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        #println("ttFMM_hiord_discradj(): NO refinement around the source! ")

        ## source location, etc.      
        ## REGULAR grid
        onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=false )

        ##
        ## Status of nodes
        status[onsrc] .= 2 ## set to accepted on src
        
        ## get all i,j accepted
        ijss = findall(status.==2) 
        is = [l[1] for l in ijss]
        js = [l[2] for l in ijss]
        naccinit = length(ijss)
        
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
            idxconv.lfmm2grid[l] = cart2lin(is[p],js[p],nx)

            ######################################
            # store arrival time for first points in FMM order
            ttgrid = ttime[is[p],js[p]]
            ## The following to avoid a singular upper triangular matrix in the
            ##  adjoint equation. Otherwise there will be a row of only zeros in the LHS.
            if ttgrid==0.0
                fmmord.ttime[l] = eps()
            else
                fmmord.ttime[l] = ttgrid
            end

            #################################################################
            # Here we store a 1 in the diagonal because we are on a source node...
            #  store arrival time for first points in FMM order
            #
            addentry!(fmmord.vecDx,l,l,1.0)                   ## <<<<<<<<<<<<<========= CHECK this! =============#####
            addentry!(fmmord.vecDy,l,l,1.0)                   ## <<<<<<<<<<<<<========= CHECK this! =============#####
            #
            #################################################################

            #@show is[p],js[p],ttime[is[p],js[p]],idxconv.lfmm2grid[l]
        end

    end # if refinearoundsrc
    ###################################################### 


    # println(" -> big loop")
    # @time begin
        
    # @show count(ttime.>0.0),count(fmmord.ttime.>0.0)
    # @show count(ttime.<0.0),count(fmmord.ttime.<0.0)
    
    #-------------------------------
    ## init FMM
    # static array
    neigh = SA[1  0;
               0  1;
              -1  0;
               0 -1]
    
    # ## get all i,j accepted
    # ijss = findall(status.==2) 
    # is = [l[1] for l in ijss]
    # js = [l[2] for l in ijss]
    # naccinit = length(ijss)
    # @show naccinit

    ## Init the min binary heap with void arrays but max size
    Nmax = nxXny
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0     

 
    # println(" -> big loop")
    # @time begin

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
                #boh = LinearIndices((nx,ny))[i,j]
                han = cart2lin(i,j,nx)
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
        ##
        lin2cart!(han,nx,ijpt)
        ia,ja = ijpt[1],ijpt[2]
        #@show ia,ja
        # ja = div(han,nx) +1
        # ia = han - nx*(ja-1)
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt

        # store the linear index of FMM order
        idxconv.lfmm2grid[node] = cart2lin(ia,ja,nx)
        # store arrival time for first points in FMM order
        fmmord.ttime[node] = tmptt


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
                han = cart2lin(i,j,nx)
                #@show i,j,han
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

                # codes of chosen derivatives for adjoint
                codeDxy[han,:] .= idD
                
            elseif status[i,j]==1 ## narrow band                
                # update the traveltime for this point
                tmptt = calcttpt_2ndord_discradj!(ttime,vel,grd,status,i,j,idD)

                # get handle
                han = cart2lin(i,j,nx)
                #@show i,j,han
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

                # codes of chosen derivatives for adjoint 
                codeDxy[han,:] .= idD
            end
        end
        ##-------------------------------
    end

     
    #end # begin @time

    # println(" -> setcoeffderiv")
    # @time begin

    ## pre-compute the mapping between fmm and orig order
    for i=1:nxXny
        ifm = idxconv.lfmm2grid[i]
        idxconv.lgrid2fmm[ifm] = i
    end

    # pre-determine derivative coefficients for positive codes (0,+1,+2)
    allcoeff = [[-1.0/hgrid, 1.0/hgrid], [-3.0/(2.0*hgrid), 4.0/(2.0*hgrid), -1.0/(2.0*hgrid)]]
    
    for irow=1:nxXny
        
        # compute the coefficients for X  derivatives
        setcoeffderiv!(fmmord.vecDx,irow,idxconv,codeDxy,allcoeff,ijpt,axis=:X)
        
        # compute the coefficients for Y derivatives
        setcoeffderiv!(fmmord.vecDy,irow,idxconv,codeDxy,allcoeff,ijpt,axis=:Y)
        
    end
    
    #end # @time begin


    # println(" -> assemble sparse matrices")
    # @time begin

    # create the actual sparse arrays from the vectors
    Nxnnz = fmmord.vecDx.Nnnz[]
    Dx_fmmord = sparse(fmmord.vecDx.i[1:Nxnnz],
                       fmmord.vecDx.j[1:Nxnnz],
                       fmmord.vecDx.v[1:Nxnnz],
                       fmmord.vecDx.Nsize[1], fmmord.vecDx.Nsize[2] )

    Nynnz = fmmord.vecDy.Nnnz[]
    Dy_fmmord = sparse(fmmord.vecDy.i[1:Nynnz],
                       fmmord.vecDy.j[1:Nynnz],
                       fmmord.vecDy.v[1:Nynnz],
                       fmmord.vecDy.Nsize[1], fmmord.vecDy.Nsize[2] ) 
    #end # @time begin

    return ttime,idxconv,fmmord.ttime,Dx_fmmord,Dy_fmmord
end

##====================================================================##


"""
$(TYPEDSIGNATURES)

 Set the coefficients (elements) of the derivative matrices in the x and y directions.
"""
function setcoeffderiv!(D::VecSPDerivMat,irow::Integer,idxconv::MapOrderGridFMM,
                        codeDxy_orig,allcoeff,ijpt; axis)
                        #irow,idxconv,codeDxy_orig,allcoeff,ijpt; axis)
                    
                        # idx_orig,idxconv.lfmm2grid,codeDxy_orig,
                        # idxconv.lin2cart,idxconv.cart2lin,allcoeff; axis)
    nx = D.Nsize[1]

    #@time begin
#@time begin
    # get the linear index in the original grid
    iptorig = idxconv.lfmm2grid[irow]
#end

    # get the (i,j) indices in the original grid
    lin2cart!(iptorig,nx,ijpt)
    igrid = ijpt[1]
    jgrid = ijpt[2]
    # get the codes of the derivatives (+1,-2,...)
    #codes_orig = view(codeDxy_orig,iptorig,:)
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


    ##--------------------------
    # closure over idx_orig
    function ijgrid2fmmord(i,j)
        iorig = cart2lin(i,j,idxconv.nx)
        #ifmmord = findfirst(idxconv.lfmm2grid.==iorig)
        ifmmord = idxconv.lgrid2fmm[iorig] #findfirst(idxconv.lfmm2grid.==iorig)
        return ifmmord
    end
    ##--------------------------

    abscode = abs(code)
    signcod = sign(code)
    ## select first or second order coefficients
    coeff = allcoeff[abscode]

    ## store coefficients in the struct for sparse matrices
    for p=1:abscode+1 
        i = igrid + whX*signcod*(p-1)  # start from 0  (e.g., 0,+1,+2)
        j = jgrid + whY*signcod*(p-1)  # start from 0  (e.g., 0,-1)
        mycoeff = signcod * coeff[p]
        addentry!(D, irow, ijgrid2fmmord(i,j), mycoeff )
    end

    ## old version...
    # if axis==:X
    #     code = codex

    #     if code==0
    #         nothing

    #     elseif code==-2
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),    3.0/(2.0*dh) ) #  0   
    #         addentry!(D, irow, ijgrid2fmmord(igrid-1,jgrid), -4.0/(2.0*dh) ) # -1
    #         addentry!(D, irow, ijgrid2fmmord(igrid-2,jgrid),  1.0/(2.0*dh) ) # -2

    #     elseif code==-1
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),    1.0/dh ) #  0
    #         addentry!(D, irow, ijgrid2fmmord(igrid-1,jgrid), -1.0/dh ) # -1

    #     elseif code==1
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),  -1.0/dh ) #  0
    #         addentry!(D, irow, ijgrid2fmmord(igrid+1,jgrid), 1.0/dh ) # +1

    #     elseif code==2
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),   -3.0/(2.0*dh) ) #  0
    #         addentry!(D, irow, ijgrid2fmmord(igrid+1,jgrid),  4.0/(2.0*dh) ) # +1
    #         addentry!(D, irow, ijgrid2fmmord(igrid+2,jgrid), -1.0/(2.0*dh) ) # +2

    #     else
    #         error("setcoeffderiv(): Wrong code...")

    #     end

    # elseif axis==:Y
    #     code = codey

    #     if code==0
    #         nothing

    #     elseif code==-2
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),   -3.0/(2.0*dh) ) #  0                        
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid-1),  4.0/(2.0*dh) ) # -1
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid-2), -1.0/(2.0*dh) ) # -2

    #     elseif code==-1
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),    1.0/dh ) #  0
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid-1), -1.0/dh ) # -1

    #     elseif code==1
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),  -1.0/dh ) #  0
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid+1), 1.0/dh ) # +1

    #     elseif code==2
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid),   -3.0/(2.0*dh) ) #  0
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid+1),  4.0/(2.0*dh) ) # +1
    #         addentry!(D, irow, ijgrid2fmmord(igrid,jgrid+2), -1.0/(2.0*dh) ) # +2

    #     else
    #         error("setcoeffderiv(): Wrong code...")

    #     end

    # else
    #     error("setcoeffderiv(): Wrong axis...")
    # end
    
    #end # begin
    return
end

##############################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order. This function is 
    intended for subsequent calculations using the discrete adjoint method.
"""
function calcttpt_2ndord_discradj!(ttime::Array{Float64,2},vel::Array{Float64,2},
                                  grd::Grid2D, status::Array{Int64,2},i::Int64,j::Int64,
                                  codeD::MVector{2,Int64}) #Vector{<:Integer})
    
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

    codeD[:] .= 0.0 #zeros(Int64,2)
    
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
                    axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                    
                    ## 2nd order
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    if !isonb2nd && status[i+ish2,j+jsh2]==2 ## 2==accepted
                        testval2 = ttime[i+ish2,j+jsh2]
                        ## pick the lowest value of the two
                        ## <=, compare to chosenval 1, *not* 2!!
                        ## This because the direction has already been chosen
                        ##  at the line "testval1<chosenval1"
                        if testval2<=chosenval1 
                            chosenval2=testval2
                            use2ndord=true

                            # save derivative choices
                            axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)

                        else
                            chosenval2=HUGE
                            use2ndord=false # this is needed!

                            # save derivative choices
                            axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                            
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
            
            codeD[:] .= 0.0

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
                            axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

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


"""
$(TYPEDSIGNATURES)

 Projection operator P containing interpolation coefficients, ordered according to FMM order.
"""
function calcprojttfmmord(ttime,grd,idxconv,coordrec)

    
    # calculate the coefficients and their indices using bilinear interpolation
    nrec = size(coordrec,1)
    Ncoe = 4
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
            iorig = cart2lin(i,j,idxconv.nx)
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

    P = sparse(P_i,P_j,P_v,nrec,grd.nx*grd.ny)
    return P
end

##############################################################################


"""
$(TYPEDSIGNATURES)

 Solve the discrete adjoint equations and return the gradient of the misfit.
"""
function discradjoint_hiord_SINGLESRC(idxconv,tt,Dx,Dy,P,pickobs,stdobs,vel2d)
    #                                                                       #
    # * * * ALL stuff must be in FMM order (e.g., fmmord.ttime) !!!! * * *  #
    #                                                                       #

    rhs = - transpose(P) * ( ((P*tt).-pickobs)./stdobs.^2)
    ## WARNING! Using copy(transpose(...)) to MATERIALIZE the transpose, otherwise
    ##   the solver (\) does not use the correct sparse algo for matrix division
    tmplhs = copy(transpose( (2.0 .* Diagonal(Dx*tt) * Dx) .+ (2.0 .* Diagonal(Dy*tt) * Dy) ))
    lhs = UpperTriangular(tmplhs)
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
    gradvec = 2.0 .* lambda ./ vec(vel2d).^3

    grad2d = reshape(gradvec,idxconv.nx,idxconv.ny)

    # end
    grad2d[:,:] .= grad2d
    return grad2d
end

##############################################################################

"""
$(TYPEDSIGNATURES)

  Refinement of the grid around the source. FMM calculated inside a finer grid 
    and then passed on to coarser grid
"""
function ttaroundsrc_discradj!(statuscoarse::Array{Int64,2},ttimecoarse::Array{Float64,2},
                               vel::Array{Float64,2},src::Vector{Float64},grdcoarse::Grid2D,inittt::Float64, 
                               idxconv, fmmord )
    
    ##
    ## 2x10 nodes -> 2x50 nodes
    ##
    downscalefactor::Int = 5
    noderadius::Int = 3 # from 5 to 3 for discrete adjoint to avoid too many artifacts...

    ##
    ijpt = MVector(0,0)

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    ixsrcglob,iysrcglob = findclosestnode(src[1],src[2],grdcoarse.xinit,grdcoarse.yinit,grdcoarse.hgrid) 
    
    ##
    ## Define chunck of coarse grid
    ##
    i1coarsevirtual = ixsrcglob - noderadius
    i2coarsevirtual = ixsrcglob + noderadius
    j1coarsevirtual = iysrcglob - noderadius
    j2coarsevirtual = iysrcglob + noderadius
    # if hitting borders
    outxmin = i1coarsevirtual<1
    outxmax = i2coarsevirtual>grdcoarse.nx
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>grdcoarse.ny
    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=grdcoarse.nx : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=grdcoarse.ny : j2coarse=j2coarsevirtual

    ##
    ## Refined grid parameters
    ##
    dh = grdcoarse.hgrid/downscalefactor
    # fine grid size
    nx = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    ny = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    # cart2lin_fine = LinearIndices((nx,ny))
    # lin2cart2_fine = CartesianIndices((nx,ny))

    # set origin of the fine grid
    xinit = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    yinit = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    grdfine = Grid2D(hgrid=dh,xinit=xinit,yinit=yinit,nx=nx,ny=ny)

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
   
    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)
    
    ##
    ## Reset coodinates to match the fine grid
    ##
    # xorig = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    # yorig = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    xsrc = src[1] #- xorig #- grdcoarse.xinit
    ysrc = src[2] #- yorig #- grdcoarse.yinit
    srcfine = Float64[xsrc,ysrc]

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,nx,ny)
    for j=1:ny
        for i=1:nx
            di=div(i-1,downscalefactor)
            ri=i-di*downscalefactor
            ii = ri>=downscalefactor/2+1 ? di+2 : di+1

            dj=div(j-1,downscalefactor)
            rj=j-dj*downscalefactor
            jj = rj>=downscalefactor/2+1 ? dj+2 : dj+1

            velfinegrd[i,j] = velcoarsegrd[ii,jj]
        end
    end
    
    ##
    ## Source location, etc. within fine grid
    ##  
    ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
    onsrc = sourceboxloctt!(ttime,velfinegrd,srcfine,grdfine, staggeredgrid=false )
   
    ######################################################
  
    neigh = [1  0;
             0  1;
            -1  0;
             0 -1]

    #-------------------------------
    ## init FMM 
    status[onsrc] .= 2 ## set to accepted on src
    #naccinit=count(status.==2)

    ## get all i,j accepted
    ijss = findall(status.==2) 
    is = [l[1] for l in ijss]
    js = [l[2] for l in ijss]
    naccinit = length(ijss)

    ##===================================
    for l=1:naccinit
        ia = is[l]
        ja = js[l]
        # if the node coincides with the coarse grid, store values
        oncoa,ia_coarse,ja_coarse = isacoarsegridnode(ia,ja,downscalefactor,i1coarse,j1coarse)
        if oncoa
            ##===================================
            ## UPDATE the COARSE GRID
            ttimecoarse[ia_coarse,ja_coarse]  = ttime[ia,ja]
            statuscoarse[ia_coarse,ja_coarse] = status[ia,ja]
            ##==================================
        end
    end

    ##==========================================================##
    # discrete adjoint

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
    node_coarse = 1
    for l=1:naccinit
        # ordered index
        p = spidx[l]

        ## if the point belongs to the coarse grid, add it
        oncoa,ia_coarse,ja_coarse = isacoarsegridnode(is[p],js[p],downscalefactor,i1coarse,j1coarse)
        if oncoa
            # go from cartesian (i,j) to linear
            idxconv.lfmm2grid[node_coarse] = cart2lin(ia_coarse,ja_coarse,idxconv.nx)

            # store arrival time for first points in FMM order
            fmmord.ttime[node_coarse] = ttime[is[p],js[p]]
            
            node_coarse +=1
        end
        #@show is[p],js[p],ttime[is[p],js[p]],idxconv.lfmm2grid[l]
    end

    ##==========================================================##
    
    ## Init the min binary heap with void arrays but max size
    Nmax=nx*ny
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
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                # get handle
                # han = sub2ind((nx,ny),i,j)
                han = cart2lin(i,j,nx)
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1
                
            end            
        end
    end

    #-------------------------------
    ## main FMM loop
    firstwarning=true
    totnpts = nx*ny
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nx,ny),han)
        lin2cart!(han,nx,ijpt)
        ia,ja = ijpt[1],ijpt[2]
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt

        ##===================================
        oncoa,ia_coarse,ja_coarse = isacoarsegridnode(ia,ja,downscalefactor,i1coarse,j1coarse)
        if oncoa

            ##===================================
            ## UPDATE the COARSE GRID
            ttimecoarse[ia_coarse,ja_coarse] = tmptt
            statuscoarse[ia_coarse,ja_coarse] = 2
            ##===================================

            ##===================================
            # Discrete adjoint
            # store the linear index of FMM order
            idxconv.lfmm2grid[node_coarse] = cart2lin(ia_coarse,ja_coarse,idxconv.nx)
            # store arrival time for first points in FMM order
            fmmord.ttime[node_coarse] = tmptt
            # increase the counter
            node_coarse += 1
        end
        ##===================================


        ##########################################################
        ##
        ## If the the accepted point is on the edge of the
        ##  fine grid, stop computing and jump to coarse grid
        ##
        ##########################################################
        #  if (ia==nx) || (ia==1) || (ja==ny) || (ja==1)
        if (ia==1 && !outxmin) || (ia==nx && !outxmax) || (ja==1 && !outymin) || (ja==ny && !outymax)
            # ttimecoarse[i1coarse:i2coarse,j1coarse:j2coarse]  =  ttime[1:downscalefactor:end,1:downscalefactor:end]
            # statuscoarse[i1coarse:i2coarse,j1coarse:j2coarse] = status[1:downscalefactor:end,1:downscalefactor:end]
            ## delete current narrow band to avoid problems when returned to coarse grid
            statuscoarse[statuscoarse.==1] .= 0

            ## Prevent the difficult case of traveltime hitting the borders but
            ##   not the coarse grid, which would produce an empty "statuscoarse" and an empty "ttimecoarse".
            ## Probably needs a better fix..."
            if count(statuscoarse.>0)<1
                if firstwarning 
                    @warn("Traveltime hitting the borders but not the coarse grid, continuing.")
                    firstwarning=false
                end
                continue
            end
            return nothing
        end
        ##########################################################

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
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                han = cart2lin(i,j,nx)
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                
                
            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                # get handle
                han = cart2lin(i,j,nx)
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------
    end
    error("ttaroundsrc_discradj!(): Ouch...")
end


#######################################################################################

function isacoarsegridnode(i::Int,j::Int,downscalefactor::Int,i1coarse::Int,j1coarse)
    a = rem(i, downscalefactor)
    b = rem(j, downscalefactor)
    if a==b==0
        icoa = div(i,downscalefactor)+i1coarse
        jcoa = div(j,downscalefactor)+j1coarse
        return true,icoa,jcoa
    else
        return false,nothing,nothing
    end
    return
end

#######################################################################################

"""
$(TYPEDSIGNATURES)

 Attempt to reconstruct how derivatives have been calculated in the area of 
  the refinement of the source. Updates the codes for derivatives used to construct
   Dx and Dy.
"""
function derivaroundsrcfmm!(lseq,idxconv,codeD)
    # lipt = index in fmmord in sequential order (1,2,3,4,...)
    
    nx = idxconv.nx
    ny = idxconv.ny

    ijpt = MVector(0,0)
    idxpt = idxconv.lfmm2grid[lseq]
    lin2cart!(idxpt,nx,ijpt)
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
                l = cart2lin(i,j,nx)
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
                        l = cart2lin(i,j,nx)
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
