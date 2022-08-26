

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################

##########################################################################

"""
$(TYPEDSIGNATURES)

Calculate traveltime for 2D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 2D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime2D(vel::Array{Float64,2},grd::Union{Grid2D,Grid2DSphere},coordsrc::Array{Float64,2},
                      coordrec::Vector{Array{Float64,2}} ; returntt::Bool=false) 
        
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
    ttpicks = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicks[i] = zeros(curnrec)
    end

    ## calculate how to subdivide the srcs among the workers
    #nsrc = size(coordsrc,1)
    nw = nworkers()
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()

    if returntt
        # return traveltime array and picks at receivers
        ttime = zeros(n1,n2,nsrc)
    end

    @sync begin

        if returntt 
            # return traveltime picks at receivers and at all grid points 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                                        vel,coordsrc[igrs,:],
                                                                        coordrec[igrs],grd,
                                                                        returntt=returntt )
            end

        else
            # return ONLY traveltime picks at receivers 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                        vel,coordsrc[igrs,:],
                                                        coordrec[igrs],grd,
                                                        returntt=returntt )
            end
        end
        
    end # sync

    if returntt 
        return ttpicks,ttime
    end
    return ttpicks
end

###########################################################################
"""
$(TYPEDSIGNATURES)

  Compute the forward problem for a group of sources.
"""
function ttforwsomesrc2D(vel::Array{Float64,2},coordsrc::Array{Float64,2},
                         coordrec::Vector{Array{Float64,2}},grd::Union{Grid2D,Grid2DSphere} ;
                         returntt::Bool=false )
    
    if typeof(grd)==Grid2D
        #simtype = :cartesian
        n1,n2 = grd.nx,grd.ny
    elseif typeof(grd)==Grid2DSphere
        #simtype = :spherical
        n1,n2 = grd.nr,grd.nθ
    end

    nsrc = size(coordsrc,1)
    
    ttpicksGRPSRC = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicksGRPSRC[i] = zeros(curnrec)
    end

    ttGRPSRC = zeros(n1,n2,nsrc)

    ## group of pre-selected sources
    for s=1:nsrc    

        ## Compute traveltime and interpolation at receivers in one go for parallelization
        ttGRPSRC[:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grd)

        ## interpolate at receivers positions
        for i=1:size(coordrec[s],1)
            ttpicksGRPSRC[s][i] = bilinear_interp(ttGRPSRC[:,:,s],grd,coordrec[s][i,1],
                                                  coordrec[s][i,2])
        end
    end

    if returntt 
        return ttGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC 
end


#################################################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(ttime::Array{Float64,2},vel::Array{Float64,2},srcpos::AbstractVector,
                         grd::Grid2D; staggeredgrid::Bool )
    ## staggeredgrid keyword required!
    
    ## source location, etc.      
    mindistsrc = 1e-5   
    xsrc,ysrc=srcpos[1],srcpos[2]

    if staggeredgrid==false
        ## regular grid
        onsrc = zeros(Bool,grd.nx,grd.ny)
        onsrc[:,:] .= false
        ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)

    elseif staggeredgrid==true
        ## STAGGERED grid
        onsrc = zeros(Bool,grd.ntx,grd.nty)
        onsrc[:,:] .= false
        ## grd.xinit-hgr because TIME array on STAGGERED grid
        hgr = grd.hgrid/2.0
        ix,iy = findclosestnode(xsrc,ysrc,grd.xinit-hgr,grd.yinit-hgr,grd.hgrid)
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
    end

    halfg = 0.0 #hgrid/2.0
    # Euclidean distance
    dist = sqrt(rx^2+ry^2)
    #@show dist,src,rx,ry
    if dist<=mindistsrc
        onsrc[ix,iy] = true
        ttime[ix,iy] = 0.0 
    else
        if (rx>=halfg) & (ry>=halfg)
            onsrc[ix:ix+1,iy:iy+1] .= true
        elseif (rx<halfg) & (ry>=halfg)
            onsrc[ix-1:ix,iy:iy+1] .= true
        elseif (rx<halfg) & (ry<halfg)
            onsrc[ix-1:ix,iy-1:iy] .= true
        elseif (rx>=halfg) & (ry<halfg)
            onsrc[ix:ix+1,iy-1:iy] .= true
        end
        
        ## set ttime around source ONLY FOUR points!!!
        ijsrc = findall(onsrc)
        for lcart in ijsrc
            i = lcart[1]
            j = lcart[2]
            if staggeredgrid==false
                ## regular grid
                xp = (i-1)*grd.hgrid+grd.xinit
                yp = (j-1)*grd.hgrid+grd.yinit
                ii = Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
                jj = Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1             
                ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]
            elseif staggeredgrid==true
                ## STAGGERED grid
                ## grd.xinit-hgr because TIME array on STAGGERED grid
                xp = (i-1)*grd.hgrid+grd.xinit-hgr
                yp = (j-1)*grd.hgrid+grd.yinit-hgr
                ii = Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1 # i-1
                jj = Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1 # j-1            
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]
            end
        end
    end
    
    return onsrc
end 

#################################################################################

#########################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 2D using traditional stencils on regular grid. 
"""
function ttFMM_hiord(vel::Array{Float64,2},src::Vector{Float64},grd::Union{Grid2D,Grid2DSphere} ;
                     dodiscradj::Bool=false)
                     
    ## Sizes
    if typeof(grd)==Grid2D
        simtype = :cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype = :spherical
    end
    if simtype==:cartesian
        n1,n2 = grd.nx,grd.ny #size(vel)  ## NOT A STAGGERED GRID!!!
    elseif simtype==:spherical
        n1,n2 = grd.nr,grd.nθ
    end
    ## 
    epsilon = 1e-6
    n12 = n1*n2
    
    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,n1,n2)
    ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,n1,n2)
    status[:,:] .= 0   ## set all to far
    ##
    ptij = MVector(0,0)
    # derivative codes
    idD = MVector(0,0)

    ##======================================================
    if dodiscradj
        ##
        ##  discrete adjoint: init stuff
        ## 
        idxconv = MapOrderGridFMM2D(n1,n2)
        fmmord = VarsFMMOrder2D(n1,n2)
        codeDxy = zeros(Int64,n12,2)
    end    
    ##======================================================
    
    ##########################################
    
    if extrapars.refinearoundsrc
        ##---------------------------------
        ## 
        ## Refinement around the source      
        ##
        if dodiscradj
            ttaroundsrc!(status,ttime,vel,src,grd,inittt,
                         dodiscradj=dodiscradj,idxconv=idxconv,fmmord=fmmord)
        else
            ttaroundsrc!(status,ttime,vel,src,grd,inittt)
        end

        ## get all i,j accepted
        ijss = findall(status.==2) 
        is = [l[1] for l in ijss]
        js = [l[2] for l in ijss]
        naccinit = length(ijss)

        ##======================================================
        if dodiscradj
            ## 
            ## DISCRETE ADJOINT WORKAROUND FOR DERIVATIVES
            ##

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
                derivaroundsrcfmm2D!(l,idxconv,idD)

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

        end # dodiscradj
        ##======================================================

  
    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        #println("ttFMM_hiord(): NO refinement around the source! ")

        ## source location, etc.      
        ## REGULAR grid
        if simtype==:cartesian
            onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=false )
        elseif simtype==:spherical
            onsrc = sourceboxloctt_sph!(ttime,vel,src,grd )
        end

        ##
        ## Status of nodes
        status[onsrc] .= 2 ## set to accepted on src

        ## get all i,j accepted
        ijss = findall(status.==2) 
        is = [l[1] for l in ijss]
        js = [l[2] for l in ijss]
        naccinit = length(ijss)
        

        ##======================================================
        if dodiscradj

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
                idxconv.lfmm2grid[l] = cart2lin2D(is[p],js[p],n1)

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

            end
        end # if dodiscradj
        ##======================================================
      
    end # if refinearoundsrc
    ###################################################### 

    #-------------------------------
    ## init FMM 
    neigh = SA[1  0;
               0  1;
              -1  0;
               0 -1]

    ## Init the min binary heap with void arrays but max size
    Nmax = n1*n2
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 

    ##-----------------------------------------
    ## construct initial narrow band
    for l=1:naccinit ##
        
        for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]

            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,vel,grd,status,i,j,idD)

                # get handle
                han = cart2lin2D(i,j,n1)
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    codeDxy[han,:] .= idD
                end
            end
        end
    end

    #-------------------------------
    ## main FMM loop
    totnpts = n1*n2
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((n1,n2),han)
        lin2cart2D!(han,n1,ptij)
        ia,ja = ptij[1],ptij[2]
        #ja = div(han-1,n1) +1
        #ia = han - n1*(ja-1)
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt


        if dodiscradj
            # store the linear index of FMM order
            idxconv.lfmm2grid[node] = cart2lin2D(ia,ja,n1)
            # store arrival time for first points in FMM order
            fmmord.ttime[node] = tmptt  
        end

        
        ## try all neighbors of newly accepted point
        for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,vel,grd,status,i,j,idD)
                han = cart2lin2D(i,j,n1)
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    codeDxy[han,:] .= idD
                end

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord!(ttime,vel,grd,status,i,j,idD)

                # get handle
                han = cart2lin2D(i,j,n1)
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    codeDxy[han,:] .= idD
                end

            end
        end
        ##-------------------------------
    end

    ##======================================================
    if dodiscradj

        ## pre-compute the mapping between fmm and orig order
        for i=1:n12
            ifm = idxconv.lfmm2grid[i]
            idxconv.lgrid2fmm[ifm] = i
        end

        # pre-determine derivative coefficients for positive codes (0,+1,+2)
        if simtype==:cartesian
            hgrid = grd.hgrid
            #allcoeff = [[-1.0/hgrid, 1.0/hgrid], [-3.0/(2.0*hgrid), 4.0/(2.0*hgrid), -1.0/(2.0*hgrid)]]
            allcoeffx = CoeffDerivCartesian( MVector(-1.0/hgrid,
                                                       1.0/hgrid), 
                                               MVector(-3.0/(2.0*hgrid),
                                                       4.0/(2.0*hgrid),
                                                       -1.0/(2.0*hgrid)) )
            # coefficients along X are the same than along Y
            allcoeffy = allcoeffx

        elseif simtype==:spherical
            Δr = grd.Δr
            ## DEG to RAD !!!!
            Δarc = [grd.r[i] * deg2rad(grd.Δθ) for i=1:grd.nr]
            # coefficients
            coe_r_1st = [-1.0/Δr  1.0/Δr]
            coe_r_2nd = [-3.0/(2.0*Δr)  4.0/(2.0*Δr) -1.0/(2.0*Δr) ]
            coe_θ_1st = [-1.0 ./ Δarc  1.0 ./ Δarc ]
            coe_θ_2nd = [-3.0./(2.0*Δarc)  4.0./(2.0*Δarc)  -1.0./(2.0*Δarc) ]
            #@show size(coe_θ_1st),size(coe_θ_2nd)

            allcoeffx = CoeffDerivSpherical2D( coe_r_1st, coe_r_2nd )
            allcoeffy = CoeffDerivSpherical2D( coe_θ_1st, coe_θ_2nd )

        end

        #@time begin 

        ## set the derivative operators
        for irow=1:n12
            
            # compute the coefficients for X  derivatives
            setcoeffderiv2D!(fmmord.vecDx,irow,idxconv,codeDxy,allcoeffx,ptij,
                           axis=:X,simtype=simtype)
            
            # compute the coefficients for Y derivatives
            setcoeffderiv2D!(fmmord.vecDy,irow,idxconv,codeDxy,allcoeffy,ptij,
                           axis=:Y,simtype=simtype)
            
        end

        #end # @time
        
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
        
        

        ## return all the stuff for discrete adjoint computations
        return ttime,idxconv,fmmord.ttime,Dx_fmmord,Dy_fmmord
    end # if dodiscradj
    ##======================================================
    
    return ttime
end

##########################################################################

"""
$(TYPEDSIGNATURES)

 Test if point is on borders of domain.
"""
function isonbord(ib::Int64,jb::Int64,n1::Int64,n2::Int64)
    isonb1st = false
    isonb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>n1 || jb<1 || jb>n2
        isonb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>n1-1 || jb<2 || jb>n2-1
        isonb2nd =  true
    end
    return isonb1st,isonb2nd
end

##########################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order. 
   Two-dimensional Cartesian or  spherical grid depending on the type of 'grd'.
"""
function calcttpt_2ndord!(ttime::Array{Float64,2},vel::Array{Float64,2},
                         grd::Union{Grid2D,Grid2DSphere},status::Array{Int64,2},
                         i::Int64,j::Int64,codeD::MVector{2,Int64})
    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    if typeof(grd)==Grid2D
        simtype=:cartesian
    elseif typeof(grd)==Grid2DSphere
        simtype=:spherical
    end

    # sizes, etc.
    if simtype==:cartesian
        n1 = grd.nx
        n2 = grd.ny
        Δh = MVector(grd.hgrid,grd.hgrid)
    elseif simtype==:spherical
        n1 = grd.nr
        n2 = grd.nθ
        Δh = MVector(grd.Δr, grd.r[i]*deg2rad(grd.Δθ))
    end
    # slowness
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
    codeD[:] .= 0.0 

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
            isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                                    
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

        ## spacing
        deltah = Δh[axis]
                
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
                    isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                    
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

                ## spacing
                deltah = Δh[axis]
                
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

################################################################################

"""
$(TYPEDSIGNATURES)

  Refinement of the grid around the source. FMM calculated inside a finer grid 
    and then passed on to coarser grid
"""
function ttaroundsrc!(statuscoarse::Array{Int64,2},ttimecoarse::Array{Float64,2},
                      vel::Array{Float64,2},src::Vector{Float64},grdcoarse::Union{Grid2D,Grid2DSphere},
                      inittt::Float64 ;                      
                      dodiscradj::Bool=false,idxconv::Union{MapOrderGridFMM2D,Nothing}=nothing,
                      fmmord::Union{VarsFMMOrder2D,Nothing}=nothing )

    
    ##
    if typeof(grdcoarse)==Grid2D
        simtype=:cartesian
    elseif typeof(grdcoarse)==Grid2DSphere
        simtype=:spherical
    end

    ##
    ## 2x10 nodes -> 2x50 nodes
    ##
    downscalefactor::Int = 0
    noderadius::Int = 0
    if dodiscradj
        downscalefactor = 5
        ## 2 instead of 5 for adjoint to avoid messing up...
        noderadius = 2 
    else
        downscalefactor = 5
        noderadius = 5
    end        

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    if simtype==:cartesian
        n1_coarse,n2_coarse = grdcoarse.nx,grdcoarse.ny
        ixsrcglob,iysrcglob = findclosestnode(src[1],src[2],grdcoarse.xinit,grdcoarse.yinit,grdcoarse.hgrid) 
    elseif simtype==:spherical
        n1_coarse,n2_coarse = grdcoarse.nr,grdcoarse.nθ
        ixsrcglob,iysrcglob = findclosestnode_sph(src[1],src[2],grdcoarse.rinit,grdcoarse.θinit,grdcoarse.Δr,grdcoarse.Δθ) 
    end
  
    ##
    ## Define chunck of coarse grid
    ##
    i1coarsevirtual = ixsrcglob - noderadius
    i2coarsevirtual = ixsrcglob + noderadius
    j1coarsevirtual = iysrcglob - noderadius
    j2coarsevirtual = iysrcglob + noderadius
    # if hitting borders
    outxmin = i1coarsevirtual<1
    outxmax = i2coarsevirtual>n1_coarse
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>n2_coarse
    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=n1_coarse : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=n2_coarse : j2coarse=j2coarsevirtual
    
    ##
    ## Refined grid parameters
    ##
    # fine grid size
    n1 = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    n2 = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)    

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,n1,n2)
    for j=1:n2
        for i=1:n1
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
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,n1,n2)
    ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,n1,n2)
    status[:,:] .= 0   ## set all to far
    # derivative codes
    idD = MVector(0,0)

    ##
    ## Reset coodinates to match the fine grid
    ##
    # xorig = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    # yorig = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    xsrc = src[1] #- xorig #- grdcoarse.xinit
    ysrc = src[2] #- yorig #- grdcoarse.yinit
    srcfine = MVector(xsrc,ysrc)

    
    if simtype==:cartesian
        # set origin of the fine grid
        xinit = grdcoarse.x[i1coarse]
        yinit = grdcoarse.y[j1coarse]
        dh = grdcoarse.hgrid/downscalefactor
        # fine grid
        grdfine = Grid2D(hgrid=dh,xinit=xinit,yinit=yinit,nx=n1,ny=n2)
        ##
        ## Source location, etc. within fine grid
        ##  
        ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
        onsrc = sourceboxloctt!(ttime,velfinegrd,srcfine,grdfine, staggeredgrid=false )
        
    elseif simtype==:spherical
        # set origin of the fine grid
        rinit = grdcoarse.r[i1coarse]
        θinit = grdcoarse.θ[j1coarse]
        dr = grdcoarse.Δr/downscalefactor
        dθ = grdcoarse.Δθ/downscalefactor
        # fine grid
        grdfine = Grid2DSphere(Δr=dr,Δθ=dθ,nr=n1,nθ=n2,rinit=rinit,θinit=θinit)
        ##
        ## Source location, etc. within fine grid
        ##  
        ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
        onsrc = sourceboxloctt_sph!(ttime,velfinegrd,srcfine,grdfine)

    end
    
    ######################################################

    ptij = MVector(0,0)

    neigh = SA[1  0;
               0  1;
              -1  0;
               0 -1]

    #-------------------------------
    ## init FMM 

    status[onsrc] .= 2 ## set to accepted on src
    naccinit=count(status.==2)

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
    if dodiscradj       

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
                idxconv.lfmm2grid[node_coarse] = cart2lin2D(ia_coarse,ja_coarse,idxconv.nx)

                # store arrival time for first points in FMM order
                fmmord.ttime[node_coarse] = ttime[is[p],js[p]]
                
                node_coarse +=1
            end
            #@show is[p],js[p],ttime[is[p],js[p]],idxconv.lfmm2grid[l]
        end
    end # if dodiscradj
    ##==========================================================##


    ## Init the min binary heap with void arrays but max size
    Nmax=n1*n2
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## construct initial narrow band
    for l=1:naccinit ##
        
         for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,velfinegrd,grdfine,status,i,j,idD)
                # get handle
                han = cart2lin2D(i,j,n1)
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
    totnpts = n1*n2
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((n1,n2),han)
        cija = lin2cart2D!(han,n1,ptij)
        ia,ja = ptij[1],ptij[2]
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
            if dodiscradj
                # store the linear index of FMM order
                idxconv.lfmm2grid[node_coarse] = cart2lin2D(ia_coarse,ja_coarse,idxconv.nx)
                # store arrival time for first points in FMM order
                fmmord.ttime[node_coarse] = tmptt
                # increase the counter
                node_coarse += 1
            end
        end
        ##===================================

        ##########################################################
        ##
        ## If the the accepted point is on the edge of the
        ##  fine grid, stop computing and jump to coarse grid
        ##
        ##########################################################
        #  if (ia==n1) || (ia==1) || (ja==n2) || (ja==1)
        if (ia==1 && !outxmin) || (ia==n1 && !outxmax) || (ja==1 && !outymin) || (ja==n2 && !outymax)
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
            if (i>n1) || (i<1) || (j>n2) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,velfinegrd,grdfine,status,i,j,idD)
                han = cart2lin2D(i,j,n1)
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord!(ttime,velfinegrd,grdfine,status,i,j,idD)
                # get handle
                han = cart2lin2D(i,j,n1)
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------
    end
    error("ttaroundsrc!(): Ouch...")
end


#######################################################################################

#################################################################################

"""
$(TYPEDSIGNATURES)
"""
function sourceboxloctt_sph!(ttime::Array{Float64,2},vel::Array{Float64,2},srcpos::AbstractVector,
                             grd::Grid2DSphere )

    ## source location, etc.      
    mindistsrc = 1e-5   
    rsrc,θsrc=srcpos[1],srcpos[2]

    ## regular grid
    onsrc = zeros(Bool,grd.nr,grd.nθ)
    onsrc[:,:] .= false
    ix,iy = findclosestnode_sph(rsrc,θsrc,grd.rinit,grd.θinit,grd.Δr,grd.Δθ)
    rr = rsrc-((ix-1)*grd.Δr+grd.rinit)
    rθ = θsrc-((iy-1)*grd.Δθ+grd.θinit)

    halfg = 0.0 #hgrid/2.0
    ## distance in POLAR COORDINATES
    ## sqrt(r1^+r2^2 - 2*r1*r2*cos(θ1-θ2))
    r1=rsrc
    r2=grd.r[ix]
    dist = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[iy]) )  #sqrt(rr^2+grd.r[ix]^2*rθ^2)
    #@show dist,src,rr,rθ
    if dist<=mindistsrc
        onsrc[ix,iy] = true
        ttime[ix,iy] = 0.0 
    else
        if (rr>=halfg) & (rθ>=halfg)
            onsrc[ix:ix+1,iy:iy+1] .= true
        elseif (rr<halfg) & (rθ>=halfg)
            onsrc[ix-1:ix,iy:iy+1] .= true
        elseif (rr<halfg) & (rθ<halfg)
            onsrc[ix-1:ix,iy-1:iy] .= true
        elseif (rr>=halfg) & (rθ<halfg)
            onsrc[ix:ix+1,iy-1:iy] .= true
        end
        
        ## set ttime around source ONLY FOUR points!!!
        ijsrc = findall(onsrc)
        for lcart in ijsrc
            i = lcart[1]
            j = lcart[2]
            ## regular grid
            # xp = (i-1)*grd.Δr+grd.rinit
            # yp = (j-1)*grd.Δθ+grd.θinit
            ii = Int(floor((rsrc-grd.rinit)/grd.Δr)) +1
            jj = Int(floor((θsrc-grd.θinit)/grd.Δθ)) +1
            ##ttime[i,j] = sqrt((rsrc-xp)^2+(θsrc-yp)^2) / vel[ii,jj]
            ## sqrt(r1^+r2^2 -2*r1*r2*cos(θ1-θ2))
            r1=rsrc
            r2=grd.r[ii]
            distp = sqrt(r1^2+r2^2-2.0*r1*r2*cosd(θsrc-grd.θ[iy]))
            ttime[i,j] = distp / vel[ii,jj]
        end
    end

    return onsrc
end 
#########################################################
#end                                                    #
#########################################################

