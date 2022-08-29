

#######################################################
##        Eikonal forward 3D                         ## 
#######################################################

##using StaticArrays


###########################################################################

"""
$(TYPEDSIGNATURES)

Calculate traveltime for 3D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 3D velocity model 
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y,z), a 3-column array 
- `coordrec`: the coordinates of the receiver(s) (x,y,z), a 3-column array
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime3D(vel::Array{Float64,3},grd::GridEik3D,coordsrc::Array{Float64,2},
                      coordrec::Vector{Array{Float64,2}}; returntt::Bool=false) 
    
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

    ##------------------
    ## parallel version
    nsrc = size(coordsrc,1)
        ttpicks = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicks[i] = zeros(curnrec)
    end    

    ## calculate how to subdivide the srcs among the workers
    nw = nworkers()
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()

    if returntt
        ttime = zeros(n1,n2,n3,nsrc)
    end

    @sync begin

        if returntt
            # return both traveltime picks at receivers and at all grid points
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,:,igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc3D,wks[s],
                                                                          vel,coordsrc[igrs,:],
                                                                          coordrec[igrs],grd,
                                                                          returntt=returntt )
            end

        else
            # return ONLY traveltime picks at receivers
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc3D,wks[s],
                                                        vel,coordsrc[igrs,:],
                                                        coordrec[igrs],grd,
                                                        returntt=returntt )
            end
        end

    end

    if returntt
        return ttpicks,ttime
    end
    return ttpicks
end


#########################################################################

"""
$(TYPEDSIGNATURES)

  Compute the forward problem for a group of sources.
"""
function ttforwsomesrc3D(vel::Array{Float64,3},coordsrc::Array{Float64,2},
                      coordrec::Vector{Array{Float64,2}},grd::GridEik3D ;
                      returntt::Bool=false )
    
    if typeof(grd)==Grid3D
        #simtype = :cartesian
        n1,n2,n3 = grd.nx,grd.ny,grd.nz #size(vel)  ## NOT A STAGGERED GRID!!!
    elseif typeof(grd)==Grid3DSphere
        #simtype = :spherical
        n1,n2,n3 = grd.nr,grd.nθ,grd.nφ
    end
    
    nsrc = size(coordsrc,1)

    ttpicksGRPSRC = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicksGRPSRC[i] = zeros(curnrec)
    end

    ttimeGRPSRC = zeros(n1,n2,n3,nsrc)
    
    ## group of pre-selected sources
    for s=1:nsrc
        ttimeGRPSRC[:,:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grd)
        
        ## Interpolate at receivers positions
        for i=1:size(coordrec[s],1)            
            ttpicksGRPSRC[s][i] = trilinear_interp( ttimeGRPSRC[:,:,:,s], grd,
                                                    coordrec[s][i,1],coordrec[s][i,2],
                                                    coordrec[s][i,3])
        end
    end

    if returntt
        return ttimeGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC
end

#############################################################################

"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 3D using traditional stencils on regular grid. 
"""
function ttFMM_hiord(vel::Array{Float64,3},src::Vector{Float64},grd::GridEik3D ;
                    dodiscradj::Bool=false ) 

    ## Sizes
    ## Sizes
    if typeof(grd)==Grid3D
        simtype = :cartesian
    elseif typeof(grd)==Grid3DSphere
        simtype = :spherical
    end
     if simtype==:cartesian
        n1,n2,n3 = grd.nx,grd.ny,grd.nz #size(vel)  ## NOT A STAGGERED GRID!!!
    elseif simtype==:spherical
        n1,n2,n3 = grd.nr,grd.nθ,grd.nφ
    end

    epsilon = 1e-6
    n123 = n1*n2*n3
    
    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,n1,n2,n3)
    ttime[:,:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,n1,n2,n3)
    status[:,:,:] .= 0   ## set all to far
    
    ptijk = MVector(0,0,0)
    # derivative codes
    idD = MVector(0,0,0)

    ##======================================================
    if dodiscradj
        ##
        ##  discrete adjoint: init stuff
        ## 
        idxconv = MapOrderGridFMM3D(n1,n2,n3)
        fmmord = VarsFMMOrder3D(n1,n2,n3)
        codeDxyz = zeros(Int64,n123,3)
    end    
    ##======================================================

    ##########################################
    ## refinearoundsrc=true

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
        ks = [l[3] for l in ijss]
        naccinit = length(ijss)

        ##======================================================
        if dodiscradj
            ## 
            ## DISCRETE ADJOINT WORKAROUND FOR DERIVATIVES
            ##

            # How many initial points to skip, considering them as "onsrc"?
            skipnptsDxyz = 4
            
            ## pre-compute some of the mapping between fmm and orig order
            for i=1:naccinit
                ifm = idxconv.lfmm2grid[i]
                idxconv.lgrid2fmm[ifm] = i
            end

            for l=1:naccinit

                if l<=skipnptsDxyz
                    #################################################################
                    # Here we store a 1 in the diagonal because we are on a source node...
                    #  store arrival time for first points in FMM order
                    addentry!(fmmord.vecDx,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                    addentry!(fmmord.vecDy,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                    addentry!(fmmord.vecDz,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                    #################################################################
                end

                ## "reconstruct" derivative stencils from known FMM order and arrival times
                derivaroundsrcfmm3D!(l,idxconv,idD)

                if idD==[0,0,0] # 3 elements!!!
                    #################################################################
                    # Here we store a 1 in the diagonal because we are on a source node...
                    #  store arrival time for first points in FMM order
                    addentry!(fmmord.vecDx,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                    addentry!(fmmord.vecDy,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                    addentry!(fmmord.vecDz,l,l,1.0)      ## <<<<<<<<<<<<<========= CHECK this! =============#####
                    #################################################################
                else
                    l_fmmord = idxconv.lfmm2grid[l]
                    codeDxyz[l_fmmord,:] .= idD
                end
            end

        end # dodiscradj
        ##======================================================


    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        #println("\nttFMM_hiord(): NO refinement around the source! \n")

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
        ks = [l[3] for l in ijss]
        naccinit = length(ijss)
        

        ##======================================================
        if dodiscradj
            
            # discrete adjoint: first visited points in FMM order
            ttfirstpts = Vector{Float64}(undef,naccinit)
            for l=1:naccinit
                i,j,k = is[l],js[l],ks[l]
                ttij = ttime[i,j,k]
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
                idxconv.lfmm2grid[l] = cart2lin3D(is[p],js[p],ks[p],n1,n2)

                ######################################
                # store arrival time for first points in FMM order
                ttgrid = ttime[is[p],js[p],ks[p]]
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
                addentry!(fmmord.vecDz,l,l,1.0)                   ## <<<<<<<<<<<<<========= CHECK this! =============#####
                #
                #################################################################

            end
        end # if dodiscradj
        ##======================================================
              
    end # if refinearoundsrc

    ######################################################################################
    
    ##===========================================================================================    
    #-------------------------------
    ## init FMM 
    neigh = SA[1  0  0;
               0  1  0;
              -1  0  0;
               0 -1  0;
               0  0  1;
               0  0 -1]


    ## Init the max binary heap with void arrays but max size
    Nmax=n1*n2*n3
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 

    ## construct initial narrow band
    for l=1:naccinit ##        
        for ne=1:6 ## six potential neighbors
            
            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>n1) || (i<1) || (j>n2) || (j<1) || (k>n3) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,vel,grd,status,i,j,k,idD)
                # get handle
                han = cart2lin3D(i,j,k,n1,n2)
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    codeDxyz[han,:] .= idD
                end

            end            
        end
    end
    
    #-------------------------------
    ## main FMM loop
    totnpts = n1*n2*n3 
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        # pop top of heap
        han,tmptt = pop_minheap!(bheap)
        lin2cart3D!(han,n1,n2,ptijk)
        ia,ja,ka = ptijk[1],ptijk[2],ptijk[3]
        # set status to accepted
        status[ia,ja,ka] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja,ka] = tmptt

        if dodiscradj
            # store the linear index of FMM order
            idxconv.lfmm2grid[node] = cart2lin3D(ia,ja,ka,n1,n2)
            # store arrival time for first points in FMM order
            fmmord.ttime[node] = tmptt  
        end

        ## try all neighbors of newly accepted point
        for ne=1:6

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>n1) || (i<1) || (j>n2) || (j<1) || (k>n3) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,vel,grd,status,i,j,k,idD)
                han = cart2lin3D(i,j,k,n1,n2)
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    codeDxyz[han,:] .= idD
                end

            elseif status[i,j,k]==1 ## narrow band

                # update the traveltime for this point
                tmptt = calcttpt_2ndord!(ttime,vel,grd,status,i,j,k,idD)
                # get handle
                han = cart2lin3D(i,j,k,n1,n2)
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

                if dodiscradj
                    # codes of chosen derivatives for adjoint
                    codeDxyz[han,:] .= idD
                end
            end
        end
        ##-------------------------------
    end


    ##======================================================

    if dodiscradj

        ## pre-compute the mapping between fmm and orig order
        for i=1:n123
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

            # coefficients along X are the same than along Z
            allcoeffz = allcoeffx
            

        elseif simtype==:spherical
            Δr = grd.Δr
            ## DEG to RAD !!!!
            Δarcθ = [grd.r[i] * deg2rad(grd.Δθ) for i=1:grd.nr]
            Δarcφ = [grd.r[i] * deg2rad(grd.Δφ) for i=1:grd.nr]
            # coefficients
            coe_r_1st = [-1.0/Δr  1.0/Δr]
            coe_r_2nd = [-3.0/(2.0*Δr)  4.0/(2.0*Δr) -1.0/(2.0*Δr) ]
            coe_θ_1st = [-1.0 ./ Δarcθ  1.0 ./ Δarcθ ]
            coe_θ_2nd = [-3.0./(2.0*Δarcθ) 4.0./(2.0*Δarcθ) -1.0./(2.0*Δarcθ) ]
            coe_φ_1st = [-1.0 ./ Δarcφ  1.0 ./ Δarcφ ]
            coe_φ_2nd = [-3.0./(2.0*Δarcφ) 4.0./(2.0*Δarcφ) -1.0./(2.0*Δarcφ) ]

            allcoeffx = CoeffDerivSpherical2D( coe_r_1st, coe_r_2nd )
            allcoeffy = CoeffDerivSpherical2D( coe_θ_1st, coe_θ_2nd )
            allcoeffz = CoeffDerivSpherical2D( coe_φ_1st, coe_φ_2nd )

        end

        #@time begin 

        ## set the derivative operators
        for irow=1:n123
            
            # compute the coefficients for X  derivatives
            setcoeffderiv3D!(fmmord.vecDx,irow,idxconv,codeDxyz,allcoeffx,ptijk,
                             axis=:X,simtype=simtype)
            
            # compute the coefficients for Y derivatives
            setcoeffderiv3D!(fmmord.vecDy,irow,idxconv,codeDxyz,allcoeffy,ptijk,
                             axis=:Y,simtype=simtype)

            # compute the coefficients for Z derivatives
            setcoeffderiv3D!(fmmord.vecDz,irow,idxconv,codeDxyz,allcoeffz,ptijk,
                             axis=:Z,simtype=simtype)

        end

        #end # @time
        
        # create the actual sparse arrays from the vectors
        Nxnnz = fmmord.vecDx.Nnnz[]
        Dx_fmmord = sparse(fmmord.vecDx.i[1:Nxnnz],
                           fmmord.vecDx.j[1:Nxnnz],
                           fmmord.vecDx.v[1:Nxnnz],
                           fmmord.vecDx.Nsize[1], fmmord.vecDx.Nsize[2])

        Nynnz = fmmord.vecDy.Nnnz[]
        Dy_fmmord = sparse(fmmord.vecDy.i[1:Nynnz],
                           fmmord.vecDy.j[1:Nynnz],
                           fmmord.vecDy.v[1:Nynnz],
                           fmmord.vecDy.Nsize[1], fmmord.vecDy.Nsize[2]) 
        
        Nynnz = fmmord.vecDz.Nnnz[]
        Dz_fmmord = sparse(fmmord.vecDz.i[1:Nynnz],
                           fmmord.vecDz.j[1:Nynnz],
                           fmmord.vecDz.v[1:Nynnz],
                           fmmord.vecDz.Nsize[1], fmmord.vecDz.Nsize[2]) 

        ## return all the stuff for discrete adjoint computations
        return ttime,idxconv,fmmord.ttime,Dx_fmmord,Dy_fmmord,Dz_fmmord
    end # if dodiscradj
    ##======================================================
        
    return ttime
end  # ttFMM_hiord


###################################################################

"""
$(TYPEDSIGNATURES)

 Test if point is on borders of domain.
"""
function isonbord(ib::Int64,jb::Int64,kb::Int64,nx::Int64,ny::Int64,nz::Int64)
    isonb1 = false
    isonb2 = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>nx || jb<1 || jb>ny || kb<1 || kb>nz
        isonb1 =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>nx-1 || jb<2 || jb>ny-1 || kb<2 || kb>nz-1 
        isonb2 =  true
    end
    return isonb1,isonb2
end

###################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order.
"""
function calcttpt_2ndord!(ttime::Array{Float64,3},vel::Array{Float64,3},grd::GridEik3D,
                         status::Array{Int64,3},i::Int64,j::Int64,k::Int64,codeD::MVector{3,Int64})

    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    if typeof(grd)==Grid3D
        simtype=:cartesian
    elseif typeof(grd)==Grid3DSphere
        simtype=:spherical
    end

    # sizes, etc.
    if simtype==:cartesian
        n1 = grd.nx
        n2 = grd.ny
        n3 = grd.nz
        Δh = MVector(grd.hgrid,grd.hgrid,grd.hgrid)
    elseif simtype==:spherical
        n1 = grd.nr
        n2 = grd.nθ
        n3 = grd.nφ
        # deg2rad!!!
        Δh = MVector(grd.Δr,
                     grd.r[i]*deg2rad(grd.Δθ),
                     grd.r[i]*sind(grd.θ[j])*deg2rad(grd.Δφ) )
    end

    slowcurpt = 1.0/vel[i,j,k]

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
    # ish = 0  ## 0, +1, -1
    # jsh = 0  ## 0, +1, -1
    # ksh = 0  ## 0, +1, -1
    HUGE = 1.0e30
    codeD[:] .= 0.0 

    ## 3 directions
    for axis=1:3
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## two sides for each direction
        for l=1:2

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

            ## check if on boundaries
            isonb1,isonb2 = isonbord(i+ish,j+jsh,k+ksh,n1,n2,n3)
                                    
            ## 1st order
            if !isonb1 && status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                testval1 = ttime[i+ish,j+jsh,k+ksh]
                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only
                    chosenval1 = testval1
                    use1stord = true
                    
                    # save derivative choices
                    if axis==1
                        codeD[axis]=ish
                    elseif axis==2
                        codeD[axis]=jsh
                    else
                        codeD[axis]=ksh
                    end

                    ## 2nd order
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    ksh2::Int64 = 2*ksh                     
                    if !isonb2 && status[i+ish2,j+jsh2,k+ksh2]==2 ## 2==accepted
                        testval2 = ttime[i+ish2,j+jsh2,k+ksh2]
                        ## pick the lowest value of the two
                        ## <=, compare to chosenval 1, *not* 2!!
                        if testval2<=chosenval1 
                            chosenval2=testval2
                            use2ndord=true
                            
                            # save derivative choices
                            if axis==1
                                codeD[axis]=2*ish
                            elseif axis==2
                                codeD[axis]=2*jsh
                            else
                                codeD[axis]=2*ksh
                            end
                                
                        else
                            chosenval2=HUGE
                            use2ndord=false # this is needed!

                            # save derivative choices
                            if axis==1
                                codeD[axis]=ish
                            elseif axis==2
                                codeD[axis]=jsh
                            else
                                codeD[axis]=ksh
                            end

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

    ## from 2D:
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

            ## 3 directions
            for axis=1:3
                
                use1stord = false
                chosenval1 = HUGE
                
                ## two sides for each direction
                for l=1:2

                    ## map the 4 cases to an integer as in linear indexing...
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
                    
                    ## check if on boundaries
                    isonb1,isonb2 = isonbord(i+ish,j+jsh,k+ksh,n1,n2,n3)
                    
                    ## 1st order
                    if !isonb1 && status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                        testval1 = ttime[i+ish,j+jsh,k+ksh]
                        ## pick the lowest value of the two
                        if testval1<chosenval1 ## < only
                            chosenval1 = testval1
                            use1stord = true

                            # save derivative choices
                            if axis==1
                                codeD[axis]=ish
                            elseif axis==2
                                codeD[axis]=jsh
                            else
                                codeD[axis]=ksh
                            end

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

        end # begin...

        ## recompute sqarg
        sqarg = beta^2-4.0*alpha*gamma

        if sqarg<0.0

            if extrapars.allowfixsqarg==true
                
                gamma = beta^2/(4.0*alpha)
                sqarg = beta^2-4.0*alpha*gamma
                println("calcttpt_2ndord(): ### Brute force fixing problems with 'sqarg', results may be quite inaccurate. ###")
            else

                ## TODO: adapt message to 3D!
                println("\n To get a non-negative discriminant, need to fulfil [in 2D]: ")
                println(" (tx-ty)^2 - 2*s^2/curalpha <= 0")
                println(" where tx,ty can be")
                println(" t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order")
                println(" t? = chosenval1  if 1st order ")
                error("calcttpt_2ndord(): sqarg<0.0, negative discriminant (at i=$i, j=$j, k=$k)")
            end
        end
    end ## if sqarg<0.0
    
    ### roots of the quadratic equation
    tmpsq = sqrt(sqarg)
    soughtt1 = (-beta + tmpsq)/(2.0*alpha)
    soughtt2 = (-beta - tmpsq)/(2.0*alpha)
    ## choose the largest solution
    soughtt = max(soughtt1,soughtt2)

    return soughtt
end

###################################################################

"""
$(TYPEDSIGNATURES)

  Refinement of the grid around the source. Traveltime calculated (FMM) inside a finer grid 
    and then passed on to coarser grid
"""
function ttaroundsrc!(statuscoarse::Array{Int64,3},ttimecoarse::Array{Float64,3},
                      vel::Array{Float64,3},src::Vector{Float64},grdcoarse::GridEik3D,
                      inittt::Float64 ;
                      dodiscradj::Bool=false,idxconv::Union{MapOrderGridFMM3D,Nothing}=nothing,
                      fmmord::Union{VarsFMMOrder3D,Nothing}=nothing )

    ##
    if typeof(grdcoarse)==Grid3D
        simtype=:cartesian
    elseif typeof(grdcoarse)==Grid3DSphere
        simtype=:spherical
    end

    if dodiscradj
        downscalefactor::Int = 5
        noderadius::Int = 5
    else
        downscalefactor = 5
        noderadius = 3
    end        

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    if simtype==:cartesian
        n1_coarse,n2_coarse,n3_coarse = grdcoarse.nx,grdcoarse.ny,grdcoarse.nz
        ixsrcglob,iysrcglob,izsrcglob = findclosestnode(src[1],src[2],src[3],grdcoarse.xinit,
                                                        grdcoarse.yinit,grdcoarse.zinit,grdcoarse.hgrid) 
        
    elseif simtype==:spherical
        n1_coarse,n2_coarse,n3_coarse = grdcoarse.nr,grdcoarse.nθ,grdcoarse.nφ
        ixsrcglob,iysrcglob,izsrcglob = findclosestnode_sph(src[1],src[2],src[3],grdcoarse.rinit,grdcoarse.θinit,grdcoarse.φinit,grdcoarse.Δr,grdcoarse.Δθ,grdcoarse.Δφ ) 
     end

    ##
    ## Define the chunck of coarse grid
    ##
    i1coarsevirtual = ixsrcglob - noderadius
    i2coarsevirtual = ixsrcglob + noderadius
    j1coarsevirtual = iysrcglob - noderadius
    j2coarsevirtual = iysrcglob + noderadius
    k1coarsevirtual = izsrcglob - noderadius
    k2coarsevirtual = izsrcglob + noderadius

    # if hitting borders
    outxmin = i1coarsevirtual<1
    outxmax = i2coarsevirtual>n1_coarse
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>n2_coarse
    outzmin = k1coarsevirtual<1 
    outzmax = k2coarsevirtual>n3_coarse

    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=n1_coarse    : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=n2_coarse    : j2coarse=j2coarsevirtual
    outzmin ? k1coarse=1            : k1coarse=k1coarsevirtual
    outzmax ? k2coarse=n3_coarse    : k2coarse=k2coarsevirtual

    ##
    ## Refined grid parameters
    ##
    # fine grid size
    n1 = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    n2 = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    n3 = (k2coarse-k1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse,k1coarse:k2coarse)

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,n1,n2,n3)
    for k=1:n3
        for j=1:n2
            for i=1:n1
                di=div(i-1,downscalefactor)
                ri=i-di*downscalefactor
                ii = ri>=downscalefactor/2+1 ? di+2 : di+1

                dj=div(j-1,downscalefactor)
                rj=j-dj*downscalefactor
                jj = rj>=downscalefactor/2+1 ? dj+2 : dj+1

                dk=div(k-1,downscalefactor)
                rk=k-dk*downscalefactor
                kk = rk>=downscalefactor/2+1 ? dk+2 : dk+1

                velfinegrd[i,j,k] = velcoarsegrd[ii,jj,kk]
            end
        end
    end 


   # # set the origin of fine grid
    # xinit = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    # yinit = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    # zinit = ((k1coarse-1)*grdcoarse.hgrid+grdcoarse.zinit)
    # grdfine = Grid3D(hgrid=dh,xinit=xinit,yinit=yinit,zinit=zinit,nx=nx,ny=ny,nz=nz)

    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,n1,n2,n3)
    ttime[:,:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,n1,n2,n3)
    status[:,:,:] .= 0   ## set all to far
    # derivative codes
    idD = MVector(0,0,0)
  
    ##
    ## Reset coodinates to match the fine grid
    ##
    # xorig = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    # yorig = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    # zorig = ((k1coarse-1)*grdcoarse.hgrid+grdcoarse.zinit)
    xsrc = src[1] #- xorig - grdcoarse.xinit
    ysrc = src[2] #- yorig - grdcoarse.yinit
    zsrc = src[3] #- zorig - grdcoarse.zinit
    srcfine = Float64[xsrc,ysrc,zsrc]


    if simtype==:cartesian
        # set origin of the fine grid
        xinit = grdcoarse.x[i1coarse]
        yinit = grdcoarse.y[j1coarse]
        zinit = grdcoarse.z[k1coarse]
        dh = grdcoarse.hgrid/downscalefactor
        # fine grid
        grdfine = Grid3D(hgrid=dh,xinit=xinit,yinit=yinit,zinit=zinit,nx=n1,ny=n2,nz=n3)
        ##
        ## Source location, etc. within fine grid
        ##  
        ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
        onsrc = sourceboxloctt!(ttime,velfinegrd,srcfine,grdfine, staggeredgrid=false )
        


    elseif simtype==:spherical
        # set origin of the fine grid
        rinit = grdcoarse.r[i1coarse]
        θinit = grdcoarse.θ[j1coarse]
        φinit = grdcoarse.φ[k1coarse]
        dr = grdcoarse.Δr/downscalefactor
        dθ = grdcoarse.Δθ/downscalefactor
        dφ = grdcoarse.Δφ/downscalefactor
        # fine grid
        grdfine = Grid3DSphere(Δr=dr,Δθ=dθ,Δφ=dφ,nr=n1,nθ=n2,nφ=n3,rinit=rinit,θinit=θinit,φinit=φinit)
        ##
        ## Source location, etc. within fine grid
        ##  
        ## REGULAR grid (not staggered), use "grdfine","finegrd", source position in the fine grid!!
        onsrc = sourceboxloctt_sph!(ttime,velfinegrd,srcfine,grdfine)

    end


    ######################################################
    ptijk = MVector(0,0,0)

    neigh = SA[1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]

    #-------------------------------
    ## init FMM
    status[onsrc] .= 2 ## set to accepted on src
    naccinit=count(status.==2)

    ## get all i,j,k accepted
    ijkss = findall(status.==2) 
    is = [l[1] for l in ijkss]
    js = [l[2] for l in ijkss]
    ks = [l[3] for l in ijkss]
    naccinit = length(ijkss)

    ##===================================
    for l=1:naccinit
        ia = is[l]
        ja = js[l]
        ka = ks[l]
        # if the node coincides with the coarse grid, store values
        oncoa,ia_coarse,ja_coarse,ka_coarse = isacoarsegridnode(ia,ja,ka,downscalefactor,
                                                                i1coarse,j1coarse,k1coarse)
        if oncoa
            ##===================================
            ## UPDATE the COARSE GRID
            ttimecoarse[ia_coarse,ja_coarse,ka_coarse]  = ttime[ia,ja,ka]
            statuscoarse[ia_coarse,ja_coarse,ka_coarse] = status[ia,ja,ka]
            ##==================================
        end
    end

    ##==========================================================##
    # discrete adjoint
    if dodiscradj       

        # discrete adjoint: first visited points in FMM order
        ttfirstpts = Vector{Float64}(undef,naccinit)
        for l=1:naccinit
            i,j,k = is[l],js[l],ks[l]
            ttij = ttime[i,j,k]
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
            oncoa,ia_coarse,ja_coarse,ka_coarse = isacoarsegridnode(is[p],js[p],ks[p],downscalefactor,
                                                                    i1coarse,j1coarse,k1coarse)
            if oncoa
                # go from cartesian (i,j) to linear
                idxconv.lfmm2grid[node_coarse] = cart2lin3D(ia_coarse,ja_coarse,ka_coarse,
                                                            idxconv.nx,idx.conv.ny)

                # store arrival time for first points in FMM order
                fmmord.ttime[node_coarse] = ttime[is[p],js[p],ks[p]]
                
                node_coarse +=1
            end
        end
    end # if dodiscradj
    ##==========================================================##


    ## Init the min binary heap with void arrays but max size
    Nmax=n1*n2*n3
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    
    ## construct initial narrow band
    for l=1:naccinit ##
        for ne=1:6 ## six potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1) || (k>n3) || (k<1)
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,velfinegrd,grdfine,status,i,j,k,idD)

                # get handle
                # han = sub2ind((nx,ny),i,j)
                han = cart2lin3D(i,j,k,n1,n2)
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1
            end            
        end
    end

    #-------------------------------
    ## main FMM loop
    firstwarning=true
    totnpts = n1*n2*n3
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nx,ny),han)
        lin2cart3D!(han,n1,n2,ptijk)
        ia,ja,ka = ptijk[1],ptijk[2],ptijk[3]
        # set status to accepted
        status[ia,ja,ka] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja,ka] = tmptt
        
        ##===================================
        oncoa,ia_coarse,ja_coarse,ka_coarse = isacoarsegridnode(ia,ja,ka,downscalefactor,i1coarse,j1coarse,k1coarse)
        if oncoa

            ##===================================
            ## UPDATE the COARSE GRID
            ttimecoarse[ia_coarse,ja_coarse,ka_coarse] = tmptt
            statuscoarse[ia_coarse,ja_coarse,ka_coarse] = 2
            ##===================================

            ##===================================
            # Discrete adjoint
            if dodiscradj
                # store the linear index of FMM order
                idxconv.lfmm2grid[node_coarse] = cart2lin3D(ia_coarse,ja_coarse,ka_coarse,idxconv.nx,idxconv.ny)
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
        #  if (ia==nx) || (ia==1) || (ja==ny) || (ja==1)
        if (ia==1 && !outxmin) || (ia==n1 && !outxmax) || (ja==1 && !outymin) || (ja==n2 && !outymax) || (ka==1 && !outzmin) || (ka==n3 && !outzmax)
 
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
        for ne=1:6 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if (i>n1) || (i<1) || (j>n2) || (j<1) || (k>n3) || (k<1)
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord!(ttime,velfinegrd,grdfine,status,i,j,k,idD)
                han = cart2lin3D(i,j,k,n1,n2)
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord!(ttime,velfinegrd,grdfine,status,i,j,k,idD)
                # get handle
                han = cart2lin3D(i,j,k,n1,n2)
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------

    end
    error("Ouch...")
end

#################################################################3


###############################################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Vector{Float64},grd::Grid3D; staggeredgrid::Bool=false )
    ## staggeredgrid keyword required!

    mindistsrc = 1e-5
  
    xsrc,ysrc,zsrc=srcpos[1],srcpos[2],srcpos[3]
    
    if staggeredgrid==false
        ## regular grid
        onsrc = zeros(Bool,grd.nx,grd.ny,grd.nz)
        onsrc[:,:,:] .= false
        ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
        rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)

    elseif staggeredgrid==true
        ## grd.xinit-hgr because TIME array on STAGGERED grid
        onsrc = zeros(Bool,grd.ntx,grd.nty,grd.ntz)
        onsrc[:,:,:] .= false
        hgr = grd.hgrid/2.0
        ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
        rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)

    end

  
    halfg = 0.0 #hgrid/2.0
    dist = sqrt(rx^2+ry^2+rz^2)
    #@show dist,src,rx,ry
    if dist<=mindistsrc
        onsrc[ix,iy,iz] = true
        ttime[ix,iy,iz] = 0.0 
    else
        
        if (rx>=halfg) & (ry>=halfg) & (rz>=halfg)
            onsrc[ix:ix+1,iy:iy+1,iz:iz+1] .= true
        elseif (rx<halfg) & (ry>=halfg) & (rz>=halfg)
            onsrc[ix-1:ix,iy:iy+1,iz:iz+1] .= true
        elseif (rx<halfg) & (ry<halfg) & (rz>=halfg)
            onsrc[ix-1:ix,iy-1:iy,iz:iz+1] .= true
        elseif (rx>=halfg) & (ry<halfg) & (rz>=halfg)
            onsrc[ix:ix+1,iy-1:iy,iz:iz+1] .= true

        elseif (rx>=halfg) & (ry>=halfg) & (rz<halfg)
            onsrc[ix:ix+1,iy:iy+1,iz-1:iz] .= true
        elseif (rx<halfg) & (ry>=halfg) & (rz<halfg)
            onsrc[ix-1:ix,iy:iy+1,iz-1:iz] .= true
        elseif (rx<halfg) & (ry<halfg) & (rz<halfg)
            onsrc[ix-1:ix,iy-1:iy,iz-1:iz] .= true
        elseif (rx>=halfg) & (ry<halfg) & (rz<halfg)
            onsrc[ix:ix+1,iy-1:iy,iz-1:iz] .= true
        end

        ## set ttime around source ONLY FOUR points!!!
        ijksrc = findall(onsrc)
        for lcart in ijksrc
            i = lcart[1]
            j = lcart[2]
            k = lcart[3]
            if staggeredgrid==false
                ## regular grid
                xp = (i-1)*grd.hgrid+grd.xinit
                yp = (j-1)*grd.hgrid+grd.yinit
                zp = (k-1)*grd.hgrid+grd.zinit
                ii = Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
                jj = Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
                kk = Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
                ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
            elseif staggeredgrid==true
                ## grd.xinit-hgr because TIME array on STAGGERED grid
                xp = (i-1)*grd.hgrid+grd.xinit-hgr
                yp = (j-1)*grd.hgrid+grd.yinit-hgr
                zp = (k-1)*grd.hgrid+grd.zinit-hgr
                ii = i-1 
                jj = j-1 
                kk = k-1 
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
            end
        end
    end
    return onsrc
end

#############################################################################


"""
$(TYPEDSIGNATURES)
"""
function sourceboxloctt_sph!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Vector{Float64},grd::Grid3DSphere )
    ## staggeredgrid keyword required!

    mindistsrc = 1e-5
  
    rsrc,θsrc,φsrc=srcpos[1],srcpos[2],srcpos[3]

    ## regular grid
    onsrc = zeros(Bool,grd.nr,grd.nθ,grd.nφ)
    onsrc[:,:,:] .= false
    ir,iθ,iφ = findclosestnode_sph(rsrc,θsrc,φsrc,grd.rinit,grd.θinit,grd.φinit,grd.Δr,grd.Δθ,grd.Δφ)

    rr = rsrc-grd.r[ir] #((ir-1)*grd.Δr+grd.rinit)
    rθ = θsrc-grd.θ[iθ] #((iθ-1)*grd.Δθ+grd.θinit)
    rφ = φsrc-grd.φ[iφ] #((iφ-1)*grd.Δφ+grd.φinit)

    halfg = 0.0 #hgrid/2.0
    ## distance in POLAR coordinates
    ## sqrt(r1^2 +r2^2 -2*r1*r2*(sin(θ1)*sin(θ2)*cos(φ1-φ2)+cos(θ1)*cos(θ2)))
    r1 = rsrc
    r2 = grd.r[ir]
    θ1 = θsrc
    θ2 = grd.θ[iθ]
    φ1 = φsrc
    φ2 = grd.φ[iφ]
    dist = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))
    #@show dist,src,rr,rθ
    if dist<=mindistsrc
        onsrc[ir,iθ,iφ] = true
        ttime[ir,iθ,iφ] = 0.0 
    else

        if (rr>=halfg) & (rθ>=halfg) & (rφ>=halfg)
            onsrc[ir:ir+1,iθ:iθ+1,iφ:iφ+1] .= true
        elseif (rr<halfg) & (rθ>=halfg) & (rφ>=halfg)
            onsrc[ir-1:ir,iθ:iθ+1,iφ:iφ+1] .= true
        elseif (rr<halfg) & (rθ<halfg) & (rφ>=halfg)
            onsrc[ir-1:ir,iθ-1:iθ,iφ:iφ+1] .= true
        elseif (rr>=halfg) & (rθ<halfg) & (rφ>=halfg)
            onsrc[ir:ir+1,iθ-1:iθ,iφ:iφ+1] .= true

        elseif (rr>=halfg) & (rθ>=halfg) & (rφ<halfg)
            onsrc[ir:ir+1,iθ:iθ+1,iφ-1:iφ] .= true
        elseif (rr<halfg) & (rθ>=halfg) & (rφ<halfg)
            onsrc[ir-1:ir,iθ:iθ+1,iφ-1:iφ] .= true
        elseif (rr<halfg) & (rθ<halfg) & (rφ<halfg)
            onsrc[ir-1:ir,iθ-1:iθ,iφ-1:iφ] .= true
        elseif (rr>=halfg) & (rθ<halfg) & (rφ<halfg)
            onsrc[ir:ir+1,iθ-1:iθ,iφ-1:iφ] .= true
        end

        ## set ttime around source ONLY FOUR points!!!
        ijksrc = findall(onsrc)
        for lcart in ijksrc
            i = lcart[1]
            j = lcart[2]
            k = lcart[3]
            ## regular grid
            # rp = (i-1)*grd.hgrid+grd.rinit
            # θp = (j-1)*grd.hgrid+grd.θinit
            # φp = (k-1)*grd.hgrid+grd.φinit
            # ii = Int(floor((rsrc-grd.rinit)/grd.rgrid) +1)
            # jj = Int(floor((θsrc-grd.θinit)/grd.θgrid) +1)
            # kk = Int(floor((φsrc-grd.φinit)/grd.φgrid) +1) 
       
            r1 = rsrc
            r2 = grd.r[i]
            θ1 = θsrc
            θ2 = grd.θ[j]
            φ1 = φsrc
            φ2 = grd.φ[k]
            distp = sqrt(r1^2+r2^2 -2*r1*r2*(sind(θ1)*sind(θ2)*cosd(φ1-φ2)+cosd(θ1)*cosd(θ2)))
            ttime[i,j,k] = distp / vel[i,j,k]
        end
    end
    return onsrc
end

################################################################################


#########################################################
#end
#########################################################



