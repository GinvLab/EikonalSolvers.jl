
#######################################################
##        Eikonal  2D                                ## 
#######################################################


"""
$(TYPEDSIGNATURES)

Calculate traveltime for 2D velocity models with various algorithms. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 2D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y) for each single source, a vector of 2-column arrays
- `ttalgo` : the algorithm to use to compute the traveltime, one amongst the following
    * "ttFS\\_podlec", fast sweeping method using Podvin-Lecomte stencils
    * "ttFMM\\_podlec," fast marching method using Podvin-Lecomte stencils
    * "ttFMM\\_hiord", second order fast marching method, the default algorithm 
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime2Dgen(vel::Array{Float64,2},grd::Grid2D,coordsrc::Array{Float64,2},
                         coordrec::Vector{Array{Float64,2}} ; ttalgo::String, ##  ="ttFMM_hiord",
                         returntt::Bool=false) 
        
    @assert size(coordsrc,2)==2
    @assert all(vel.>0.0)
    @assert all(grd.xinit.<=coordsrc[:,1].<=((grd.nx-1)*grd.hgrid+grd.xinit))
    @assert all(grd.yinit.<=coordsrc[:,2].<=((grd.ny-1)*grd.hgrid+grd.yinit))

    @assert size(coordsrc,1)==length(coordrec)
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
        if ttalgo=="ttFMM_hiord"
            # in this case velocity and time arrays have the same shape
            ttime = zeros(grd.nx,grd.ny,nsrc)
        else
            # in this case the time array has shape(velocity)+1
            ttime = zeros(grd.ntx,grd.nty,nsrc)
        end
    end

    @sync begin

        if returntt 
            # return traveltime picks at receivers and at all grid points 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2Dgen,wks[s],
                                                                        vel,coordsrc[igrs,:],
                                                                        coordrec[igrs],grd,ttalgo,
                                                                        returntt=returntt )
            end

        else
            # return ONLY traveltime picks at receivers 
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc2Dgen,wks[s],
                                                        vel,coordsrc[igrs,:],
                                                        coordrec[igrs],grd,ttalgo,
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
function ttforwsomesrc2Dgen(vel::Array{Float64,2},coordsrc::Array{Float64,2},
                            coordrec::Vector{Array{Float64,2}},grd::Grid2D,
                            ttalgo::String ; returntt::Bool=false )
    
    nsrc = size(coordsrc,1)
    #nrec = size(coordrec,1)                
    #ttpicksGRPSRC = zeros(nrec,nsrc)
    
    ttpicksGRPSRC = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicksGRPSRC[i] = zeros(curnrec)
    end

    
    if ttalgo=="ttFMM_hiord"
        # in this case velocity and time arrays have the same shape
        ttGRPSRC = zeros(grd.nx,grd.ny,nsrc)
    else
        # in this case the time array has shape(velocity)+1
        ttGRPSRC = zeros(grd.ntx,grd.nty,nsrc)
    end

    ## group of pre-selected sources
    for s=1:nsrc    

        ## Compute traveltime and interpolation at receivers in one go for parallelization
        
        if ttalgo=="ttFS_podlec"
            ttGRPSRC[:,:,s] = ttFS_podlec(vel,coordsrc[s,:],grd)
            
        elseif ttalgo=="ttFMM_podlec"
            ttGRPSRC[:,:,s] = ttFMM_podlec(vel,coordsrc[s,:],grd)

        elseif ttalgo=="ttFMM_hiord"
            ttGRPSRC[:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grd)

        else
            println("\nttforwsomesrc(): Wrong ttalgo name... \n")
            return nothing
        end

        ## interpolate at receivers positions
        for i=1:size(coordrec[s],1) 
            ttpicksGRPSRC[s][i] = bilinear_interp( ttGRPSRC[:,:,s], grd,
                                                   coordrec[s][i,1],coordrec[s][i,2])
        end
    end

    
    if returntt 
        return ttGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC 
end

####################################################################################


"""
$(TYPEDSIGNATURES)
 
 Fast sweeping method for a single source in 2D using using Podvin-Lecomte stencils on a staggered grid.
"""
function ttFS_podlec(vel::Array{Float64,2},src::Vector{Float64},grd::Grid2D) 

    epsilon = 1e-6
      
    ## ttime
    ntx,nty=grd.ntx,grd.nty #size(vel).+1  ## STAGGERED GRID!!!
    nvx = grd.nx
    nvy = grd.ny
    inittt = 1e30
    ttime = inittt * ones(Float64,ntx,nty)

    #println("ix iy  $ix $iy")
    # global dispfirstonly
    # if dispfirstonly
    #     println("\ntteikonal(): FIX TIME AROUND SOURCE: take into account different velocity pixels!!\n ")
    #    dispfirstonly = false 
    # end

    ## source location, etc.      
    ## STAGGERED grid
    onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=true )

    ######################################################

    iswe = [1 ntx 1; ntx 1 -1; ntx 1 -1; 1  ntx 1]
    jswe = [1 nty 1; 1 nty 1;  nty 1 -1; nty 1 -1]
    
    ###########################################    
 
    ## indices of points clockwise
    ## A
    cooa = [[-1 0];
            [0  1];
            [0  1];
            [1  0];
            [1  0];
            [0 -1];
            [0 -1];
            [-1 0] ]
    ## B
    coob = [[-1  1];
            [-1  1];
            [1   1];
            [1   1];
            [1  -1];
            [1  -1];
            [-1 -1];
            [-1 -1] ]  
    ## velocity in ABCD
    coovin = [[-1  0];
              [-1  0];
              [0   0];
              [0   0];
              [0  -1];
              [0  -1];
              [-1 -1];
              [-1 -1] ]
    # velocity in AFGC see paper
    coovadj = [[-1 -1];
               [0   0];
               [-1  0];
               [0  -1];
               [0   0];
               [-1 -1];
               [0  -1];
               [-1  0] ]
    
    ##====== distances ===============
    distab2 = grd.hgrid^2
    distac = grd.hgrid
    distbc = sqrt(2.0)*grd.hgrid
    distab = grd.hgrid
    distAB2divBC = distab2/distbc
    ttlocmin = inittt
    ##================================
    slowness = 1.0./vel
    ##================================
 
    ttlocmin = zeros(Float64,8) 
    swedifference::Float64 = 0.0
    ttimeold = Array{Float64,2}(undef,ntx,nty)
    #--------------------------------------
    
    #pa=0
    swedifference = 100.0*epsilon
    while swedifference>epsilon
        #pa +=1
        ttimeold[:,:] = ttime
        
        for swe=1:4             
            for j=jswe[swe,1]:jswe[swe,3]:jswe[swe,2]
                for i=iswe[swe,1]:iswe[swe,3]:iswe[swe,2]
                    
                    ##--------------------------------##
                    ## if on src, skip this iteration
                    onsrc[i,j]==true && continue
                    
                    ####################################################
                    ##   Local solver (Podvin, Lecomte, 1991)         ##
                    ####################################################
                    
                    ttlocmin[:] .= inittt  
                    ttdiffr = inittt  
                    ttheadw = inittt
                    ttc = inittt
                    
                    ## loop on triangles
                    for cou=1:8
                        
                        ipix = coovin[cou,1] + i
                        jpix = coovin[cou,2] + j
               
                        ## if the point is out of bounds skip this iteration
                        if (ipix>nvx) || (ipix<1) || (jpix>nvy) || (jpix<1)
                            continue
                        end                          
                        
                        # ###################################################
                        # ##  transmission
                        tta = ttime[i+cooa[cou,1], j+cooa[cou,2]]
                        ttb = ttime[i+coob[cou,1], j+coob[cou,2]]
                        testas = distAB2divBC * slowness[ipix, jpix]
                        ## stability condition
                        if ((tta-ttb)>=0.0) & ( (tta-ttb)<=testas )
                            ## distab = grd.hgrid #sqrt((xa-xb)^2+(ya-yb)^2)
                            ttc = tta + sqrt( (distab*slowness[ipix,jpix])^2 - (tta-ttb)^2 )
                        end  

                        ## to avoid multiple calculations of the same arrivals...
                        if isodd(cou)  ## cou=[1,3,5,7]
                            
                            ###################################################
                            ## diffraction
                            ttdiffr = ttb + slowness[ipix, jpix] * distbc
                            
                            ###################################################
                            ## head wave
                            ## all these cases ARE necessary!
                            ## ttheadw = inittt
                            iadj = coovadj[cou,1] + i
                            jadj = coovadj[cou,2] + j

                            if (iadj>nvx) || (iadj<1) || (jadj>nvy) || (jadj<1) 
                                ##ttheadw = inittt
                                ttheadw = tta + distac * slowness[ipix,jpix]
                            else
                                #slowadj = slowness[i+coovadj[cou,1], j+coovadj[cou,2]]
                                #1.0/vel[i+coovadj[cou,1], j+coovadj[cou,2]]
                                ttheadw = tta + distac * min( slowness[ipix,jpix],
                                                              slowness[iadj,jadj] )
                            end
                        end
                        
                        ## minimum time
                        ttlocmin[cou] = min(ttc,ttheadw,ttdiffr) # ,ttlocmin)!!
                        
                    end
                    
                    ##################################################
                    
                    ttime[i,j] = min(ttimeold[i,j],minimum(ttlocmin))

                    ##################################################
                end
            end
        end    
        ##============================================##        
        swedifference = maximum(abs.(ttime.-ttimeold))
    end
       
    return ttime
end ## ttFS_podlec

###############################################################################

"""
$(TYPEDSIGNATURES)
 
 Fast marching method for a single source in 2D using using Podvin-Lecomte stencils on a staggered grid.
"""
function ttFMM_podlec(vel::Array{Float64,2},src::Vector{Float64},grd::Grid2D) 
 
    epsilon = 1e-6
      
    ## ttime
    nx,ny=grd.ntx,grd.nty # size of time array == size(vel).+1  ## STAGGERED GRID!!!
    nvx = grd.nx
    nvy = grd.ny
    inittt = 1e30
    ttime = Array{Float64}(undef,nx,ny)
    ttime[:,:] .= inittt
    
    #println("ix iy  $ix $iy")
    # global dispfirstonly
    # if dispfirstonly
    #     println("\ntteikonal(): FIX TIME AROUND SOURCE: take into account different velocity pixels!!\n ")
    #    dispfirstonly = false 
    # end

    ## source location, etc.
    ## STAGGERED grid
    onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=true )

    ###################################################### 
    ## indices of points clockwise
    ## A
    cooa = [[-1 0];
            [0  1];
            [0  1];
            [1  0];
            [1  0];
            [0 -1];
            [0 -1];
            [-1 0] ]
    ## B
    coob = [[-1  1];
            [-1  1];
            [1   1];
            [1   1];
            [1  -1];
            [1  -1];
            [-1 -1];
            [-1 -1] ]  
    ## velocity in ABCD
    coovin = [[-1  0];
              [-1  0];
              [0   0];
              [0   0];
              [0  -1];
              [0  -1];
              [-1 -1];
              [-1 -1] ]
    # velocity in AFGC see paper
    coovadj = [[-1 -1];
               [0   0];
               [-1  0];
               [0  -1];
               [0   0];
               [-1 -1];
               [0  -1];
               [-1  0] ]
   
    ##================================
    slowness = 1.0./vel
    # static array
    neigh = SA[1  0;
               0  1;
              -1  0;
               0 -1]

    #-------------------------------
    ## init FMM 

    status = Array{Int64}(undef,nx,ny)
    status[:,:] .= 0   ## set all to far
    status[onsrc] .= 2 ## set to accepted on src
    naccinit=count(status.==2)
    
    ## get all i,j accepted
    ijss = findall(status.==2) 
    is = [l[1] for l in ijss]
    js = [l[2] for l in ijss]
    naccinit = length(ijss)

    ## Init the min binary heap with void arrays but max size
    Nmax=nx*ny
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    ttlocmin = zeros(Float64,8)
    
    ## conversion cart to lin indices, old sub2ind
    linid_nxny = LinearIndices((nx,ny))
    ## conversion lin to cart indices, old sub2ind
    cartid_nxny = CartesianIndices((nx,ny))
    
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
                tmptt = calcttpt_podlec!(ttime,ttlocmin,inittt,slowness,grd,cooa,coob,coovin,coovadj,i,j)
                # get handle
                # han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j]
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1
            end            
        end
    end

    #-------------------------------
    ## main FMM loop
    totnpts = nx*ny
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        # pop node
        han,tmptt = pop_minheap!(bheap)
        cija = cartid_nxny[han]
        ia,ja = cija[1],cija[2]
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt

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
                tmptt = calcttpt_podlec!(ttime,ttlocmin,inittt,slowness,grd,cooa,coob,coovin,coovadj,i,j)
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_podlec!(ttime,ttlocmin,inittt,slowness,grd,cooa,coob,coovin,coovadj,i,j)
                # get handle
                han = linid_nxny[i,j]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)
                
            end
        end
        ##-------------------------------
    end
    return ttime
end

############################################################################


"""
$(TYPEDSIGNATURES)

 Compute the traveltime at requested node using Podvin-Lecomte stencils on a staggered grid.
"""
function calcttpt_podlec!(ttime::Array{Float64,2},ttlocmin::Vector{Float64},inittt::Float64,
                          slowness::Array{Float64,2},grd::Grid2D,
                          cooa::Array{Int64,2},coob::Array{Int64,2},
                          coovin::Array{Int64,2},coovadj::Array{Int64,2},
                          i::Int64,j::Int64)
    
    ##--------------------------------##
    distab2 = grd.hgrid^2
    distac = grd.hgrid
    distbc = sqrt(2.0)*grd.hgrid
    distab = grd.hgrid
    distAB2divBC = distab2/distbc
    nvx = grd.nx
    nvy = grd.ny
    
    ####################################################
    ##   Local solver (Podvin, Lecomte, 1991)         ##
    ####################################################
    
    ttlocmin[:] .= inittt  
    ttdiffr = inittt  
    ttheadw = inittt
    ttc = inittt
    
    ## loop on triangles
    for cou=1:8
        
        ipix = coovin[cou,1] + i
        jpix = coovin[cou,2] + j
        
        ## if the point is out of bounds skip this iteration
        if (ipix>nvx) || (ipix<1) || (jpix>nvy) || (jpix<1)
            continue
        end                          
        
        # ###################################################
        # ##  transmission
        tta = ttime[i+cooa[cou,1], j+cooa[cou,2]]
        ttb = ttime[i+coob[cou,1], j+coob[cou,2]]
        testas = distAB2divBC * slowness[ipix, jpix]
        ## stability condition
        if ((tta-ttb)>=0.0) & ( (tta-ttb)<=testas )
            ## distab = grd.hgrid #sqrt((xa-xb)^2+(ya-yb)^2)
            ttc = tta + sqrt( (distab*slowness[ipix,jpix])^2 - (tta-ttb)^2 )
        end  

        ## to avoid multiple calculations of the same arrivals...
        if isodd(cou)  ## cou=[1,3,5,7]
            
            ###################################################
            ## diffraction
            ttdiffr = ttb + slowness[ipix, jpix] * distbc
            
            ###################################################
            ## head wave
            ## all these cases ARE necessary!
            ## ttheadw = inittt
            iadj = coovadj[cou,1] + i
            jadj = coovadj[cou,2] + j

            if (iadj>nvx) || (iadj<1) || (jadj>nvy) || (jadj<1) 
                ##ttheadw = inittt
                ttheadw = tta + distac * slowness[ipix,jpix]
            else
                #slowadj = slowness[i+coovadj[cou,1], j+coovadj[cou,2]]
                #1.0/vel[i+coovadj[cou,1], j+coovadj[cou,2]]
                ttheadw = tta + distac * min( slowness[ipix,jpix],
                                              slowness[iadj,jadj] )
            end
        end
        
        ## minimum time
        ttlocmin[cou] = min(ttc,ttheadw,ttdiffr) # ,ttlocmin)!!            
    end
    
    ##################################################
    #ttime[i,j] = min(ttimeold[i,j],minimum(ttlocmin))
    ttf = minimum(ttlocmin)
    return ttf
    ##################################################
end


#################################################################################


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
- `gradttalgo`: the algorithm to use to compute the forward and gradient, one amongst the following
    * "gradFS\\_podlec", fast sweeping method using Podvin-Lecomte stencils for forward, fast sweeping method for adjoint   
    * "gradFMM\\_podlec," fast marching method using Podvin-Lecomte stencils for forward, fast marching method for adjoint 
    * "gradFMM\\_hiord", second order fast marching method for forward, high order fast marching method for adjoint 
- `smoothgrad`: smooth the gradient? true or false

# Returns
- `grad`: the gradient as a 2D array

"""
function gradttime2Dgen(vel::Array{Float64,2}, grd::Grid2D,coordsrc::Array{Float64,2},coordrec::Vector{Array{Float64,2}},
                        pickobs::Vector{Vector{Float64}},stdobs::Vector{Vector{Float64}} ;
                        gradttalgo::String,
                        smoothgradsourceradius::Integer=0,smoothgrad::Bool=false )

    @assert size(coordsrc,2)==2
    #@assert size(coordrec,2)==2
    @assert all(vel.>0.0)
    @assert all(grd.xinit.<=coordsrc[:,1].<=((grd.nx-1)*grd.hgrid+grd.xinit))
    @assert all(grd.yinit.<=coordsrc[:,2].<=((grd.ny-1)*grd.hgrid+grd.yinit))

    @assert size(coordsrc,1)==length(coordrec)

    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(grd.nx,grd.ny,nchu)
    ## do the calculations
    @sync begin 
        for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @async tmpgrad[:,:,s] = remotecall_fetch(calcgradsomesrc2Dgen,wks[s],vel,
                                                     coordsrc[igrs,:],coordrec[igrs],
                                                     grd,stdobs[igrs],pickobs[igrs],
                                                     gradttalgo,smoothgradsourceradius )
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
function calcgradsomesrc2Dgen(vel::Array{Float64,2},xysrc::Array{Float64,2},
                           coordrec::Vector{Array{Float64,2}},grd::Union{Grid2D,Grid3D},
                           stdobs::Vector{Vector{Float64}},pickobs1::Vector{Vector{Float64}},
                           gradalgo::String,smoothgradsourceradius::Integer )
                           
    nx,ny=size(vel)
    nsrc = size(xysrc,1)
    nrec = size(coordrec,1)                
    #ttpicks1 = zeros(nrec)
    ttpicks1 = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicks1[i] = zeros(curnrec)
    end

    grad1 = zeros(nx,ny)
    
    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        curnrec = size(coordrec[s],1) 
        ttpicks1 = zeros(curnrec)

        ###########################################
        ## calc ttime

        if gradalgo=="gradFMM_podlec"
            ttgrdonesrc = ttFMM_podlec(vel,xysrc[s,:],grd)
            
        elseif gradalgo=="gradFMM_hiord"
            ttgrdonesrc = ttFMM_hiord(vel,xysrc[s,:],grd)

        elseif gradalgo=="gradFS_podlec"
            ttgrdonesrc = ttFS_podlec(vel,xysrc[s,:],grd)
            
        else
            println("\ncalcgradsomesrc(): Wrong gradalgo name: $(gradalgo)... \n")
            return nothing
        end
        
        ## ttime at receivers
        for i=1:curnrec
            ttpicks1[i] = bilinear_interp( ttgrdonesrc,grd,coordrec[s][i,1],coordrec[s][i,2] )
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        if gradalgo=="gradFS_podlec"

            grad1 .+= eikgrad_FS_SINGLESRC(ttgrdonesrc,vel,xysrc[s,:],coordrec[s],grd,
                                           pickobs1[s],ttpicks1,stdobs[s])

        elseif gradalgo=="gradFMM_podlec"

            grad1 .+= eikgrad_FMM_SINGLESRC(ttgrdonesrc,vel,xysrc[s,:],coordrec[s],
                                            grd,pickobs1[s],ttpicks1,stdobs[s])
            
        elseif gradalgo=="gradFMM_hiord"
            grad1 .+= eikgrad_FMM_hiord_SINGLESRC(ttgrdonesrc,vel,xysrc[s,:],coordrec[s],
                                                  grd,pickobs1[s],ttpicks1,stdobs[s])

        else
            println("Wrong gradalgo algo name: $(gradalgo)... ")
            return
        end
        
        ###########################################
        ## smooth gradient around the source
        smoothgradaroundsrc!(grad1,xysrc[s,1],xysrc[s,2],grd,
                             radiuspx=smoothgradsourceradius)

    end

    return grad1
end 

#############################################################################
