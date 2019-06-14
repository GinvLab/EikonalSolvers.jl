

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################


##########################################################################

## @doc raw because of some backslashes in the string...
@doc raw"""
    traveltime2D(vel::Array{Float64,2},grd::Grid2D,coordsrc::Array{Float64,2},
                 coordrec::Array{Float64,2} ; algo::String="ttFMM_hiord", 
                 returntt::Bool=false ) 

Calculate traveltime for 2D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.

# Arguments
- `vel`: the 2D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y), a 2-column array
- `algo` (optional): the algorithm to use to compute the traveltime, one amongst the following
    * "ttFS\_podlec", fast sweeping method using Podvin-Lecomte stencils
    * "ttFMM\_podlec," fast marching method using Podvin-Lecomte stencils
    * "ttFMM\_hiord", second order fast marching method, the default algorithm 
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime2D( vel::Array{Float64,2},grd::Grid2D,coordsrc::Array{Float64,2},
                       coordrec::Array{Float64,2} ; algo::String="ttFMM_hiord", returntt::Bool=false ) 
        
    @assert size(coordsrc,2)==2
    @assert size(coordrec,2)==2
    @assert all(vel.>=0.0)
    @assert all(grd.xinit.<coordsrc[:,1].<((grd.nx-1)*grd.hgrid+grd.xinit))
    @assert all(grd.yinit.<coordsrc[:,2].<((grd.ny-1)*grd.hgrid+grd.yinit))
    

    nsrc = size(coordsrc,1)
    nrec = size(coordrec,1)    
    ttpicks = zeros(nrec,nsrc)

    ## calculate how to subdivide the srcs among the workers
    nsrc=size(coordsrc,1)
    nw = nworkers()
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()

    if returntt
        # return traveltime array and picks at receivers
        if algo=="ttFMM_hiord"
            # in this case velocity and time arrays have the same shape
            ttime = zeros(grd.nx,grd.ny,nsrc)
        else
            # in this case the time array has shape(velocity)+1
            ttime = zeros(grd.ntx,grd.nty,nsrc)
        end
    end

    @sync begin

        if !returntt
            # return ONLY traveltime picks at receivers
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[:,igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                          vel,coordsrc[igrs,:],
                                                          coordrec,grd,algo,
                                                          returntt=returntt )
            end
        elseif returntt
            # return both traveltime picks at receivers and at all grid points
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,igrs],ttpicks[:,igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                                          vel,coordsrc[igrs,:],
                                                                          coordrec,grd,algo,
                                                                          returntt=returntt )
            end
        end

    end # sync

    if !returntt
        return ttpicks
    elseif returntt
        return ttpicks,ttime
    end
end

###########################################################################

function ttforwsomesrc2D(vel::Array{Float64,2},coordsrc::Array{Float64,2},
                      coordrec::Array{Float64,2},grd::Grid2D,
                      algo::String ; returntt::Bool=false )
    
    nsrc = size(coordsrc,1)
    nrec = size(coordrec,1)                
    ttpicksGRPSRC = zeros(nrec,nsrc) 

    if algo=="ttFMM_hiord"
        # in this case velocity and time arrays have the same shape
        ttGRPSRC = zeros(grd.nx,grd.ny,nsrc)
    else
        # in this case the time array has shape(velocity)+1
        ttGRPSRC = zeros(grd.ntx,grd.nty,nsrc)
    end

    ## group of pre-selected sources
    @inbounds for s=1:nsrc    

        ## Compute traveltime and interpolation at receivers in one go for parallelization
        
        if algo=="ttFS_podlec"
            ttGRPSRC[:,:,s] = ttFS_podlec(vel,coordsrc[s,:],grd)
            
        elseif algo=="ttFMM_podlec"
            ttGRPSRC[:,:,s] = ttFMM_podlec(vel,coordsrc[s,:],grd)

        elseif algo=="ttFMM_hiord"
            ttGRPSRC[:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grd)

        else
            println("\nttforwsomesrc(): Wrong algo name... \n")
            return nothing
        end

        ## interpolate at receivers positions
        @inbounds for i=1:nrec
            ttpicksGRPSRC[i,s] = bilinear_interp( ttGRPSRC[:,:,s], grd.hgrid,grd.xinit,
                                                grd.yinit,coordrec[i,1],coordrec[i,2])
        end
    end
    
    if returntt
        return ttGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC 
end


#################################################################################

function ttFS_podlec(vel::Array{Float64,2},src::Array{Float64,1},grd::Grid2D) 

    epsilon = 1e-6
      
    ## ttime
    nx,ny=grd.ntx,grd.nty #size(vel).+1  ## STAGGERED GRID!!!
    nvx = grd.nx
    nvy = grd.ny
    inittt = 1e30
    ttime = inittt * ones(Float64,nx,ny)

    #println("ix iy  $ix $iy")
    # global dispfirstonly
    # if dispfirstonly
    #     println("\ntteikonal(): FIX TIME AROUND SOURCE: take into account different velocity pixels!!\n ")
    #    dispfirstonly = false 
    # end

    ## source location, etc.      
    mindistsrc = 1e-5
    onsrc = zeros(Bool,nx,ny)
    onsrc[:,:] .= false
    xsrc,ysrc=src[1],src[2]
    
    hgr = grd.hgrid/2.0
    ## grd.xinit-hgr because TIME array on STAGGERED grid
    ix,iy = findclosestnode(xsrc,ysrc,grd.xinit-hgr,grd.yinit-hgr,grd.hgrid) 
    rx = src[1]-((ix-1)*grd.hgrid+grd.xinit-hgr)
    ry = src[2]-((iy-1)*grd.hgrid+grd.yinit-hgr)
    halfg = 0.0 #hgrid/2.0

    
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
        # isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
        # for (j,i) in zip(jsrc,isrc)
        ijsrc = findall(onsrc)
        @inbounds for lcart in ijsrc
            i = lcart[1]
            j = lcart[2]
            xp = (i-1)*grd.hgrid+grd.xinit-hgr
            yp = (j-1)*grd.hgrid+grd.yinit-hgr
            ii = i-1 #Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
            jj = j-1 #Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1             
            #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
            ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]#[isrc[1,1],jsrc[1,1]]
                #println("$i $j  $xp $yp")
        end
    end
        
    ######################################################

    iswe = [1 nx 1; nx 1 -1; nx 1 -1; 1  nx 1]
    jswe = [1 ny 1; 1 ny 1;  ny 1 -1; ny 1 -1]
    
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
    ttimeold = Array{Float64,2}(undef,nx,ny)
    #--------------------------------------
 
    #pa=0
    swedifference = 100.0*epsilon
    while swedifference>epsilon
        #pa +=1
        ttimeold[:,:] = ttime
        
        @inbounds for swe=1:4 
            
            @inbounds  for j=jswe[swe,1]:jswe[swe,3]:jswe[swe,2]
                @inbounds for i=iswe[swe,1]:iswe[swe,3]:iswe[swe,2]
                    
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
                    @inbounds for cou=1:8
                        
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

function ttFMM_podlec(vel::Array{Float64,2},src::Array{Float64,1},grd::Grid2D) 
 
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
    mindistsrc = 1e-5
    onsrc = zeros(Bool,nx,ny)
    onsrc[:,:] .= false
    xsrc,ysrc=src[1],src[2]
    
    hgr = grd.hgrid/2.0
    ## grd.xinit-hgr because TIME array on STAGGERED grid
    ix,iy = findclosestnode(xsrc,ysrc,grd.xinit-hgr,grd.yinit-hgr,grd.hgrid)
    rx = src[1]-((ix-1)*grd.hgrid+grd.xinit-hgr)
    ry = src[2]-((iy-1)*grd.hgrid+grd.yinit-hgr)
    halfg = 0.0 #hgrid/2.0

    
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
        #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
        ijsrc = findall(onsrc)
        @inbounds for lcart in ijsrc
            i = lcart[1]
            j = lcart[2]
            xp = (i-1)*grd.hgrid+grd.xinit-hgr
            yp = (j-1)*grd.hgrid+grd.yinit-hgr
            ii = i-1 #Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
            jj = j-1 #Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1             
            #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
            ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]#[isrc[1,1],jsrc[1,1]]
            #println("$i $j  $xp $yp")
        end
    end
        
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
    neigh = [1  0;
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
    # is = Array{Int64,1}(undef,naccinit)
    # js = Array{Int64,1}(undef,naccinit)
    # l=1
    # for j=1:nty,i=1:ntx
    #     if status[i,j]==2
    #         is[l] = i
    #         js[l] = j
    #         l+=1
    #     end
    # end

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
    @inbounds for l=1:naccinit ##
        
        @inbounds  for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                continue
            end

            if status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt!(ttime,ttlocmin,inittt,slowness,grd,cooa,coob,coovin,coovadj,i,j)
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
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

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

        ## try all neighbors of newly accepted point
        @inbounds for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt!(ttime,ttlocmin,inittt,slowness,grd,cooa,coob,coovin,coovadj,i,j)
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt!(ttime,ttlocmin,inittt,slowness,grd,cooa,coob,coovin,coovadj,i,j)
                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxny[i,j]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)
                
            end
        end
        ##-------------------------------
    end
    return ttime
end

##====================================================================##

function calcttpt!(ttime::Array{Float64,2},ttlocmin::Array{Float64,1},inittt::Float64,
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
    #ttlocmin = Array{Float64,1}(8)
    nvx = grd.nx
    nvy = grd.ny
    
    ## if on src, skip this iteration
    # onsrc[i,j]==true && continue
    
    ####################################################
    ##   Local solver (Podvin, Lecomte, 1991)         ##
    ####################################################
    
    ttlocmin[:] .= inittt  
    ttdiffr = inittt  
    ttheadw = inittt
    ttc = inittt
    
    ## loop on triangles
    @inbounds for cou=1:8
        
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

#########################################################################

function ttFMM_hiord(vel::Array{Float64,2},src::Array{Float64,1},
                       grd::Grid2D) 
 

    ## Sizes
    nx,ny=grd.nx,grd.ny #size(vel)  ## NOT A STAGGERED GRID!!!

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
    refinearoundsrc=true
    
    if refinearoundsrc
        ##---------------------------------
        ## 
        ## Refinement around the source      
        ##
        ttaroundsrc!(status,ttime,vel,src,grd,inittt)
        # println("\n\n#### DELETE using PyPlot at the TOP of eikonalforward2D.jl\n\n")
        # figure()
        # subplot(121)
        # title("ttime")
        # imshow(ttime,vmax=30)
        # colorbar()
        # subplot(122)
        # title("status")
        # imshow(status)
        # colorbar()
        

    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        
        ## source location, etc.      
        mindistsrc = 1e-5
        onsrc = zeros(Bool,nx,ny)
        onsrc[:,:] .= false
        xsrc,ysrc=src[1],src[2]
        
        ix,iy = findclosestnode(xsrc,ysrc,grd.xinit,grd.yinit,grd.hgrid) 
        rx = src[1]-((ix-1)*grd.hgrid+grd.xinit)
        ry = src[2]-((iy-1)*grd.hgrid+grd.yinit)
        halfg = 0.0 #hgrid/2.0

        
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
            #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
            ijsrc = findall(onsrc)
            @inbounds for lcart in ijsrc
                i = lcart[1]
                j = lcart[2]
                xp = (i-1)*grd.hgrid+grd.xinit
                yp = (j-1)*grd.hgrid+grd.yinit
                ii = Int(floor((xsrc-grd.xinit)/grd.hgrid)) +1
                jj = Int(floor((ysrc-grd.yinit)/grd.hgrid)) +1             
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / vel[ii,jj]#[isrc[1,1],jsrc[1,1]]
                #println("$i $j  $xp $yp")
            end
        end

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

    ## Init the min binary heap with void arrays but max size
    Nmax=nx*ny
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## conversion cart to lin indices, old sub2ind
    linid_nxny = LinearIndices((nx,ny))
    ## conversion lin to cart indices, old ind2sub
    cartid_nxny = CartesianIndices((nx,ny))
    
    ## construct initial narrow band
    @inbounds for l=1:naccinit ##
        
        @inbounds for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            

            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                    continue
            end

            if status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j)
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
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

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

        ## try all neighbors of newly accepted point
        @inbounds for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j)
                han = linid_nxny[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j)
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

##====================================================================##

function isonbord(ib::Int64,jb::Int64,nx::Int64,ny::Int64)
    isonb1st = false
    isonb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>nx || jb<1 || jb>ny
        isonb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>nx-1 || jb<2 || jb>ny-1
        isonb2nd =  true
    end
    return isonb1st,isonb2nd
end

##====================================================================##

"""
   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order.
"""
function calcttpt_2ndord(ttime::Array{Float64,2},vel::Array{Float64,2},
                         grd::Grid2D, status::Array{Int64,2},i::Int64,j::Int64)
    
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

    ## 2 directions
    @inbounds for axis=1:2
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## two sides for each direction
        @inbounds for l=1:2

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
                        else
                            chosenval2=HUGE
                            use2ndord=false # this is needed!
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

            chovalxy = zeros(2)
            
            ## 2 directions
            @inbounds for axis=1:2
                
                use1stord = false
                chosenval1 = HUGE
                
                ## two sides for each direction
                @inbounds for l=1:2

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
                
                chovalxy[axis] = chosenval1
            end
            
        end ## begin...

        ## recompute sqarg
        sqarg = beta^2-4.0*alpha*gamma

        if sqarg<0.0
            println("\n To get a non-negative discriminant, need to fulfil: ")
            println(" (tx-ty)^2 - 2*s^2/curalpha <= 0")
            println(" where tx,ty can be")
            println(" t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order")
            println(" t? = chosenval1  if 1st order ")
            error("sqarg<0.0")
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

##====================================================================##


"""
  Refinement of the grid around the source. FMM calculated inside a finer grid 
    and then passed on to coarser grid
"""
function ttaroundsrc!(statuscoarse::Array{Int64,2},ttimecoarse::Array{Float64,2},
    vel::Array{Float64,2},src::Array{Float64,1},grdcoarse::Grid2D,inittt::Float64)
      
    ##
    ## 2x10 nodes -> 2x50 nodes
    ##
    downscalefactor::Int = 5
    noderadius::Int = 5

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

    # @show ixsrcglob,iysrcglob
    # @show i1coarse,i2coarse,j1coarse,j2coarse
    # @show nx,ny
    xinit = 0.0
    yinit = 0.0
    grdfine = Grid2D(dh,xinit,yinit,nx,ny)

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
    xorig = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    yorig = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    xsrc = src[1] - xorig - grdcoarse.xinit
    ysrc = src[2] - yorig - grdcoarse.yinit
    
    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,nx,ny)
    @inbounds for j=1:ny
     @inbounds for i=1:nx
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
    mindistsrc = 1e-6
    onsrc = zeros(Bool,nx,ny)
    onsrc[:,:] .= false
    ix,iy = findclosestnode(xsrc,ysrc,0.0,0.0,dh) 
    rx = xsrc-((ix-1)*dh)
    ry = ysrc-((iy-1)*dh)
    halfg = 0.0 #hgrid/2.0
    
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
        #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
        ijsrc = findall(onsrc)
        @inbounds for lcart in ijsrc
            i = lcart[1]
            j = lcart[2]
            xp = (i-1)*dh
            yp = (j-1)*dh
            ii = Int(floor(xsrc/dh)) +1
            jj = Int(floor(ysrc/dh)) +1             
            ttime[i,j] = sqrt((xsrc-xp)^2+(ysrc-yp)^2) / velfinegrd[ii,jj]
        end
    end
        
    ######################################################
  
    ##================================
    neigh = [1  0;
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

    ## Init the min binary heap with void arrays but max size
    Nmax=nx*ny
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## conversion cart to lin indices, old sub2ind
    linid_nxny = LinearIndices((nx,ny))
    ## conversion lin to cart indices, old ind2sub
    cartid_nxny = CartesianIndices((nx,ny))
    
    ## construct initial narrow band
    @inbounds for l=1:naccinit ##
        
         @inbounds for ne=1:4 ## four potential neighbors

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
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nx,ny),han)
        cija = cartid_nxny[han]
        ia,ja = cija[1],cija[2]
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt

        ##########################################################
        ##
        ## If the the accepted point is on the edge of the
        ##  fine grid, stop computing and jump to coarse grid
        ##
        ##########################################################
        #  if (ia==nx) || (ia==1) || (ja==ny) || (ja==1)
        if (ia==1 && !outxmin) || (ia==nx && !outxmax) || (ja==1 && !outymin) || (ja==ny && !outxmax)
            ttimecoarse[i1coarse:i2coarse,j1coarse:j2coarse]  =  ttime[1:downscalefactor:end,1:downscalefactor:end]
            statuscoarse[i1coarse:i2coarse,j1coarse:j2coarse] = status[1:downscalefactor:end,1:downscalefactor:end]
            ## delete current narrow band to avoid problems when returned to coarse grid
            statuscoarse[statuscoarse.==1] .= 0
            return ttime,status
        end
        ##########################################################

        ## try all neighbors of newly accepted point
        @inbounds for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                han = linid_nxny[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                # get handle
                han = linid_nxny[i,j]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------
    end

    return error("Ouch...")
    
end


#######################################################################################

#########################################################
#end                                                    #
#########################################################

