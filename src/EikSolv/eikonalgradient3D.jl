

#######################################################
##     Misfit of gradient using adjoint              ## 
#######################################################


###############################################################################
########################################################################################
## @doc raw because of some backslashes in the string...
@doc raw""" 
    gradttime3D(vel::Array{Float64,3},grd::Grid3D,coordsrc::Array{Float64,2},
                coordrec::Array{Float64,2},pickobs::Array{Float64,2},
                pickcalc::Array{Float64,2}, stdobs::Float64 ; 
                gradttalgo::String="gradFMM_hiord")

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
- `stdobs`: standard deviation of error on observed traveltime picks
- `gradttalgo`: the algorithm to use to compute the forward and gradient, one amongst the following
    * "gradFS\_podlec", fast sweeping method using Podvin-Lecomte stencils for forward, fast sweeping method for adjoint   
    * "gradFMM\_podlec," fast marching method using Podvin-Lecomte stencils for forward, fast marching method for adjoint 
    * "gradFMM\_hiord", second order fast marching method for forward, high order fast marching method for adjoint  

# Returns
- `grad`: the gradient as a 3D array

"""
function gradttime3D(vel::Array{Float64,3},grd::Grid3D,coordsrc::Array{Float64,2},coordrec::Array{Float64,2},
                     pickobs::Array{Float64,2},stdobs::Float64 ; gradttalgo::String="gradFMM_hiord")
   
    @assert size(coordsrc,2)==3
    @assert size(coordrec,2)==3
    @assert all(vel.>=0.0)
    @assert all(grd.xinit.<coordsrc[:,1].<((grd.nx-1)*grd.hgrid+grd.xinit))
    @assert all(grd.yinit.<coordsrc[:,2].<((grd.ny-1)*grd.hgrid+grd.yinit))
    @assert all(grd.zinit.<coordsrc[:,3].<((grd.nz-1)*grd.hgrid+grd.zinit))

    nsrc=size(coordsrc,1)
    nw = nworkers()

    ## calculate how to subdivide the srcs among the workers
    grpsrc = distribsrcs(nsrc,nw)
    nchu = size(grpsrc,1)
    ## array of workers' ids
    wks = workers()
    
    tmpgrad = zeros(grd.nx,grd.ny,grd.nz,nchu)
    ## do the calculations
    @sync begin 
        for s=1:nchu
            igrs = grpsrc[s,1]:grpsrc[s,2]
            @async tmpgrad[:,:,:,s] = remotecall_fetch(calcgradsomesrc3D,wks[s],vel,
                                                     coordsrc[igrs,:],coordrec,
                                                     grd,stdobs,pickobs[:,igrs],
                                                     gradttalgo )
        end
    end
    grad = sum(tmpgrad,dims=4)
    return grad
end

################################################################3

"""
Calculate the gradient for some requested sources 
"""
function calcgradsomesrc3D(vel::Array{Float64,3},xysrc::Array{Float64,2},
                         coordrec::Array{Float64,2},
                         grd::Grid3D,
                         stdobs::Float64,pickobs1::Array{Float64,2},
                         adjalgo::String)

    nx,ny,nz=size(vel)
    ttpicks1 = zeros(size(coordrec,1))
    nsrc = size(xysrc,1)
    grad1 = zeros(nx,ny,nz)


    # looping on 1...nsrc because only already selected srcs have been
    #   passed to this routine
    for s=1:nsrc

        ###########################################
        ## calc ttime
    
        if adjalgo=="gradFMM_podlec"
            ttonesrc = ttFMM_podlec(vel,xysrc[s,:],grd)
 
        elseif adjalgo=="gradFMM_hiord"
            ttonesrc = ttFMM_hiord(vel,xysrc[s,:],grd)

        elseif adjalgo=="gradFS_podlec"
            ttonesrc = ttFS_podlec(vel,xysrc[s,:],grd)
 
        else
            println("\ncalcgradsomesrc(): Wrong adjalgo name: $(adjalgo)... \n")
            return nothing
        end
            
        ## ttime at receivers
        for i=1:size(coordrec,1)
            ttpicks1[i] = trilinear_interp( ttonesrc,grd.hgrid,grd.xinit,
                                            grd.yinit,grd.zinit,coordrec[i,1],
                                            coordrec[i,2],coordrec[i,3] )
        end

        ###########################################
        ## compute gradient for last ttime
        ##  add gradients from different sources

        if adjalgo=="gradFS_podlec"

            grad1 += eikgrad_FS_SINGLESRC(ttonesrc,vel,xysrc[s,:],coordrec,grd,
                                     pickobs1[:,s],ttpicks1,stdobs)

        elseif adjalgo=="gradFMM_podlec"

            grad1 += eikgrad_FMM_SINGLESRC(ttonesrc,vel,xysrc[s,:],coordrec,
                                          grd,pickobs1[:,s],ttpicks1,stdobs)
            
        elseif adjalgo=="gradFMM_hiord"
            grad1 += eikgrad_FMM_hiord_SINGLESRC(ttonesrc,vel,xysrc[s,:],coordrec,
                                                  grd,pickobs1[:,s],ttpicks1,stdobs)

        else
            println("Wrong adjalgo algo name: $(adjalgo)... ")
            return
        end
        
    end
    
    return grad1
end

############################################################################

function sourceboxlocgrad!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Array{Float64,1},
                           grd::Grid3D; staggeredgrid::Bool )

    ##########################
    ##   Init source
    ##########################
    mindistsrc = 1e-5
    xsrc,ysrc,zsrc = srcpos[1],srcpos[2],srcpos[3]
    dh = grd.hgrid

    if staggeredgrid==false
        ## regular grid
        onsrc = zeros(Bool,grd.nx,grd.ny,grd.nz)
        onsrc[:,:,:] .= false
        ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
        rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)
        max_x = (grd.nx-1)*grd.hgrid+grd.xinit
        max_y = (grd.ny-1)*grd.hgrid+grd.yinit
        max_z = (grd.nz-1)*grd.hgrid+grd.zinit
        
    elseif staggeredgrid==true
        ## STAGGERED grid
        onsrc = zeros(Bool,grd.ntx,grd.nty,grd.ntz)
        onsrc[:,:,:] .= false
        ## grd.xinit-hgr because TIME array on STAGGERED grid
        hgr = grd.hgrid/2.0
        ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
        rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
        ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
        rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)
        max_x = (grd.ntx-1)*grd.hgrid+grd.xinit
        max_y = (grd.nty-1)*grd.hgrid+grd.yinit
        max_z = (grd.ntz-1)*grd.hgrid+grd.zinit
    end

    halfg = 0.0
    src_on_nodeedge = false

    ## loop to make sure we don't accidentally move the src to another node/edge
    while (sqrt(rx^2+ry^2+rz^2)<=mindistsrc) || (abs(rx)<=mindistsrc) || (abs(ry)<=mindistsrc) || (abs(rz)<=mindistsrc)
        src_on_nodeedge = true
        
        ## shift the source 
        if xsrc < max_x-0.002*dh
            xsrc = xsrc+0.001*dh
        else #(make sure it's not already at the bottom y)
            xsrc = xsrc-0.001*dh
        end
        if ysrc < max_y-0.002*dh
            ysrc = ysrc+0.001*dh
        else #(make sure it's not already at the bottom y)
            ysrc = ysrc-0.001*dh
        end
        if zsrc < max_z-0.002*dh
            zsrc = zsrc+0.001*dh
        else #(make sure it's not already at the bottom z)
            zsrc = zsrc-0.001*dh
        end

        # print("new time at src:  $(tt[ix,iy]) ")
        ## recompute parameters related to position of source
        if staggeredgrid==false
            ## regular grid
            ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
            rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
            ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
            rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)        
        elseif staggeredgrid==true
            ## STAGGERED grid
            hgr = grd.hgrid/2.0
            ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
            rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
            ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
            rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)        
        end
    end

    ## To avoid singularities, the src can only be inside a box,
    ##  not on a grid node or edge... see above. So we need to have
    ##  always 4 nodes where onsrc is true
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


    ## if src on node or edge, recalculate traveltimes in box
    if src_on_nodeedge==true
        ## RE-set a new ttime around source 
        #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
        ijksrc = findall(onsrc)
        # println(" set time around src $isrc  $jsrc $ksrc") 
        #for (k,j,i) in zip(ksrc,jsrc,isrc)
        for lcart in ijksrc
            # for j in jsrc
            #     for i in isrc
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
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
            elseif staggeredgrid==true
                ## STAGGERED grid
                xp = (i-1)*grd.hgrid+grd.xinit-hgr
                yp = (j-1)*grd.hgrid+grd.yinit-hgr
                zp = (k-1)*grd.hgrid+grd.zinit-hgr
                ii = i-1 ##Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
                jj = j-1 ##Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
                kk = k-1 ##Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
                #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
                ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
            end
        end
    end#

    return onsrc
end

############################################################################

function recboxlocgrad!(lambda::Array{Float64,3},ttpicks::Array{Float64,1},grd::Grid3D,
                        rec::Array{Float64,2},pickobs::Array{Float64,1},stdobs::Float64; staggeredgrid::Bool)

    ##########################
    ## Init receivers
    ##########################

    if staggeredgrid==false
        ## regular grid
        onarec = zeros(Bool,grd.nx,grd.ny,grd.nz)
        nxmax = grd.nx
        nymax = grd.ny
        nzmax = grd.nz
    elseif staggeredgrid==true
        ## STAGGERED grid
        onarec = zeros(Bool,grd.ntx,grd.nty,grd.ntz)
        nxmax = grd.ntx
        nymax = grd.nty
        nzmax = grd.ntz
        hgr = grd.hgrid/2.0
    end
    onarec[:,:,:] .= false
    nrec=size(rec,1)
    
   ## init receivers
    nrec=size(rec,1)
    for r=1:nrec
        if staggeredgrid==false
            i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,2],grd.xinit,grd.yinit,grd.zinit,grd.hgrid)
        elseif staggeredgrid==true
            i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,2],grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid)
        end
        if (i==1) || (i==nxmax) || (j==1) || (j==nymax) || (k==1) || (k==nzmax)
            println(" Receiver on border of model (i==1)||(i==ntx)||(j==1)||(j==nty || (k==1) || (k==ntz))")
            println(" Not yet implemented...")
            return nothing
        end
        onarec[i,j,k] = true
        lambda[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs^2
    end
  
    return onarec
end

############################################################################

function eikgrad_FS_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                         src::Array{Float64,1},rec::Array{Float64,2},grd::Grid3D,
                         pickobs::Array{Float64,1},ttpicks::Array{Float64,1},stdobs::Float64)

    @assert size(src)==(3,)
    
    @assert size(ttime)==(size(vel).+1)
    ##nx,ny,nz = size(ttime)

    epsilon = 1e-5
    mindistsrc = 1e-5
    
    ## init adjoint variable
    lambdaold = zeros((grd.ntx,grd.nty,grd.ntz)) .+ 1e32
    lambdaold[:,1,:]   .= 0.0
    lambdaold[:,end,:] .= 0.0
    lambdaold[1,:,:]   .= 0.0
    lambdaold[end,:,:] .= 0.0
    lambdaold[:,:,1]   .= 0.0
    lambdaold[:,:,end] .= 0.0

 
    #copy ttime 'cause it might be changed around src
    tt = copy(ttime)
    dh = grd.hgrid

    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    zinit = grd.zinit
    ntx = grd.ntx
    nty = grd.nty
    ntz = grd.ntz
    
    ## source
    ## STAGGERED grid
    onsrc = sourceboxlocgrad!(tt,vel,src,grd; staggeredgrid=true )
    # receivers
    # lambdaold!!!!
    onarec = recboxlocgrad!(lambdaold,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=true)

    xsrc,ysrc,zsrc = src[1],src[2],src[3]


    # ## grd.xinit-hgr because TIME array on STAGGERED grid
    # hgr = grd.hgrid/2.0
    # ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
    # rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
    # ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
    # rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)
    # #@show dist,src,rx,ry
    # onsrc = zeros(Bool,nx,ny,nz)
    # onsrc[:,:,:] .= false    
    # halfg = 0.0

    # #@show xsrc,ysrc,zsrc,rx,ry,rz
    
    # src_on_nodeedge = false
    # max_x = (grd.ntx-1)*grd.hgrid+grd.xinit
    # max_y = (grd.nty-1)*grd.hgrid+grd.yinit
    # max_z = (grd.ntz-1)*grd.hgrid+grd.zinit
    # ## loop to make sure we don't accidentally move the src to another node/edge
    # while (sqrt(rx^2+ry^2+rz^2)<=mindistsrc) || (abs(rx)<=mindistsrc) || (abs(ry)<=mindistsrc) || (abs(rz)<=mindistsrc)
    #     src_on_nodeedge = true
        
    #     ## shift the source 
    #     if xsrc < max_x-0.002*dh
    #         xsrc = xsrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         xsrc = xsrc-0.001*dh
    #     end
    #     if ysrc < max_y-0.002*dh
    #         ysrc = ysrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         ysrc = ysrc-0.001*dh
    #     end
    #     if zsrc < max_z-0.002*dh
    #         zsrc = zsrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         zsrc = zsrc-0.001*dh
    #     end

    #     # print("new time at src:  $(tt[ix,iy]) ")
    #     ## recompute parameters related to position of source
    #     ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
    #     rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
    #     ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
    #     rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)

    #     #@show xsrc,ysrc,zsrc,rx,ry,rz
    # end

    
    # ## To avoid singularities, the src can only be inside a box,
    # ##  not on a grid node or edge... see above. So we need to have
    # ##  always 8 nodes (corners) where onsrc is true
    # if (rx>=halfg) & (ry>=halfg) & (rz>=halfg)
    #     onsrc[ix:ix+1,iy:iy+1,iz:iz+1] .= true
    # elseif (rx<halfg) & (ry>=halfg) & (rz>=halfg)
    #     onsrc[ix-1:ix,iy:iy+1,iz:iz+1] .= true
    # elseif (rx<halfg) & (ry<halfg) & (rz>=halfg)
    #     onsrc[ix-1:ix,iy-1:iy,iz:iz+1] .= true
    # elseif (rx>=halfg) & (ry<halfg) & (rz>=halfg)
    #     onsrc[ix:ix+1,iy-1:iy,iz:iz+1] .= true

    # elseif (rx>=halfg) & (ry>=halfg) & (rz<halfg)
    #     onsrc[ix:ix+1,iy:iy+1,iz-1:iz] .= true
    # elseif (rx<halfg) & (ry>=halfg) & (rz<halfg)
    #     onsrc[ix-1:ix,iy:iy+1,iz-1:iz] .= true
    # elseif (rx<halfg) & (ry<halfg) & (rz<halfg)
    #     onsrc[ix-1:ix,iy-1:iy,iz-1:iz] .= true
    # elseif (rx>=halfg) & (ry<halfg) & (rz<halfg)
    #     onsrc[ix:ix+1,iy-1:iy,iz-1:iz] .= true
    # end

    # ## if src on node or edge, recalculate traveltimes in box
    # if src_on_nodeedge==true
    #     ## RE-set a new ttime around source 
    #     # isrc,jsrc,ksrc = ind2sub(size(onsrc),find(onsrc))
    #     ijksrc = findall(onsrc)
    #     for lcart in ijksrc
    #         i = lcart[1]
    #         j = lcart[2]
    #         k = lcart[3]
    #         xp = (i-1)*grd.hgrid+grd.xinit
    #         yp = (j-1)*grd.hgrid+grd.yinit
    #         zp = (k-1)*grd.hgrid+grd.zinit
    #         ii = Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
    #         jj = Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
    #         kk = Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
    #         #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
    #         ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
    #         ##println("$i $j $j  $xp $yp $zp")
    #     end
    # end

    # ## init receivers
    # nrec=size(rec,1)
    # for r=1:nrec
    #     i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,3],xinit,yinit,zinit,grd.hgrid)
    #     onarec[i,j,k] = true
    #     dtonarec[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs^2
    #     lambdaold[i,j,k] = dtonarec[i,j,k]
    #     #@show r,i,j,k
    # end
  
    ######################################################
 
    ##============================================================================

    iswe = [2 ntx-1  1; 2 ntx-1  1; 2 ntx-1  1; 2 ntx-1  1;
            ntx-1 2 -1; ntx-1 2 -1; ntx-1 2 -1; ntx-1 2 -1 ] 
    jswe = [2 nty-1  1; 2 nty-1  1; nty-1 2 -1; nty-1 2 -1;
            2 nty-1  1; 2 nty-1  1; nty-1 2 -1; nty-1 2 -1 ]
    kswe = [2 ntz-1  1; ntz-1 2 -1; 2 ntz-1  1; ntz-1 2 -1;
            2 ntz-1  1; ntz-1 2 -1; 2 ntz-1  1; ntz-1 2 -1 ]

    ##=============================================================================    
  
    lambdanew=copy(lambdaold)
   
    #############################
    pa=0
    swedifference = 100*epsilon
    while swedifference>epsilon 
        pa +=1
        lambdaold[:,:,:] =  lambdanew

        for swe=1:8

            for k=kswe[swe,1]:kswe[swe,3]:kswe[swe,2]            
                for j=jswe[swe,1]:jswe[swe,3]:jswe[swe,2]
                    for i=iswe[swe,1]:iswe[swe,3]:iswe[swe,2]
                        
                        if onsrc[i,j,k]==true

                                                        
                            ## If we are in the square containing the source,
                            ## use real position of source for the derivatives.
                            ## Calculate tt on x and y on the edges of the square
                            ##  H is the projection of the point src onto X, Y and Z edges
                            xp = Float64(i-1)*dh+xinit
                            yp = Float64(j-1)*dh+yinit
                            zp = Float64(k-1)*dh+zinit

                            distPSx = abs(xp-xsrc) ## abs needed for later...
                            distPSy = abs(yp-ysrc)
                            distPSz = abs(zp-zsrc)
                            dist2src = sqrt(distPSx^2+distPSy^2+distPSz^2)

                            ## Calculate the traveltime to hit the x edge
                            ## time at H along x
                            thx = tt[i,j,k]*(sqrt(distPSy^2+distPSz^2))/dist2src
                            ## Calculate the traveltime to hit the y edge
                            ## time at H along y
                            thy = tt[i,j,k]*(sqrt(distPSx^2+distPSz^2))/dist2src
                            ## Calculate the traveltime to hit the z edge
                            ## time at H along z
                            thz = tt[i,j,k]*(sqrt(distPSx^2+distPSy^2))/dist2src
                            
                            if onsrc[i+1,j,k]==true              
                                aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
                                aforw = -(thx     -tt[i,j,k])/distPSx
                            elseif onsrc[i-1,j,k]==true
                                aback = -(tt[i,j,k]  -thx)/distPSx
                                aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
                            end
                            if onsrc[i,j+1,k]==true
                                bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
                                bforw = -(thy     -tt[i,j,k])/distPSy
                            else onsrc[i,j-1,k]==true
                                bback = -(tt[i,j,k]  -thy)/distPSy 
                                bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
                            end
                            if onsrc[i,j,k+1]==true     
                                cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
                                cforw = -(thz      -tt[i,j,k])/distPSz
                            elseif onsrc[i,j,k-1]==true
                                cback = -(tt[i,j,k]  -thz)/distPSz
                                cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
                            end
                            
                        else
                            ## along x
                            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
                            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
                            ## along y
                            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
                            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
                            ## along z
                            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
                            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
                        end

                        ##-----------------------------------
                        aforwplus  = ( aforw+abs(aforw) )/2.0
                        aforwminus = ( aforw-abs(aforw) )/2.0
                        abackplus  = ( aback+abs(aback) )/2.0
                        abackminus = ( aback-abs(aback) )/2.0

                        bforwplus  = ( bforw+abs(bforw) )/2.0
                        bforwminus = ( bforw-abs(bforw) )/2.0
                        bbackplus  = ( bback+abs(bback) )/2.0
                        bbackminus = ( bback-abs(bback) )/2.0

                        cforwplus  = ( cforw+abs(cforw) )/2.0
                        cforwminus = ( cforw-abs(cforw) )/2.0
                        cbackplus  = ( cback+abs(cback) )/2.0
                        cbackminus = ( cback-abs(cback) )/2.0
                        
                        ##-------------------------------------------

                        numer =
                            (abackplus * lambdanew[i-1,j,k] - aforwminus * lambdanew[i+1,j,k]) / dh +
                            (bbackplus * lambdanew[i,j-1,k] - bforwminus * lambdanew[i,j+1,k]) / dh +
                            (cbackplus * lambdanew[i,j,k-1] - cforwminus * lambdanew[i,j,k+1]) / dh
                        
                        denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
                            (cforwplus - cbackminus)/dh
                        
                        ###################################################################
                        ###################################################################

                        # if (i<6 && j==2 && k==2 && pa==1)
                        #     println("$i,$j,$k numer, denom $numer $denom")
                        # end
                            
                        # if( abs(denom)<=1e-9 ) 
                        #     println("\n$i,$j: (abs(denom)<=1e-9)  ")
                        #     @show swe,i,j,k,numer,denom,onsrc[i,j,k],aback,aforw,bback,bforw,cback,cforw
                        #     @show aforwplus,aforwminus,abackplus,abackminus,cbackplus,cbackminus
                        #     @show bforwplus,bforwminus,bbackplus,bbackminus,cbackplus,cbackminus
                        #     @show tt[i,j,k]#,thx,thy,thz
                        # end
                        

                        ####################################################
                        
                        if onarec[i,j,k]==true                            
                            #gradT = sqrt( (tt[i+1,j]-tt[i,j])^2 + (tt[i,j+1]-tt[i,j])^2 )
                            # numer2 = numer + dtonarec[i,j,k]
                            # lambdaprop = numer2/denom 
                            # lambdanew[i,j,k] = min(lambdaold[i,j,k],lambdaprop                                                   )
                            lambdanew[i,j,k] = lambdaold[i,j,k]
                        else 
                            lambdaprop = numer/denom
                            lambdanew[i,j,k] = min(lambdaold[i,j,k],lambdaprop)
                        end
  
                    end
                end
            end
        end
        ##################################################
        # println(">>>>err=',NP.sum(abs(lambdanew-lambdaold)),'\n'
        # print 'max old new',lambdanew.max(),lambdaold.max()
        swedifference = sum(abs.(lambdanew.-lambdaold))
        ##################################################
    end

    ## STAGGERED GRID, so lambdanew[1:end-1,1:end-1]... any better idea? Interpolation?
    # gradadj = -lambdanew[1:end-1,1:end-1,1:end-1]./vel.^3
    # Average grad at the center of cells (velocity cells -STAGGERED GRID)
    averlambda = ( lambdanew[1:end-1, 1:end-1, 1:end-1].+
                   lambdanew[2:end,   1:end-1, 1:end-1].+
                   lambdanew[1:end-1, 2:end,   1:end-1].+
                   lambdanew[2:end,   2:end,   1:end-1].+
                   lambdanew[1:end-1, 1:end-1, 2:end].+
                   lambdanew[2:end,   1:end-1, 2:end].+
                   lambdanew[1:end-1, 2:end,   2:end].+
                   lambdanew[2:end,   2:end,   2:end] ) ./ 8.0
    gradadj = -averlambda./vel.^3
        
    return gradadj
end


#############################################################################


function eikgrad_FMM_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                         src::Array{Float64,1},rec::Array{Float64,2},
                         grd::Grid3D,pickobs::Array{Float64,1},
                         ttpicks::Array{Float64,1},stdobs::Float64)

    @assert size(src)==(3,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel).+1)
    ##nx,ny,nz = size(ttime)

    lambda = zeros((grd.ntx,grd.nty,grd.ntz)) 
 
    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    ntx = grd.ntx
    nty = grd.nty
    ntz = grd.ntz
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    zinit = grd.zinit

    ## source
    ## source
    ## STAGGERED grid
    onsrc = sourceboxlocgrad!(tt,vel,src,grd, staggeredgrid=true )
    ## receivers
    onarec = recboxlocgrad!(lambda,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=true)
    
    
    xsrc,ysrc,zsrc = src[1],src[2],src[3]



    # ## grd.xinit-hgr because TIME array on STAGGERED grid
    # hgr = grd.hgrid/2.0
    # ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
    # rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
    # ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
    # rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)
    # #@show dist,src,rx,ry
    # onsrc = zeros(Bool,nx,ny,nz)
    # onsrc[:,:,:] .= false    
    # halfg = 0.0
    
    # src_on_nodeedge = false
    # max_x = (grd.ntx-1)*grd.hgrid+grd.xinit
    # max_y = (grd.nty-1)*grd.hgrid+grd.yinit
    # max_z = (grd.ntz-1)*grd.hgrid+grd.zinit

    # ## loop to make sure we don't accidentally move the src to another node/edge
    # while (sqrt(rx^2+ry^2+rz^2)<=mindistsrc) || (abs(rx)<=mindistsrc) || (abs(ry)<=mindistsrc) || (abs(rz)<=mindistsrc)
    #     src_on_nodeedge = true
        
    #     ## shift the source 
    #     if xsrc < max_x-0.002*dh
    #         xsrc = xsrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         xsrc = xsrc-0.001*dh
    #     end
    #     if ysrc < max_y-0.002*dh
    #         ysrc = ysrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         ysrc = ysrc-0.001*dh
    #     end
    #     if zsrc < max_z-0.002*dh
    #         zsrc = zsrc+0.001*dh
    #     else #(make sure it's not already at the bottom z)
    #         zsrc = zsrc-0.001*dh
    #     end

    #     # print("new time at src:  $(tt[ix,iy]) ")
    #     ## recompute parameters related to position of source
    #     ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit-hgr,grd.yinit-hgr,grd.zinit-hgr,grd.hgrid) 
    #     rx = xsrc-((ix-1)*grd.hgrid+grd.xinit-hgr)
    #     ry = ysrc-((iy-1)*grd.hgrid+grd.yinit-hgr)
    #     rz = zsrc-((iz-1)*grd.hgrid+grd.zinit-hgr)        

    # end

    # ## To avoid singularities, the src can only be inside a box,
    # ##  not on a grid node or edge... see above. So we need to have
    # ##  always 4 nodes where onsrc is true
    # if (rx>=halfg) & (ry>=halfg) & (rz>=halfg)
    #     onsrc[ix:ix+1,iy:iy+1,iz:iz+1] .= true
    # elseif (rx<halfg) & (ry>=halfg) & (rz>=halfg)
    #     onsrc[ix-1:ix,iy:iy+1,iz:iz+1] .= true
    # elseif (rx<halfg) & (ry<halfg) & (rz>=halfg)
    #     onsrc[ix-1:ix,iy-1:iy,iz:iz+1] .= true
    # elseif (rx>=halfg) & (ry<halfg) & (rz>=halfg)
    #     onsrc[ix:ix+1,iy-1:iy,iz:iz+1] .= true

    # elseif (rx>=halfg) & (ry>=halfg) & (rz<halfg)
    #     onsrc[ix:ix+1,iy:iy+1,iz-1:iz] .= true
    # elseif (rx<halfg) & (ry>=halfg) & (rz<halfg)
    #     onsrc[ix-1:ix,iy:iy+1,iz-1:iz] .= true
    # elseif (rx<halfg) & (ry<halfg) & (rz<halfg)
    #     onsrc[ix-1:ix,iy-1:iy,iz-1:iz] .= true
    # elseif (rx>=halfg) & (ry<halfg) & (rz<halfg)
    #     onsrc[ix:ix+1,iy-1:iy,iz-1:iz] .= true
    # end


    # ## if src on node or edge, recalculate traveltimes in box
    # if src_on_nodeedge==true
    #     ## RE-set a new ttime around source 
    #     #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
    #     ijksrc = findall(onsrc)
    #     # println(" set time around src $isrc  $jsrc $ksrc") 
    #     #for (k,j,i) in zip(ksrc,jsrc,isrc)
    #     for lcart in ijksrc
    #         # for j in jsrc
    #         #     for i in isrc
    #         i = lcart[1]
    #         j = lcart[2]
    #         k = lcart[3]
    #         xp = (i-1)*grd.hgrid+grd.xinit-hgr
    #         yp = (j-1)*grd.hgrid+grd.yinit-hgr
    #         zp = (k-1)*grd.hgrid+grd.zinit-hgr
    #         ii = i-1 ##Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
    #         jj = j-1 ##Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
    #         kk = k-1 ##Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
    #         #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
    #         ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
    #     end
    # end# 
        
    # ##-------------------------------------------
    # ## init receivers
    # nrec=size(rec,1)
    # for r=1:nrec
    #     i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,2],grd.xinit,grd.yinit,grd.zinit,grd.hgrid)
    #     if (i==1) || (i==ntx) || (j==1) || (j==nty) || (k==1) || (k==ntz)
    #         println(" Receiver on border of model (i==1)||(i==ntx)||(j==1)||(j==nty || (k==1) || (k==ntz))")
    #         println(" Not yet implemented...")
    #         return nothing
    #     end
    #     onarec[i,j,k] = true
    #     ##dtonarec[i,j] = (ttpicks[r]-pickobs[r])/stdobs^2
    #     lambda[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs^2
    # end
  
    ######################################################
    #-------------------------------
    ## init FMM 
    neigh = [1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]

    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,ntx,nty,ntz)
    status[:,:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :, : ] .= 2 ## set to accepted on boundary
    status[end,:, : ] .= 2 ## set to accepted on boundary
    status[:, 1, : ] .= 2 ## set to accepted on boundary
    status[:, end,: ] .= 2 ## set to accepted on boundary
    status[:, :, 1 ] .= 2 ## set to accepted on boundary
    status[:, :, end] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nx,nty),find(status.==2))
    ijkrec = findall(status.==2)
    irec = [x[1] for x in ijkrec]
    jrec = [x[2] for x in ijkrec]
    krec = [x[3] for x in ijkrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=ntx*nty*ntz
    bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## conversion cart to lin indices, old sub2ind
    linid_ntxntyntz = LinearIndices((ntx,nty,ntz))
    ## conversion lin to cart indices, old sub2ind
    cartid_ntxntyntz = CartesianIndices((ntx,nty,ntz))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:4 ## four potential neighbors
            
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            k = krec[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For the adjoint SKIP BORDERS!!!
            if (i>ntx-1) || (i<2) || (j>nty-1) || (j<2) || (k>ntz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far

                # get handle
                #han = sub2ind((ntx,nty),i,j)
                han = linid_ntxntyntz[i,j,k]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = ntx*nty*ntz
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 3D indices from handle
        cijka = cartid_ntxntyntz[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted     
        status[ia,ja,ka] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA!(tt,status,onsrc,onarec,dh,
                    xinit,yinit,zinit,xsrc,ysrc,zsrc,lambda,ia,ja,ka)
        
        ## try all neighbors of newly accepted point
        for ne=1:4 
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
 
            ## if the point is out of bounds skip this iteration
            ## For the adjoint SKIP BORDERS!!!
            if (i>ntx-1) || (i<2) || (j>nty-1) || (j<2) || (k>ntz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far, active

                # get handle
                #han = sub2ind((ntx,nty),i,j)
                han = linid_ntxntyntz[i,j,k] #i+ntx*(j-1)
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1
                
                ## NO NEED to update in narrow band because
                ##   we already know the time tt everywhere...
            end
        end
    end

    ## STAGGERED GRID, so lambdanew[1:end-1,1:end-1]... any better idea? Interpolation?
    # gradadj = -lambdanew[1:end-1,1:end-1]./vel.^3
    # Average grad at the center of cells (velocity cells -STAGGERED GRID)
    averlambda = ( lambda[1:end-1, 1:end-1, 1:end-1].+
                   lambda[2:end,   1:end-1, 1:end-1].+
                   lambda[1:end-1, 2:end,   1:end-1].+
                   lambda[2:end,   2:end,   1:end-1].+
                   lambda[1:end-1, 1:end-1, 2:end].+
                   lambda[2:end,   1:end-1, 2:end].+
                   lambda[1:end-1, 2:end,   2:end].+
                   lambda[2:end,   2:end,   2:end] ) ./ 8.0
    gradadj = -averlambda./vel.^3
        
    return gradadj
end # eikadj_FMM_SINGLESRC  


#############################################################

function calcLAMBDA!(tt::Array{Float64,3},status::Array{Int64,3},onsrc::Array{Bool,3},onarec::Array{Bool,3},
                     dh::Float64,xinit::Float64,yinit::Float64,zinit::Float64,xsrc::Float64,ysrc::Float64,zsrc::Float64,
                     lambda::Array{Float64,3},i::Int64,j::Int64,k::Int64)

    
    if onsrc[i,j,k]==true 

        
        ## If we are in the square containing the source,
        ## use real position of source for the derivatives.
        ## Calculate tt on x and y on the edges of the square
        ##  H is the projection of the point src onto X, Y and Z edges
        xp = Float64(i-1)*dh+xinit
        yp = Float64(j-1)*dh+yinit
        zp = Float64(k-1)*dh+zinit

        distPSx = abs(xp-xsrc) ## abs needed for later...
        distPSy = abs(yp-ysrc)
        distPSz = abs(zp-zsrc)
        dist2src = sqrt(distPSx^2+distPSy^2+distPSz^2)

        ## Calculate the traveltime to hit the x edge
        ## time at H along x
        thx = tt[i,j,k]*(sqrt(distPSy^2+distPSz^2))/dist2src
        ## Calculate the traveltime to hit the y edge
        ## time at H along y
        thy = tt[i,j,k]*(sqrt(distPSx^2+distPSz^2))/dist2src
        ## Calculate the traveltime to hit the z edge
        ## time at H along z
        thz = tt[i,j,k]*(sqrt(distPSx^2+distPSy^2))/dist2src
        
        if onsrc[i+1,j,k]==true              
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
            aforw = -(thx     -tt[i,j,k])/distPSx # dist along x
        elseif onsrc[i-1,j,k]==true
            aback = -(tt[i,j,k]  -thx)/distPSx # dist along x
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
        end
        if onsrc[i,j+1,k]==true
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
            bforw = -(thy     -tt[i,j,k])/distPSy # dist along y
        else onsrc[i,j-1,k]==true
            bback = -(tt[i,j,k]  -thy)/distPSy # dist along y
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
        end
        if onsrc[i,j,k+1]==true     
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
            cforw = -(thz      -tt[i,j,k])/distPSz # dist along z
        elseif onsrc[i,j,k-1]==true
            cback = -(tt[i,j,k]  -thz)/distPSz # dist along z
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
        end
        
    else # not on src
        
        ## along x
        aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
        aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
        ## along y
        bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
        bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
        ## along z
        cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
        cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
        
    end

    ##-----------------------------------
    aforwplus  = ( aforw+abs(aforw) )/2.0
    aforwminus = ( aforw-abs(aforw) )/2.0
    abackplus  = ( aback+abs(aback) )/2.0
    abackminus = ( aback-abs(aback) )/2.0

    bforwplus  = ( bforw+abs(bforw) )/2.0
    bforwminus = ( bforw-abs(bforw) )/2.0
    bbackplus  = ( bback+abs(bback) )/2.0
    bbackminus = ( bback-abs(bback) )/2.0

    cforwplus  = ( cforw+abs(cforw) )/2.0
    cforwminus = ( cforw-abs(cforw) )/2.0
    cbackplus  = ( cback+abs(cback) )/2.0
    cbackminus = ( cback-abs(cback) )/2.0
    
    ##-------------------------------------------
    ## make SURE lambda was INITIALIZED TO ZERO
    ## Leung & Qian, 2006
    numer =
        (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / dh +
        (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / dh +
        (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / dh
    
    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
        (cforwplus - cbackminus)/dh

    lambda[i,j,k] = numer/denom

    return
end


#############################################################################

function eikgrad_FMM_hiord_SINGLESRC(ttime::Array{Float64,3},vel::Array{Float64,3},
                         src::Array{Float64,1},rec::Array{Float64,2},
                         grd::Grid3D,pickobs::Array{Float64,1},
                         ttpicks::Array{Float64,1},stdobs::Float64)

    @assert size(src)==(3,)
    mindistsrc = 1e-5
    epsilon = 1e-5
    
    #@assert size(src)=
    @assert size(ttime)==(size(vel)) ## SAME SIZE!!
    nx,ny,nz = grd.nx,grd.ny,grd.nz  ##size(ttime)

    lambda = zeros((grd.nx,grd.ny,grd.nz)) 
 
    #copy ttime 'cause it might be changed around src
    tt=copy(ttime)
    dh = grd.hgrid
    # ntx = nx ##grd.ntx
    # nty = ny ##grd.nty
    # ntz = nz ##grd.nty
    
    ## Grid position
    xinit = grd.xinit
    yinit = grd.yinit
    zinit = grd.zinit
    
    ## source
    ## regular grid
    onsrc = sourceboxlocgrad!(tt,vel,src,grd, staggeredgrid=false )
    ## receivers
    onarec = recboxlocgrad!(lambda,ttpicks,grd,rec,pickobs,stdobs,staggeredgrid=false)

    xsrc,ysrc,zsrc = src[1],src[2],src[3]

    # ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
    # rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
    # ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
    # rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)
    # #@show dist,src,rx,ry
    # onsrc = zeros(Bool,nx,ny,nz)
    # onsrc[:,:,:] .= false    
    # halfg = 0.0
    
    # src_on_nodeedge = false
    # max_x = (grd.ntx-1)*grd.hgrid+grd.xinit
    # max_y = (grd.nty-1)*grd.hgrid+grd.yinit
    # max_z = (grd.ntz-1)*grd.hgrid+grd.zinit

    # ## loop to make sure we don't accidentally move the src to another node/edge
    # while (sqrt(rx^2+ry^2+rz^2)<=mindistsrc) || (abs(rx)<=mindistsrc) || (abs(ry)<=mindistsrc) || (abs(rz)<=mindistsrc)
    #     src_on_nodeedge = true
        
    #     ## shift the source 
    #     if xsrc < max_x-0.002*dh
    #         xsrc = xsrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         xsrc = xsrc-0.001*dh
    #     end
    #     if ysrc < max_y-0.002*dh
    #         ysrc = ysrc+0.001*dh
    #     else #(make sure it's not already at the bottom y)
    #         ysrc = ysrc-0.001*dh
    #     end
    #     if zsrc < max_z-0.002*dh
    #         zsrc = zsrc+0.001*dh
    #     else #(make sure it's not already at the bottom z)
    #         zsrc = zsrc-0.001*dh
    #     end

    #     # print("new time at src:  $(tt[ix,iy]) ")
    #     ## recompute parameters related to position of source
    #     ix,iy,iz = findclosestnode(xsrc,ysrc,zsrc,grd.xinit,grd.yinit,grd.zinit,grd.hgrid) 
    #     rx = xsrc-((ix-1)*grd.hgrid+grd.xinit)
    #     ry = ysrc-((iy-1)*grd.hgrid+grd.yinit)
    #     rz = zsrc-((iz-1)*grd.hgrid+grd.zinit)        

    # end

    # ## To avoid singularities, the src can only be inside a box,
    # ##  not on a grid node or edge... see above. So we need to have
    # ##  always 4 nodes where onsrc is true
    # if (rx>=halfg) & (ry>=halfg) & (rz>=halfg)
    #     onsrc[ix:ix+1,iy:iy+1,iz:iz+1] .= true
    # elseif (rx<halfg) & (ry>=halfg) & (rz>=halfg)
    #     onsrc[ix-1:ix,iy:iy+1,iz:iz+1] .= true
    # elseif (rx<halfg) & (ry<halfg) & (rz>=halfg)
    #     onsrc[ix-1:ix,iy-1:iy,iz:iz+1] .= true
    # elseif (rx>=halfg) & (ry<halfg) & (rz>=halfg)
    #     onsrc[ix:ix+1,iy-1:iy,iz:iz+1] .= true

    # elseif (rx>=halfg) & (ry>=halfg) & (rz<halfg)
    #     onsrc[ix:ix+1,iy:iy+1,iz-1:iz] .= true
    # elseif (rx<halfg) & (ry>=halfg) & (rz<halfg)
    #     onsrc[ix-1:ix,iy:iy+1,iz-1:iz] .= true
    # elseif (rx<halfg) & (ry<halfg) & (rz<halfg)
    #     onsrc[ix-1:ix,iy-1:iy,iz-1:iz] .= true
    # elseif (rx>=halfg) & (ry<halfg) & (rz<halfg)
    #     onsrc[ix:ix+1,iy-1:iy,iz-1:iz] .= true
    # end


    # ## if src on node or edge, recalculate traveltimes in box
    # if src_on_nodeedge==true
    #     ## RE-set a new ttime around source 
    #     #isrc,jsrc = ind2sub(size(onsrc),find(onsrc))
    #     ijksrc = findall(onsrc)
    #     # println(" set time around src $isrc  $jsrc $ksrc") 
    #     #for (k,j,i) in zip(ksrc,jsrc,isrc)
    #     for lcart in ijksrc
    #         # for j in jsrc
    #         #     for i in isrc
    #         i = lcart[1]
    #         j = lcart[2]
    #         k = lcart[3]
    #         xp = (i-1)*grd.hgrid+grd.xinit
    #         yp = (j-1)*grd.hgrid+grd.yinit
    #         zp = (k-1)*grd.hgrid+grd.zinit
    #         ii = Int(floor((xsrc-grd.xinit)/grd.hgrid) +1)
    #         jj = Int(floor((ysrc-grd.yinit)/grd.hgrid) +1)
    #         kk = Int(floor((zsrc-grd.zinit)/grd.hgrid) +1)            
    #         #### vel[isrc[1,1],jsrc[1,1]] STAGGERED GRID!!!
    #         ttime[i,j,k] = sqrt( (xsrc-xp)^2+(ysrc-yp)^2+(zsrc-zp)^2) / vel[ii,jj,kk]
    #     end
    # end# 
        
    # ##-------------------------------------------
    # ## init receivers
    # nrec=size(rec,1)
    # for r=1:nrec
    #     i,j,k = findclosestnode(rec[r,1],rec[r,2],rec[r,2],grd.xinit,grd.yinit,grd.zinit,grd.hgrid)
    #     if (i==1) || (i==ntx) || (j==1) || (j==nty) || (k==1) || (k==ntz)
    #         println(" Receiver on border of model (i==1)||(i==ntx)||(j==1)||(j==nty || (k==1) || (k==ntz))")
    #         println(" Not yet implemented...")
    #         return nothing
    #     end
    #     onarec[i,j,k] = true
    #     ##dtonarec[i,j] = (ttpicks[r]-pickobs[r])/stdobs^2
    #     lambda[i,j,k] = (ttpicks[r]-pickobs[r])/stdobs^2
    # end
  
    ######################################################
    #-------------------------------
    ## init FMM 
    neigh = [1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]
 
    #-------------------------------
    ## init FMM 
    status = Array{Int64}(undef,nx,ny,nz)
    status[:,:,:] .= 0  ## set all to far

    ## LIMIT FOR DERIVATIVES... WHAT TO DO?
    status[1, :, : ] .= 2 ## set to accepted on boundary
    status[nx,:, : ] .= 2 ## set to accepted on boundary
    status[:, 1, : ] .= 2 ## set to accepted on boundary
    status[:, ny,: ] .= 2 ## set to accepted on boundary
    status[:, :, 1 ] .= 2 ## set to accepted on boundary
    status[:, :, nz] .= 2 ## set to accepted on boundary
    
    status[onarec] .= 2 ## set to accepted on RECS

    ## get the i,j acccepted
    #irec,jrec = findn(status.==2) #ind2sub((nx,ny),find(status.==2))
    ijkrec = findall(status.==2)
    irec = [x[1] for x in ijkrec]
    jrec = [x[2] for x in ijkrec]
    krec = [x[3] for x in ijkrec]
    naccinit = length(irec)

    ## Init the max binary heap with void arrays but max size
    Nmax=nx*ny*nz
 
    bheap = build_maxheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## conversion cart to lin indices, old sub2ind
    linid_nxnynz = LinearIndices((nx,ny,nz))
    ## conversion lin to cart indices, old sub2ind
    cartid_nxnynz = CartesianIndices((nx,ny,nz))

    ## construct initial narrow band
    for l=1:naccinit 
        for ne=1:4 ## four potential neighbors
            
            i = irec[l] + neigh[ne,1]
            j = jrec[l] + neigh[ne,2]
            k = jrec[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nx-1) || (i<2) || (j>ny-1) || (j<2) || (k>nz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far

                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxnynz[i,j,k]
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1

            end
        end
    end
 
    #-------------------------------
    ## main FMM loop
    totnpts = nx*ny*nz
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
   
        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end
        
        # remove from heap and get value and handle
        han,value = pop_maxheap!(bheap)
        
        # get 3D indices from handle
        cijka = cartid_nxnynz[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted     
        status[ia,ja,ka] = 2 # 2=accepted

        # set lambda of the new accepted point
        calcLAMBDA_hiord!(tt,status,onsrc,onarec,dh,
                            xinit,yinit,zinit,xsrc,ysrc,zsrc,lambda,ia,ja,ka)
        
        ## try all neighbors of newly accepted point
        for ne=1:4 
            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            ## For adjoint SKIP BORDERS!!!
            if (i>nx-1) || (i<2) || (j>ny-1) || (j<2) || (k>nz-1) || (k<2)
                continue
            end
            
            if status[i,j,k]==0 ## far, active

                # get handle
                #han = sub2ind((nx,ny),i,j)
                han = linid_nxnynz[i,j,k] #i+nx*(j-1)
                ## add tt of point to binary heap 
                insert_maxheap!(bheap,tt[i,j,k],han)
                # change status, add to narrow band
                status[i,j,k]=1
                
                ## NO NEED to update in narrow band because
                ##   we already know the time tt everywhere...
            end
        end
    end
                               
    ## *NOT* a staggered grid, so no interpolation...
    gradadj = -lambda./vel.^3
        
    return gradadj
end # eikadj_FMM_hiord_SINGLESRC  


##====================================================================##

function isooutrange(ib::Int64,jb::Int64,kb::Int64,nx::Int64,ny::Int64,nz::Int64)
    isoutb1st = false
    isoutb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<=1 || ib>=nx || jb<=1 || jb>=ny || kb<=1 || kb>=nz
        isoutb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<=2 || ib>=nx-1 || jb<=2 || jb>=ny-1 || kb<=2 || kb>=nz-1
        isoutb2nd = true
    end
    return isoutb1st,isoutb2nd
end

##====================================================================##

function calcLAMBDA_hiord!(tt::Array{Float64,3},status::Array{Int64},
                           onsrc::Array{Bool,3},onarec::Array{Bool,3},
                           dh::Float64,xinit::Float64,yinit::Float64,zinit::Float64,
                           xsrc::Float64,ysrc::Float64,zsrc::Float64,lambda::Array{Float64,3},
                           i::Int64,j::Int64,k::Int64)

    if onsrc[i,j,k]==true 
        
        ## If we are in the square containing the source,
        ## use real position of source for the derivatives.
        ## Calculate tt on x and y on the edges of the square
        ##  H is the projection of the point src onto X, Y and Z edges
        xp = Float64(i-1)*dh+xinit
        yp = Float64(j-1)*dh+yinit
        zp = Float64(k-1)*dh+zinit

        distPSx = abs(xp-xsrc) ## abs needed for later...
        distPSy = abs(yp-ysrc)
        distPSz = abs(zp-zsrc)
        dist2src = sqrt(distPSx^2+distPSy^2+distPSz^2)

        ## Calculate the traveltime to hit the x edge
        ## time at H along x
        thx = tt[i,j,k]*(sqrt(distPSy^2+distPSz^2))/dist2src
        ## Calculate the traveltime to hit the y edge
        ## time at H along y
        thy = tt[i,j,k]*(sqrt(distPSx^2+distPSz^2))/dist2src
        ## Calculate the traveltime to hit the z edge
        ## time at H along z
        thz = tt[i,j,k]*(sqrt(distPSx^2+distPSy^2))/dist2src
        
        if onsrc[i+1,j,k]==true              
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
            aforw = -(thx     -tt[i,j,k])/distPSx # dist along x
        elseif onsrc[i-1,j,k]==true
            aback = -(tt[i,j,k]  -thx)/distPSx # dist along x
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
        end
        if onsrc[i,j+1,k]==true
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
            bforw = -(thy     -tt[i,j,k])/distPSy # dist along y
        else onsrc[i,j-1,k]==true
            bback = -(tt[i,j,k]  -thy)/distPSy # dist along y
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
        end
        if onsrc[i,j,k+1]==true     
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
            cforw = -(thz      -tt[i,j,k])/distPSz # dist along z
        elseif onsrc[i,j,k-1]==true
            cback = -(tt[i,j,k]  -thz)/distPSz # dist along z
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh
        end
        
    else # not on src

        nx,ny,nz = size(lambda)
        isout1st,isout2nd = isooutrange(i,j,k,nx,ny,nz)

        ##
        ## Central differences in Leung & Qian, 2006 scheme!
        ##  
        if !isout2nd
            ##
            ## Fourth order central diff O(h^4); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ## 12*h => 12*dh/2 = 6*dh
            aback = -( -tt[i+1,j,k]+8.0*tt[i,j,k]-8.0*tt[i-1,j,k]+tt[i-2,j,k] )/(6.0*dh)
            aforw = -( -tt[i+2,j,k]+8.0*tt[i+1,j,k]-8.0*tt[i,j,k]+tt[i-1,j,k] )/(6.0*dh)
            bback = -( -tt[i,j+1,k]+8.0*tt[i,j,k]-8.0*tt[i,j-1,k]+tt[i,j-2,k] )/(6.0*dh)
            bforw = -( -tt[i,j+2,k]+8.0*tt[i,j+1,k]-8.0*tt[i,j,k]+tt[i,j-1,k] )/(6.0*dh)
            cback = -( -tt[i,j,k+1]+8.0*tt[i,j,k]-8.0*tt[i,j,k-1]+tt[i,j,k-2] )/(6.0*dh)
            cforw = -( -tt[i,j,k+2]+8.0*tt[i,j,k+1]-8.0*tt[i,j,k]+tt[i,j,k-1] )/(6.0*dh)
            
        else
            ##
            ## Central diff (Leung & Qian, 2006); a,b computed in between nodes of velocity
            ##  dh = 2*dist because distance from edge of cells is dh/2
            ##  2*h => 2*dh/2 = dh
            aback = -(tt[i,j,k]  -tt[i-1,j,k])/dh
            aforw = -(tt[i+1,j,k]-tt[i,j,k])/dh
            bback = -(tt[i,j,k]  -tt[i,j-1,k])/dh
            bforw = -(tt[i,j+1,k]-tt[i,j,k])/dh
            cback = -(tt[i,j,k]  -tt[i,j,k-1])/dh
            cforw = -(tt[i,j,k+1]-tt[i,j,k])/dh

        end                     

    end                    

    ##-----------------------------------
    aforwplus  = ( aforw+abs(aforw) )/2.0
    aforwminus = ( aforw-abs(aforw) )/2.0
    abackplus  = ( aback+abs(aback) )/2.0
    abackminus = ( aback-abs(aback) )/2.0

    bforwplus  = ( bforw+abs(bforw) )/2.0
    bforwminus = ( bforw-abs(bforw) )/2.0
    bbackplus  = ( bback+abs(bback) )/2.0
    bbackminus = ( bback-abs(bback) )/2.0

    cforwplus  = ( cforw+abs(cforw) )/2.0
    cforwminus = ( cforw-abs(cforw) )/2.0
    cbackplus  = ( cback+abs(cback) )/2.0
    cbackminus = ( cback-abs(cback) )/2.0
    
    ##-------------------------------------------
    ## make SURE lambda was INITIALIZED TO ZERO
    ## Leung & Qian, 2006
    numer =
        (abackplus * lambda[i-1,j,k] - aforwminus * lambda[i+1,j,k]) / dh +
        (bbackplus * lambda[i,j-1,k] - bforwminus * lambda[i,j+1,k]) / dh +
        (cbackplus * lambda[i,j,k-1] - cforwminus * lambda[i,j,k+1]) / dh
    
    denom = (aforwplus - abackminus)/dh + (bforwplus - bbackminus)/dh +
        (cforwplus - cbackminus)/dh

    lambda[i,j,k] = numer/denom

    return #lambda
end



##################################################
#end # end module
##################################################
