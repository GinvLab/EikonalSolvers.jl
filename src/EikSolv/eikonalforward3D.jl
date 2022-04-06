

#######################################################
##        Eikonal forward 3D                         ## 
#######################################################

##using StaticArrays

##################################

struct RotoStencils
    Qce::Array{Int64,2}
    Pce::Array{Int64,2}
    Mce::Array{Int64,2}
    Nce::Array{Int64,2}
    Med::Array{Int64,2}
    Ned::Array{Int64,2}
    Pli::Array{Int64,2}
    slopt_ce::Array{Int64,2}
    slopt_ed::Array{Int64,2}
    sloadj_ed::Array{Int64,2}
    slopt_li::Array{Int64,3}
end

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
- `ttalgo` (optional): the algorithm to use to compute the traveltime, one amongst the following
    * "ttFS\\_podlec", fast sweeping method using Podvin-Lecomte stencils
    * "ttFMM\\_podlec", fast marching method using Podvin-Lecomte stencils
    * "ttFMM\\_hiord", second order fast marching method, the default algorithm 
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime3D(vel::Array{Float64,3},grd::Grid3D,coordsrc::Array{Float64,2},
                      coordrec::Vector{Array{Float64,2}}; ttalgo::String="ttFMM_hiord",
                      returntt::Bool=false) 
    
    #println("Check the source/rec to be in bounds!!!")
    @assert size(coordsrc,2)==3
    #@assert size(coordrec,2)==3
    @assert all(vel.>0.0)
    @assert all(grd.xinit.<=coordsrc[:,1].<=((grd.nx-1)*grd.hgrid+grd.xinit))
    @assert all(grd.yinit.<=coordsrc[:,2].<=((grd.ny-1)*grd.hgrid+grd.yinit))
    @assert all(grd.zinit.<=coordsrc[:,3].<=((grd.nz-1)*grd.hgrid+grd.zinit))
    
    @assert size(coordsrc,1)==length(coordrec)

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
        # return traveltime array and picks at receivers
        if ttalgo=="ttFMM_hiord"
            # in this case velocity and time arrays have the same shape
            ttime = zeros(grd.nx,grd.ny,grd.nz,nsrc)
        else
            # in this case the time array has shape(velocity)+1
            ttime = zeros(grd.ntx,grd.nty,grd.ntz,nsrc)
        end
    end

    @sync begin

        if !returntt
            # return ONLY traveltime picks at receivers
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc3D,wks[s],
                                                          vel,coordsrc[igrs,:],
                                                          coordrec[igrs],grd,ttalgo,
                                                          returntt=returntt )
            end
        elseif returntt
            # return both traveltime picks at receivers and at all grid points
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,:,igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc3D,wks[s],
                                                                          vel,coordsrc[igrs,:],
                                                                          coordrec[igrs],grd,ttalgo,
                                                                          returntt=returntt )
            end
        end

    end

    if !returntt
        return ttpicks
    elseif returntt
        return ttpicks,ttime
    end
end


#########################################################################

"""
$(TYPEDSIGNATURES)

  Compute the forward problem for a group of sources.
"""
function ttforwsomesrc3D(vel::Array{Float64,3},coordsrc::Array{Float64,2},
                      coordrec::Vector{Array{Float64,2}},grd::Grid3D,
                      ttalgo::String ; returntt::Bool=false )
    
    nsrc = size(coordsrc,1)
    # nrec = size(coordrec,1)                
    # ttpicks = zeros(nrec,nsrc)

    ttpicksGRPSRC = Vector{Vector{Float64}}(undef,nsrc)
    for i=1:nsrc
        curnrec = size(coordrec[i],1) 
        ttpicksGRPSRC[i] = zeros(curnrec)
    end

    if ttalgo=="ttFMM_hiord" 
        # in this case velocity and time arrays have the same shape
        ttimeGRPSRC = zeros(grd.nx,grd.ny,grd.nz,nsrc)
    else
        # in this case the time array has shape(velocity)+1
        ttimeGRPSRC = zeros(grd.ntx,grd.nty,grd.ntz,nsrc)
    end
    
    ## group of pre-selected sources
    for s=1:nsrc
        ## Compute traveltime and interpolation at receivers in one go for parallelization
        
        if ttalgo=="ttFS_podlec"        
            ttimeGRPSRC[:,:,:,s] = ttFS_podlec(vel,coordsrc[s,:],grd)
        

        elseif ttalgo=="ttFMM_podlec"            
            ttimeGRPSRC[:,: ,:,s] = ttFMM_podlec(vel,coordsrc[s,:],grd)
        

        elseif ttalgo=="ttFMM_hiord"        
            ttimeGRPSRC[:,:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grd)
                    
        else
            println("\n WRONG ttalgo name $algo .... \n")
            return nothing

        end
        
        ## Interpolate at receivers positions
        for i=1:size(coordrec[s],1)
            ttpicksGRPSRC[s][i] = trilinear_interp( ttimeGRPSRC[:,:,:,s], grd.hgrid,
                                                    grd.xinit,grd.yinit,grd.zinit,
                                                    coordrec[s][i,1],coordrec[s][i,2],coordrec[s][i,3])
        end
    end

     if returntt
        return ttime,ttpicksGRPSRC
     end
    return ttpicksGRPSRC
end

###############################################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(ttime::Array{Float64,3},vel::Array{Float64,3},srcpos::Vector{Float64},grd::Grid3D; staggeredgrid::Bool )
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

###############################################################################

"""
$(TYPEDSIGNATURES)

 Fast sweeping method for a single source in 3D using using Podvin-Lecomte stencils on a staggered grid.
"""
function ttFS_podlec(vel::Array{Float64,3},src::Vector{Float64},grd::Grid3D) 

    epsilon = 1e-5

    ## ttime
    HUGE::Float64 = 1.0e300
    inittt = HUGE
    ttime = inittt * ones(grd.ntx,grd.nty,grd.ntz)
    ntx,nty,ntz = grd.ntx,grd.nty,grd.ntz
    ntxyz = [ntx,nty,ntz]
    
    ##------------------------
    ## source location, etc.      
    ## STAGGERED grid
    onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=true )

    ######################################################################################

    ##================================
    slowness = 1.0./vel
    ##================================
        
    swedifference::Float64 = 0.0
    curpt = zeros(Int64,3)
    ttlocmin::Float64 = 0.0
    ttimeold = zeros(Float64,ntx,nty,ntz)
    i::Int64 = 0
    j::Int64 = 0
    k::Int64 = 0
    
    ##==========================================
    ## Get all the rotations, points, etc.
    rotste = initloccomp()
    
    ##===========================================================================================

    iswe = [1 ntx 1; 1 ntx  1; 1 ntx  1; 1  ntx 1; ntx 1 -1; ntx 1 -1; ntx 1 -1; ntx 1 -1 ] 
    jswe = [1 nty 1; 1 nty  1; nty 1 -1; nty 1 -1; 1  nty 1; 1 nty  1; nty 1 -1; nty 1 -1 ]
    kswe = [1 ntz 1; ntz 1 -1; 1  ntz 1; ntz 1 -1; 1  ntz 1; ntz 1 -1; 1 ntz  1; ntz 1 -1 ]

    ##===========================================================================================    

    tdiff_cor  = Vector{Float64}(undef,8)   # 8 diff from corners
    tdiff_edge = Vector{Float64}(undef,24)  # 8+24=32  3D diffracted
    ttra_cel   = Vector{Float64}(undef,24)  # 24x4=96  3D transmitted
    tdiff_fac  = Vector{Float64}(undef,12)  # 12  2D diffracted
    ttra_fac   = Vector{Float64}(undef,24)  # 24  2D transmitted
    ttra_lin   = Vector{Float64}(undef,6)   #  6  1D transmitted
    
    #slowli = MVector{4,Float64}() # = zeros(Float64,4)
    slowli = Vector{Float64}(undef,4)
        
    N = Array{Int64,1}(undef,3)
    M = Array{Int64,1}(undef,3)
    P = Array{Int64,1}(undef,3)
    Q = Array{Int64,1}(undef,3)
    slopt  = Array{Int64,1}(undef,3)
    sloadj = Array{Int64,1}(undef,3)
    sloptli= Array{Int64,1}(undef,3)

    tmin = Vector{Float64}(undef,6)
 
    NXYZ = size(ttime)
    SLOWNXYZ = size(slowness)
    
    SQRTOF2 = sqrt(2.0)
    SQRTOF3 = sqrt(3.0)

    ##===========================================================================================    
    
    #pa=0
    swedifference = 100.0*epsilon
    while swedifference>epsilon 
        #pa +=1
        ttimeold[:,:,:] = ttime
        #ttimeold = copy(ttime)
        
        for swe=1:8 ##in [1,2,3,4,5,6,7,8]

            for k=kswe[swe,1]:kswe[swe,3]:kswe[swe,2]            
                for j=jswe[swe,1]:jswe[swe,3]:jswe[swe,2]
                    for i=iswe[swe,1]:iswe[swe,3]:iswe[swe,2]
                        
                        ##===========================================
                        ## If on a source node, skip this iteration
                        onsrc[i,j,k]==true && continue
                       
                        ####################################################
                        ##   Local solver (Podvin, Lecomte, 1991)         ##
                        curpt[1],curpt[2],curpt[3] = i,j,k  ## faster than the above...

                        tdiff_cor[:]  .= HUGE
                        tdiff_edge[:] .= HUGE
                        ttra_cel[:]   .= HUGE
                        tdiff_fac[:]  .= HUGE
                        ttra_fac[:]   .= HUGE
                        ttra_lin[:]   .= HUGE

                        ##===========================================
                        ## Stencils 3D

                        ## loop over 8 cells 
                        for ice=1:8
                            ## rotate for each cell
                            for a=1:3
                                M[a] = rotste.Mce[a,ice] + curpt[a]
                            end
                            
                            ## if M is outside model boundaries, skip this iteration
                            if (M[1]<1) || (M[1]>ntx) || (M[2]<1) || (M[2]>nty) || (M[3]<1) || (M[3]>ntz)
                                continue
                            end
                            ##isinboundsmod(M,NXYZ)==false  && continue ## much slower...
                            
                            ## Get current cell velocity
                            for a=1:3
                                slopt[a] = rotste.slopt_ce[a,ice] + curpt[a]
                            end
                            
                            hs = grd.hgrid*slowness[slopt[1],slopt[2],slopt[3]] 
                            hs2 = hs^2

                            ##-------------------------------
                            ## 3D diffraction
                            ## from corners of the structure (point M)
                            tm =  ttime[M[1],M[2],M[3]] 
                            tdiff_cor[ice] = tm + hs*SQRTOF3
                            
                            ## loop over 3 faces
                            for iface=1:3

                                ## global index
                                l=(ice-1)*3+iface

                                ## rotate for each cell
                                ## change Q,N,P for each face
                                for a=1:3
                                    Q[a] = rotste.Qce[a,l] + curpt[a]
                                    N[a] = rotste.Nce[a,l] + curpt[a] 
                                    P[a] = rotste.Pce[a,l] + curpt[a]
                                end
                                
                                ## splat time for points
                                tq = ttime[Q[1],Q[2],Q[3]]
                                tn = ttime[N[1],N[2],N[3]]
                                tp = ttime[P[1],P[2],P[3]]
                                
                                ##-------------------------------
                                ## 3D transmission (conditional)
                                t_mnp,t_qnp,t_nmq,t_pmq = HUGE,HUGE,HUGE,HUGE
                                
                                tptm = (tp-tm)
                                tptm2 = tptm^2
                                tntm = (tn-tm)
                                tntm2 = tntm^2
                                tqtn = (tq-tn)
                                tqtn2 = tqtn^2
                                tqtp = (tq-tp)
                                tqtp2 = tqtp^2
                                
                                ## triangle MNP
                                if (tm<=tn) & (tm<=tp) 
                                    if ( 2.0*tptm2 + tntm2 <= hs2 )
                                        if ( 2.0*tntm2 + tptm2 <= hs2 )
                                            if ( tntm2 + tptm2 + tntm*tptm >= hs2/2.0 )
                                                t_mnp = tn+tp-tm+sqrt(hs2-tntm2-tptm2)
                                            end
                                        end
                                    end
                                end                                    

                                ## triangle QNP
                                if (tn<=tq) & (tp<=tq)
                                    if (tqtn2+tqtp2+tqtn*tqtp) <= hs2/2.0
                                        t_qnp = tq+sqrt(hs2-tqtn2-tqtp2)
                                    end
                                end                

                                ## triangle NMQ
                                if (0.0<=tntm<=tqtn)
                                    if (2.0*(tqtn2+tntm2)<=hs2)
                                        t_nmq = tq+sqrt(hs2-tqtn2-tntm2)
                                    end
                                end
                                
                                ## triangle PMQ
                                ## there are typos in Podvin and Lecomte 1991 
                                if (0.0<=tptm<=tqtp)
                                    if (2.0*(tqtp2+tptm2)<=hs2)
                                        t_pmq = tq+sqrt(hs2-tqtp2-tptm2) 
                                    end
                                end
                                
                                ##-------------
                                ## min of 4 triangles (4x24 transmitted...), l runs on 1:24
                                ttra_cel[l] = min(t_mnp,t_qnp,t_nmq,t_pmq)

                                
                                ##------------------------------------
                                ## 3D diffraction (conditional)
                                ## from edges, 1 for every face
                                if (0.0<=(tn-tm)<=(grd.hgrid*slowness[slopt[1],slopt[2],
                                                                      slopt[3]]/SQRTOF3) ) 
                                    tdiff_edge[l] = tn + SQRTOF2*sqrt(
                                        (grd.hgrid*slowness[slopt[1],slopt[2],
                                                            slopt[3]])^2-(tn-tm)^2 ) 
                                end                                
                            end
                        end                        
                        
                        ##===========================================
                        ## Stencils 2D

                        ## 12 faces
                        for ifa=1:12                            
                            ## rotate for each "face" (12 faces: 3 planes x=0,y=0,z=0)
                            for a=1:3
                                M[a] = rotste.Med[a,ifa] + curpt[a]
                            end
                            
                            ## if M is outside model boundaries, skip this iteration
                            if (M[1]<1) || (M[1]>ntx) || (M[2]<1) || (M[2]>nty) || (M[3]<1) || (M[3]>ntz)
                                continue
                            end
                            ##isinboundsmod(M,NXYZ)==false  && continue
                            
                            ## Get current cell velocity
                            for a=1:3
                                slopt[a]  = rotste.slopt_ed[a,ifa] + curpt[a] 
                                sloadj[a] = rotste.sloadj_ed[a,ifa] + curpt[a] 
                            end

                            ##---------------------------------------
                            ## BOUNDARY conditions
                            ##  if indices of slowness are outside boundaries of array...
                            ## minimum(slopt)  minimum for all x,y,z 
                            ##if isinboundsmod(slopt,SLOWNXYZ)==false
                            if (slopt[1]<1) || (slopt[1]>grd.nx) || (slopt[2]<1) ||
                                (slopt[2]>grd.ny) || (slopt[3]<1) || (slopt[3]>grd.nz)  
                                ## slopt is out of bounds
                                hs = grd.hgrid*slowness[sloadj[1],sloadj[2],sloadj[3]] 
                            #elseif isinboundsmod(sloadj,SLOWNXYZ)==false
                            elseif (sloadj[1]<1) || (sloadj[1]>grd.nx) || (sloadj[2]<1) ||
                                (sloadj[2]>grd.ny) || (sloadj[3]<1) || (sloadj[3]>grd.nz)
                                ## sloadj is out of bounds
                                hs = grd.hgrid*slowness[slopt[1],slopt[2],slopt[3]]  
                            else
                                ## get the minimum slowness of the two cells
                                hs = grd.hgrid*min(slowness[slopt[1],slopt[2],slopt[3]],
                                               slowness[sloadj[1],sloadj[2],sloadj[3]]) 
                            end
                            ##---------------------------------------
                            
                            tm = ttime[M[1],M[2],M[3]]
                            hs2 = hs^2

                            ##------------------------------
                            ## 2D diffracted
                            tdiff_fac[ifa] = tm + hs*SQRTOF2

                            ## there are 2 edges to consider
                            for ied=1:2
                                l=2*(ifa-1)+ied            

                                ## rotate for each of 12 faces x 2 edges
                                for a=1:3
                                    N[a] = rotste.Ned[a,l] + curpt[a]
                                end
                                tn = ttime[N[1],N[2],N[3]]
                            
                                ##------------------------------
                                ## 2D transmitted (conditional)
                                if ( 0.0<=(tn-tm)<=hs/SQRTOF2 )
                                    tntm2 = (tn-tm)^2
                                    ttra_fac[l] = tn+sqrt(hs2-tntm2)
                                end
                            end
                        end

                        ##===========================================
                        ## Stencils 1D
                        
                        # 6 1D transmissions...
                        for ise=1:6

                            for a=1:3
                                P[a] = rotste.Pli[a,ise] + curpt[a]
                            end
                            
                            ## if P is outside model boundaries, skip this iteration
                            if (P[1]<1) || (P[1]>ntx) || (P[2]<1) ||
                                (P[2]>nty) || (P[3]<1) || (P[3]>ntz)
                                continue
                            end
                            ##isinboundsmod(P,NXYZ)==false  && continue
                            
                            for iv=1:4            
                                for a=1:3
                                    sloptli[a] = rotste.slopt_li[a,ise,iv] + curpt[a]
                                end
                                ##-----------------------------------------------
                                ## BOUNDARY conditions
                                ##  if indices of slowness are outside boundaries of array...
                                ## minimum(slopt)  minimum for all x,y,z
                                if (sloptli[1]<1) || (sloptli[1]>grd.nx) || (sloptli[2]<1) ||
                                    (sloptli[2]>grd.ny) || (sloptli[3]<1) || (sloptli[3]>grd.nz)
                                 ##isinboundsmod(sloptli,SLOWNXYZ)==false
                                    ## sloptli[iv] is out of bounds
                                    slowli[iv] = HUGE
                                else
                                    slowli[iv] = slowness[sloptli[1],sloptli[2],sloptli[3]]
                                end    
                                ##-----------------------------------------------
                            end
                            
                            ##-----------------------------------------------
                            ## 1D stencil
                            ttra_lin[ise] = ttime[P[1],P[2],P[3]] + grd.hgrid*minimum(slowli)
                        end

                        ##===========================================
                        ## Overall minimum
                        tmin[1] = minimum(tdiff_cor)
                        tmin[2] = minimum(ttra_cel)
                        tmin[3] = minimum(tdiff_edge)
                        tmin[4] = minimum(tdiff_fac)
                        tmin[5] = minimum(ttra_fac)
                        tmin[6] = minimum(ttra_lin)
                        
                        ttlocmin = minimum(tmin)                        

                        #############################################
                        
                        ttime[i,j,k] = min(ttimeold[i,j,k],ttlocmin)                
                        #############################################
                    end
                end
            end
        end
        ##============================================
        swedifference = maximum(abs.(ttime.-ttimeold))
    end
    
    return ttime
end

#############################################################
#############################################################

"""
$(TYPEDSIGNATURES)

Initialize local computations: 3D, 2D and 1D stencils
   See Podvin and Lecompte 1991 for stencils.
"""
function initloccomp() 
    
    ## rotation matrices
    rot90X = [1 0  0; 0 0 -1; 0 1 0]
    rot90Y = [0 0 -1; 0 1 0; -1 0 0]
    rot90Z = [0 -1 0; 1 0 0;  0 0 1]
    rotrev90X = [1 0 0; 0 0 1; 0 -1 0]
    rotrev90Y = [0 0 -1; 0 1 0; 1 0 0]

    ## 3D stuff ========
    ## see related figure
    Mfirst_ce = [1, 1, 1]
    Qfirst_ce = [1 0 0; 0 1 0; 0 0 1]' # transpose! columns are the vectors
    Pfirst_ce = [1 1 0; 0 1 1; 1 0 1]'
    Nfirst_ce = [1 0 1; 1 1 0; 0 1 1]'
    # keep all integers
    # the order of rotations MATTTERS!
    eye3int = [1 0 0; 0 1 0; 0 0 1]
    rotcel = [eye3int, rot90Z,  rot90Z*rot90Z,
              rot90Z*rot90Z*rot90Z,
              rotrev90X,  rotrev90X*rot90Z,
              rot90Z*rot90Z*rotrev90X,
              rot90Z*rot90Z*rot90Z*rotrev90X ]  

    ## 2D stuff ========
    ## see related figure
    Mfirst_ed = [1,1,0]
    Nfirst_ed = [1 0 0; 0 1 0]' # transpose! columns are the vectors
    # keep all integers
    rotfac = [eye3int, rot90Z, rot90Z*rot90Z, rot90Z*rot90Z*rot90Z,
              rotrev90X,  rot90Z*rotrev90X, rot90Z*rot90Z*rotrev90X,
              rot90Z*rot90Z*rot90Z*rotrev90X,
              rot90X,  rot90Z*rot90X,
              rot90Z*rot90Z*rot90X,
              rot90Z*rot90Z*rot90Z*rot90X ]

    ## 1D stuff ========
    Pfirst_li = [1, 0, 0]
    rotseg = [eye3int, rot90Z, rot90Z*rot90Z,
              rot90Z*rot90Z*rot90Z,
              rotrev90Y, rot90Y]

    ##=================================
    ### Pre-apply rotations, etc.

    ## 3D ============
    Mce = zeros(Int64,3,12)
    Qce = zeros(Int64,3,24)
    Nce = zeros(Int64,3,24)
    Pce = zeros(Int64,3,24)
    slopt_ce = zeros(Int64,3,8)

    ## loop on cells
    for i=1:8
        Mce[:,i] = rotcel[i] * Mfirst_ce
        ## shift the grid/axis of rotation for slowness
        slopt_ce[:,i] = round.(Int64, rotcel[i]*[0.5,0.5,0.5]-[0.5,0.5,0.5])
        ## loop on faces
        for j=1:3
            l=3*(i-1)+j            
            Qce[:,l] = rotcel[i] * Qfirst_ce[:,j]
            Nce[:,l] = rotcel[i] * Nfirst_ce[:,j]
            Pce[:,l] = rotcel[i] * Pfirst_ce[:,j]
        end
    end

    ## 2D ============
    Med = zeros(Int64,3,12)
    Ned = zeros(Int64,3,24)
    slopt_ed = zeros(Int64,3,12)
    sloadj_ed = zeros(Int64,3,12)

    #slopt_ed = ???
    #sloadj_ed = ???
    # for i=1:12
    #        A = rotfac[i]* Mfirst_ed
    #        B = round.(Int64,rotfac[i]*[0.5,0.5,0.5] - [0.5,0.5,0.5])
    #        C = round.(Int64,rotfac[i]*[0.5,0.5,-0.5] - [0.5,0.5,0.5])
    #        println("$i: $A  $B $C")
    #   end

    ## loop on faces
    for i=1:12
        Med[:,i] = rotfac[i] * Mfirst_ed
        ## shift the grid/axis of rotation for slowness
        slopt_ed[:,i] = round.(Int64,rotfac[i]*[0.5,0.5,0.5]-[0.5,0.5,0.5])
        ## [0.5, 0.5, -0.5]  MINUS 0.5 for initial point for adjacent
        sloadj_ed[:,i] = round.(Int64,rotfac[i]*[0.5,0.5,-0.5]-[0.5,0.5,0.5])
        ##  2 edges
        for j=1:2
            l=2*(i-1)+j
            Ned[:,l] = rotfac[i]*Nfirst_ed[:,j]
        end
    end

    ## 1D ============
    Pli = zeros(Int64,3,6)
    slopt_li = zeros(Int64,3,6,4)
    ## loop on lines
    for i=1:6
        Pli[:,i] = rotseg[i]*Pfirst_li
        ## slowness indices for 1D (4 indices)
        slopt_li[:,i,1] = round.(Int64,rotcel[i]*[0.5,0.5,0.5]-[0.5,0.5,0.5])
        slopt_li[:,i,2] = round.(Int64,rotcel[i]*[0.5,-0.5,0.5]-[0.5,0.5,0.5])
        slopt_li[:,i,3] = round.(Int64,rotcel[i]*[0.5,-0.5,-0.5]-[0.5,0.5,0.5])
        slopt_li[:,i,4] = round.(Int64,rotcel[i]*[0.5,0.5,-0.5]-[0.5,0.5,0.5])
    end
     
    rotste = RotoStencils(Qce,Pce,Mce,Nce,Med,Ned,Pli,
                          slopt_ce,slopt_ed,sloadj_ed,slopt_li)
    
    return rotste
end

#################################################################
#################################################################

"""
$(TYPEDSIGNATURES)

 Fast marching method for a single source in 3D using using Podvin-Lecomte stencils on a staggered grid.
"""
function ttFMM_podlec(vel::Array{Float64,3},src::Vector{Float64},grd::Grid3D) 

    epsilon = 1e-5
    
    ## ttime
    ## Init to HUGE to make the FMM work
    HUGE::Float64 = 1.0e300
    ttime = HUGE*ones(Float64,grd.ntx,grd.nty,grd.ntz)
    ntx,nty,ntz = grd.ntx,grd.nty,grd.ntz
    ntxyz = [ntx,nty,ntz]
    
    ## source location, etc.      
    ## STAGGERED grid
    onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=true )
    
    ##================================
    slowness = 1.0./vel
    ##================================
    
    ttlocmin::Float64 = 0.0
    i::Int64 = 0
    j::Int64 = 0
    k::Int64 = 0
    
    ##==========================================
    ## Get all the rotations, points, etc.
    rotste = initloccomp()
    
    ##==========================================
    ## init FMM 
    neigh = [1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]

    status = Array{Int64}(undef,ntx,nty,ntz)
    status[:,:,:] .= 0   ## set all to far
    status[onsrc] .= 2  ## set to accepted on src
    naccinit = count(status.==2)

    ijkss = findall(status.==2) 
    is = [l[1] for l in ijkss]
    js = [l[2] for l in ijkss]
    ks = [l[3] for l in ijkss]

    ## Init the max binary heap with void arrays but max size
    Nmax=ntx*nty*ntz
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 

    ## conversion cart to lin indices, old sub2ind
    linid_nxnynz = LinearIndices((ntx,nty,ntz))
    ## conversion lin to cart indices, old sub2ind
    cartid_nxnynz = CartesianIndices((ntx,nty,ntz))

    ## construct initial narrow band
    for l=1:naccinit ##
        
        for ne=1:6 ## six potential neighbors
            
            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>ntx) || (i<1) || (j>nty) || (j<1) || (k>ntz) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt(ttime,rotste,onsrc,slowness,grd,HUGE,i,j,k)
                # get handle
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nxnynz[i,j,k]
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
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

        # pop top of heap
        han,tmptt = pop_minheap!(bheap)
        cijka = cartid_nxnynz[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        #ja = div(han,ntx) +1
        #ia = han - ntx*(ja-1)
        # set status to accepted
        status[ia,ja,ka] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja,ka] = tmptt

        ## try all neighbors of newly accepted point
        for ne=1:6

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>ntx) || (i<1) || (j>nty) || (j<1) || (k>ntz) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt(ttime,rotste,onsrc,slowness,grd,HUGE,i,j,k)
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nxnynz[i,j,k]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band

                # update the traveltime for this point
                tmptt = calcttpt(ttime,rotste,onsrc,slowness,grd,HUGE,i,j,k)
                # get handle
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nxnynz[i,j,k]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)
#end     
            end
        end
        ##-------------------------------
    end
    return ttime
end  # ttFMM_podlec



#################################################################

"""
$(TYPEDSIGNATURES)

 Compute the traveltime at requested node using Podvin-Lecomte stencils on a staggered grid.
"""
function calcttpt(ttime::Array{Float64,3},rotste::RotoStencils,
                   onsrc::Array{Bool,3},slowness::Array{Float64,3},
                   grd::Grid3D,HUGE::Float64,i::Int64,j::Int64,k::Int64)


    #HUGE::Float64 = 1.0e300
    inittt = HUGE

    ntx,nty,ntz = grd.ntx,grd.nty,grd.ntz
    
    tdiff_cor  = Vector{Float64}(undef,8)   # 8 diff from corners
    tdiff_edge = Vector{Float64}(undef,24)  # 8+24=32  3D diffracted
    ttra_cel   = Vector{Float64}(undef,24)  # 24x4=96  3D transmitted
    tdiff_fac  = Vector{Float64}(undef,12)  # 12  2D diffracted
    ttra_fac   = Vector{Float64}(undef,24)  # 24  2D transmitted
    ttra_lin   = Vector{Float64}(undef,6)   #  6  1D transmitted
    
    #slowli = MVector{4,Float64}() # = zeros(Float64,4)
    slowli = Vector{Float64}(undef,4)
    
    N = Array{Int64,1}(undef,3)
    M = Array{Int64,1}(undef,3)
    P = Array{Int64,1}(undef,3)
    Q = Array{Int64,1}(undef,3)
    slopt  = Array{Int64,1}(undef,3)
    sloadj = Array{Int64,1}(undef,3)
    sloptli= Array{Int64,1}(undef,3)
    
    tmin = Vector{Float64}(undef,6)
    
    NXYZ = size(ttime)
    SLOWNXYZ = size(slowness)
    
    SQRTOF2 = sqrt(2.0)
    SQRTOF3 = sqrt(3.0)

    ##===========================================
    ## If on a source node, skip this iteration
    if onsrc[i,j,k]==true
        println("calcttpt(): on a source node... return 0.0")
        return 0.0 ##???
    end

    ####################################################
    ##   Local solver (Podvin, Lecomte, 1991)         ##
    curpt = zeros(Int64,3)
    curpt[1],curpt[2],curpt[3] = i,j,k  ## faster than the above...

    tdiff_cor[:]  .= HUGE
    tdiff_edge[:] .= HUGE
    ttra_cel[:]   .= HUGE
    tdiff_fac[:]  .= HUGE
    ttra_fac[:]   .= HUGE
    ttra_lin[:]   .= HUGE

    ##===========================================
    ## Stencils 3D

    ## loop over 8 cells 
    for ice=1:8
        ## rotate for each cell
        for a=1:3
            M[a] = rotste.Mce[a,ice] + curpt[a]
        end
        
        ## if M is outside model boundaries, skip this iteration
        if (M[1]<1) || (M[1]>ntx) || (M[2]<1) || (M[2]>nty) || (M[3]<1) || (M[3]>ntz)
            continue
        end
        ##isinboundsmod(M,NXYZ)==false  && continue ## much slower...
        
        ## Get current cell velocity
        for a=1:3
            slopt[a] = rotste.slopt_ce[a,ice] + curpt[a]
        end
        
        hs = grd.hgrid*slowness[slopt[1],slopt[2],slopt[3]] 
        hs2 = hs^2

        ##-------------------------------
        ## 3D diffraction
        ## from corners of the structure (point M)
        tm =  ttime[M[1],M[2],M[3]] 
        tdiff_cor[ice] = tm + hs*SQRTOF3
        
        ## loop over 3 faces
        for iface=1:3

            ## global index
            l=(ice-1)*3+iface

            ## rotate for each cell
            ## change Q,N,P for each face
            for a=1:3
                Q[a] = rotste.Qce[a,l] + curpt[a]
                N[a] = rotste.Nce[a,l] + curpt[a] 
                P[a] = rotste.Pce[a,l] + curpt[a]
            end
            
            ## splat time for points
            tq = ttime[Q[1],Q[2],Q[3]]
            tn = ttime[N[1],N[2],N[3]]
            tp = ttime[P[1],P[2],P[3]]
            
            ##-------------------------------
            ## 3D transmission (conditional)
            t_mnp,t_qnp,t_nmq,t_pmq = HUGE,HUGE,HUGE,HUGE
            
            tptm = (tp-tm)
            tptm2 = tptm^2
            tntm = (tn-tm)
            tntm2 = tntm^2
            tqtn = (tq-tn)
            tqtn2 = tqtn^2
            tqtp = (tq-tp)
            tqtp2 = tqtp^2
            
            ## triangle MNP
            if (tm<=tn) & (tm<=tp) 
                if ( 2.0*tptm2 + tntm2 <= hs2 )
                    if ( 2.0*tntm2 + tptm2 <= hs2 )
                        if ( tntm2 + tptm2 + tntm*tptm >= hs2/2.0 )
                            t_mnp = tn+tp-tm+sqrt(hs2-tntm2-tptm2)
                        end
                    end
                end
            end                                    

            ## triangle QNP
            if (tn<=tq) & (tp<=tq)
                if (tqtn2+tqtp2+tqtn*tqtp) <= hs2/2.0
                    t_qnp = tq+sqrt(hs2-tqtn2-tqtp2)
                end
            end                

            ## triangle NMQ
            if (0.0<=tntm<=tqtn)
                if (2.0*(tqtn2+tntm2)<=hs2)
                    t_nmq = tq+sqrt(hs2-tqtn2-tntm2)
                end
            end
            
            ## triangle PMQ
            ## there are typos in Podvin and Lecomte 1991 
            if (0.0<=tptm<=tqtp)
                if (2.0*(tqtp2+tptm2)<=hs2)
                    t_pmq = tq+sqrt(hs2-tqtp2-tptm2) 
                end
            end
            
            ##-------------
            ## min of 4 triangles (4x24 transmitted...), l runs on 1:24
            ttra_cel[l] = min(t_mnp,t_qnp,t_nmq,t_pmq)

            
            ##------------------------------------
            ## 3D diffraction (conditional)
            ## from edges, 1 for every face
            if (0.0<=(tn-tm)<=(grd.hgrid*slowness[slopt[1],slopt[2],
                                                  slopt[3]]/SQRTOF3) ) 
                tdiff_edge[l] = tn + SQRTOF2*sqrt(
                    (grd.hgrid*slowness[slopt[1],slopt[2],
                                        slopt[3]])^2-(tn-tm)^2 ) 
            end                                
        end
    end                        
    
    ##===========================================
    ## Stencils 2D

    ## 12 faces
    for ifa=1:12                            
        ## rotate for each "face" (12 faces: 3 planes x=0,y=0,z=0)
        for a=1:3
            M[a] = rotste.Med[a,ifa] + curpt[a]
        end
        
        ## if M is outside model boundaries, skip this iteration
        if (M[1]<1) || (M[1]>ntx) || (M[2]<1) || (M[2]>nty) || (M[3]<1) || (M[3]>ntz)
            continue
        end
        ##isinboundsmod(M,NXYZ)==false  && continue
        
        ## Get current cell velocity
        for a=1:3
            slopt[a]  = rotste.slopt_ed[a,ifa] + curpt[a] 
            sloadj[a] = rotste.sloadj_ed[a,ifa] + curpt[a] 
        end

        ##---------------------------------------
        ## BOUNDARY conditions
        ##  if indices of slowness are outside boundaries of array...
        ## minimum(slopt)  minimum for all x,y,z 
        ##if isinboundsmod(slopt,SLOWNXYZ)==false
        if (slopt[1]<1) || (slopt[1]>grd.nx) || (slopt[2]<1) ||
            (slopt[2]>grd.ny) || (slopt[3]<1) || (slopt[3]>grd.nz)  
            ## slopt is out of bounds
            hs = grd.hgrid*slowness[sloadj[1],sloadj[2],sloadj[3]] 
            #elseif isinboundsmod(sloadj,SLOWNXYZ)==false
        elseif (sloadj[1]<1) || (sloadj[1]>grd.nx) || (sloadj[2]<1) ||
            (sloadj[2]>grd.ny) || (sloadj[3]<1) || (sloadj[3]>grd.nz)
            ## sloadj is out of bounds
            hs = grd.hgrid*slowness[slopt[1],slopt[2],slopt[3]]  
        else
            ## get the minimum slowness of the two cells
            hs = grd.hgrid*min(slowness[slopt[1],slopt[2],slopt[3]],
                               slowness[sloadj[1],sloadj[2],sloadj[3]]) 
        end
        ##---------------------------------------
        
        tm = ttime[M[1],M[2],M[3]]
        hs2 = hs^2

        ##------------------------------
        ## 2D diffracted
        tdiff_fac[ifa] = tm + hs*SQRTOF2

        ## there are 2 edges to consider
        for ied=1:2
            l=2*(ifa-1)+ied            

            ## rotate for each of 12 faces x 2 edges
            for a=1:3
                N[a] = rotste.Ned[a,l] + curpt[a]
            end
            tn = ttime[N[1],N[2],N[3]]
            
            ##------------------------------
            ## 2D transmitted (conditional)
            if ( 0.0<=(tn-tm)<=hs/SQRTOF2 )
                tntm2 = (tn-tm)^2
                ttra_fac[l] = tn+sqrt(hs2-tntm2)
            end
        end
    end

    ##===========================================
    ## Stencils 1D
    
    # 6 1D transmissions...
    for ise=1:6

        for a=1:3
            P[a] = rotste.Pli[a,ise] + curpt[a]
        end
        
        ## if P is outside model boundaries, skip this iteration
        if (P[1]<1) || (P[1]>ntx) || (P[2]<1) ||
            (P[2]>nty) || (P[3]<1) || (P[3]>ntz)
            continue
        end
        
        for iv=1:4            
            for a=1:3
                sloptli[a] = rotste.slopt_li[a,ise,iv] + curpt[a]
            end
            ##-----------------------------------------------
            ## BOUNDARY conditions
            ##  if indices of slowness are outside boundaries of array...
            ## minimum(slopt)  minimum for all x,y,z
            if (sloptli[1]<1) || (sloptli[1]>grd.nx) || (sloptli[2]<1) ||
                (sloptli[2]>grd.ny) || (sloptli[3]<1) || (sloptli[3]>grd.nz)
                ##isinboundsmod(sloptli,SLOWNXYZ)==false
                ## sloptli[iv] is out of bounds
                slowli[iv] = HUGE
            else
                slowli[iv] = slowness[sloptli[1],sloptli[2],sloptli[3]]
            end    
            ##-----------------------------------------------
        end        
        ##-----------------------------------------------
        ## 1D stencil
        ttra_lin[ise] = ttime[P[1],P[2],P[3]] + grd.hgrid*minimum(slowli)
    end

    ##===========================================
    ## Overall minimum
    tmin[1] = minimum(tdiff_cor)
    tmin[2] = minimum(ttra_cel)
    tmin[3] = minimum(tdiff_edge)
    tmin[4] = minimum(tdiff_fac)
    tmin[5] = minimum(ttra_fac)
    tmin[6] = minimum(ttra_lin)
    
    ttlocmin = minimum(tmin)                        

    ttpt = ttlocmin
    return ttpt
end

#############################################################################
##=========================================================================##
"""
$(TYPEDSIGNATURES)

 Higher order (2nd) fast marching method in 3D using traditional stencils on regular grid. 
"""
function ttFMM_hiord(vel::Array{Float64,3},src::Vector{Float64},grd::Grid3D) 

    ## Sizes
    nx,ny,nz=grd.nx,grd.ny,grd.nz #size(vel)  ## NOT A STAGGERED GRID!!!

    epsilon = 1e-5
    
    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nx,ny,nz)
    ttime[:,:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nx,ny,nz)
    status[:,:,:] .= 0   ## set all to far
    
    ##########################################
    ## refinearoundsrc=true

    if extrapars.refinearoundsrc
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
        println("\nttFMM_hiord(): NO refinement around the source! \n")

        ## source location, etc.      
        ## REGULAR grid
        onsrc = sourceboxloctt!(ttime,vel,src,grd, staggeredgrid=false )
        
        ##
        ## Status of nodes
        status[onsrc] .= 2 ## set to accepted on src
        
    end # if refinearoundsrc

    ######################################################################################
    
    ttlocmin::Float64 = 0.0
    i::Int64 = 0
    j::Int64 = 0
    k::Int64 = 0
    
    ##===========================================================================================    
    #-------------------------------
    ## init FMM 
    neigh = [1  0  0;
             0  1  0;
            -1  0  0;
             0 -1  0;
             0  0  1;
             0  0 -1]

    ## get all i,j,k accepted
    ijkss = findall(status.==2) 
    is = [l[1] for l in ijkss]
    js = [l[2] for l in ijkss]
    ks = [l[3] for l in ijkss]
    naccinit = length(ijkss)

    ## Init the max binary heap with void arrays but max size
    Nmax=nx*ny*nz
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## conversion cart to lin indices, old sub2ind
    linid_nxnynz = LinearIndices((nx,ny,nz))
    ## conversion lin to cart indices, old sub2ind
    cartid_nxnynz = CartesianIndices((nx,ny,nz))

    ## pre-allocate
    tmptt::Float64 = 0.0 

    ## construct initial narrow band
    for l=1:naccinit ##        
        for ne=1:6 ## six potential neighbors
            
            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>nx) || (i<1) || (j>ny) || (j<1) || (k>nz) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j,k)
                # get handle
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nxnynz[i,j,k]
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
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

        # pop top of heap
        han,tmptt = pop_minheap!(bheap)
        cijka = cartid_nxnynz[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted
        status[ia,ja,ka] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja,ka] = tmptt

        ## try all neighbors of newly accepted point
        for ne=1:6

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>nx) || (i<1) || (j>ny) || (j<1) || (k>nz) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                #print("calc tt  ")
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j,k)
                han = linid_nxnynz[i,j,k]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j,k)
                # get handle
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nxnynz[i,j,k]
                # update the traveltime for this point in the heap
                #print("update heap  ")
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------
    end
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
function calcttpt_2ndord(ttime::Array{Float64,3},vel::Array{Float64,3},grd::Grid3D,
                         status::Array{Int64,3},i::Int64,j::Int64,k::Int64)

    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    dx = grd.hgrid
    dy = grd.hgrid
    dz = grd.hgrid
    # dx2 = dx^2
    # dy2 = dy^2
    nx = grd.nx
    ny = grd.ny
    nz = grd.nz
    ttcurpt = ttime[i,j,k]
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

    ## 2 directions
    for axis=1:3
        
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
            isonb1,isonb2 = isonbord(i+ish,j+jsh,k+ksh,nx,ny,nz)
                                    
            ## 1st order
            if !isonb1 && status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                testval1 = ttime[i+ish,j+jsh,k+ksh]
                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only
                    chosenval1 = testval1
                    use1stord = true

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
                        else
                            chosenval2=HUGE
                            use2ndord=false # this is needed!
                        end
                    end
                    
                end
            end
        end # end two sides

        if axis==1
            deltah = dx
        elseif axis==2
            deltah = dy
        elseif axis==3
            deltah = dz
        end
        
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
                    isonb1,isonb2 = isonbord(i+ish,j+jsh,k+ksh,nx,ny,nz)
                    
                    ## 1st order
                    if !isonb1 && status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                        testval1 = ttime[i+ish,j+jsh,k+ksh]
                        ## pick the lowest value of the two
                        if testval1<chosenval1 ## < only
                            chosenval1 = testval1
                            use1stord = true
                        end
                    end
                end # end two sides

                if axis==1
                    deltah = dx
                elseif axis==2
                    deltah = dy
                elseif axis==3
                    deltah = dz
                end

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
    vel::Array{Float64,3},src::Vector{Float64},grdcoarse::Grid3D,inittt::Float64)
      
    ## downscaling factor
    downscalefactor::Int = 5
    ## extent of refined grid in terms of coarse grid nodes
    noderadius::Int = 5

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    ixsrcglob,iysrcglob,izsrcglob = findclosestnode(src[1],src[2],src[3],grdcoarse.xinit,
                                                    grdcoarse.yinit,grdcoarse.zinit,grdcoarse.hgrid) 
    
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
    outxmax = i2coarsevirtual>grdcoarse.nx
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>grdcoarse.ny
    outzmin = k1coarsevirtual<1 
    outzmax = k2coarsevirtual>grdcoarse.nz

    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=grdcoarse.nx : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=grdcoarse.ny : j2coarse=j2coarsevirtual
    outzmin ? k1coarse=1            : k1coarse=k1coarsevirtual
    outzmax ? k2coarse=grdcoarse.nz : k2coarse=k2coarsevirtual

    ##
    ## Refined grid parameters
    ##
    dh = grdcoarse.hgrid/downscalefactor
    # fine grid size
    nx = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    ny = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    nz = (k2coarse-k1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    # set the origin of fine grid
    xinit = ((i1coarse-1)*grdcoarse.hgrid+grdcoarse.xinit)
    yinit = ((j1coarse-1)*grdcoarse.hgrid+grdcoarse.yinit)
    zinit = ((k1coarse-1)*grdcoarse.hgrid+grdcoarse.zinit)
    grdfine = Grid3D(hgrid=dh,xinit=xinit,yinit=yinit,zinit=zinit,nx=nx,ny=ny,nz=nz)

    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nx,ny,nz)
    ttime[:,:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nx,ny,nz)
    status[:,:,:] .= 0   ## set all to far
   
    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse,k1coarse:k2coarse)
    
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

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,nx,ny,nz)
    for k=1:nz
        for j=1:ny
            for i=1:nx
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
  
    ##
    ## Source location, etc. within fine grid
    ##  
    ## REGULAR grid, use"grdfine","finegrd", source position in the fine grid!!
    onsrc = sourceboxloctt!(ttime,velfinegrd,srcfine,grdfine, staggeredgrid=false )

    ######################################################

    neigh = [1  0  0;
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

    ## Init the min binary heap with void arrays but max size
    Nmax=nx*ny*nz
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## conversion cart to lin indices, old sub2ind
    linid_nxnynz= LinearIndices((nx,ny,nz))
    ## conversion lin to cart indices, old ind2sub
    cartid_nxnynz = CartesianIndices((nx,ny,nz))
    
    ## construct initial narrow band
    for l=1:naccinit ##
        for ne=1:6 ## six potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if (i>nx) || (i<1) || (j>ny) || (j<1) || (k>nz) || (k<1)
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j,k)

                # get handle
                # han = sub2ind((nx,ny),i,j)
                han = linid_nxnynz[i,j,k]
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
    totnpts = nx*ny*nz
    for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nx,ny),han)
        cija = cartid_nxnynz[han]
        ia,ja,ka = cija[1],cija[2],cija[3]
        # set status to accepted
        status[ia,ja,ka] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja,ka] = tmptt
        
        ##########################################################
        ##
        ## If the the accepted point is on the edge of the
        ##  fine grid, stop computing and jump to coarse grid
        ##
        ##########################################################
        #  if (ia==nx) || (ia==1) || (ja==ny) || (ja==1)
        if (ia==1 && !outxmin) || (ia==nx && !outxmax) || (ja==1 && !outymin) || (ja==ny && !outymax) || (ka==1 && !outzmin) || (ka==nz && !outzmax)
            ttimecoarse[i1coarse:i2coarse,j1coarse:j2coarse,k1coarse:k2coarse]  =  ttime[1:downscalefactor:end,1:downscalefactor:end,1:downscalefactor:end]
            statuscoarse[i1coarse:i2coarse,j1coarse:j2coarse,k1coarse:k2coarse] = status[1:downscalefactor:end,1:downscalefactor:end,1:downscalefactor:end]
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
            if (i>nx) || (i<1) || (j>ny) || (j<1) || (k>nz) || (k<1)
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j,k)
                han = linid_nxnynz[i,j,k]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j,k)
                # get handle
                han = linid_nxnynz[i,j,k]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------

    end
    error("Ouch...")
end

#################################################################3

#########################################################
#end
#########################################################



