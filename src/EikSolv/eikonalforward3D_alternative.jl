
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

####################################################################

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
function eiktraveltime3Dalt(vel::Array{Float64,3},grd::Grid3DCart,coordsrc::Array{Float64,2},
                            coordrec::Vector{Array{Float64,2}}; ttalgo::String,
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
                @async ttpicks[igrs] = remotecall_fetch(ttforwsomesrc3Dalt,wks[s],
                                                        vel,coordsrc[igrs,:],
                                                        coordrec[igrs],grd,ttalgo,
                                                        returntt=returntt )
            end
        elseif returntt
            # return both traveltime picks at receivers and at all grid points
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,:,igrs],ttpicks[igrs] = remotecall_fetch(ttforwsomesrc3Dalt,wks[s],
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
function ttforwsomesrc3Dalt(vel::Array{Float64,3},coordsrc::Array{Float64,2},
                            coordrec::Vector{Array{Float64,2}},grd::Grid3DCart,
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
        ## pre-allocate ttime and status arrays plus the binary heap
        fmmvars = FMMvars3D(grd.nx,grd.ny,grd.nz,refinearoundsrc=true,
                        allowfixsqarg=false)
        ##  No discrete adjoint calculations
        adjvars = nothing
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
            ttFMM_hiord!(fmmvars,vel,coordsrc[s,:],grd,adjvars)
            ttimeGRPSRC[:,:,:,s] .= fmmvars.ttime
            
        else
            println("\n WRONG ttalgo name $algo .... \n")
            return nothing

        end
        
        ## Interpolate at receivers positions
        for i=1:size(coordrec[s],1)
            ttpicksGRPSRC[s][i] = trilinear_interp(view(ttimeGRPSRC,:,:,:,s),grd,
                                                   coordrec[s][i,:])
        end
    end

     if returntt
        return ttimeGRPSRC,ttpicksGRPSRC
     end
    return ttpicksGRPSRC
end


###############################################################################

"""
$(TYPEDSIGNATURES)

 Fast sweeping method for a single source in 3D using using Podvin-Lecomte stencils on a staggered grid.
"""
function ttFS_podlec(vel::Array{Float64,3},src::Vector{Float64},grd::Grid3DCart) 

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
    onsrc = sourceboxloctt_alternative!(ttime,vel,src,grd, staggeredgrid=true )

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
function ttFMM_podlec(vel::Array{Float64,3},src::Vector{Float64},grd::Grid3DCart) 

    epsilon = 1e-5
    
    ## ttime
    ## Init to HUGE to make the FMM work
    HUGE::Float64 = 1.0e300
    ttime = HUGE*ones(Float64,grd.ntx,grd.nty,grd.ntz)
    ntx,nty,ntz = grd.ntx,grd.nty,grd.ntz
    ntxyz = [ntx,nty,ntz]
    
    ## source location, etc.      
    ## STAGGERED grid
    onsrc = sourceboxloctt_alternative!(ttime,vel,src,grd, staggeredgrid=true )
    
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
    neigh = SA[1  0  0;
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
                tmptt = calcttpt_podlec(ttime,rotste,onsrc,slowness,grd,HUGE,i,j,k)
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
        if bheap.Nh[]<1
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
                tmptt = calcttpt_podlec(ttime,rotste,onsrc,slowness,grd,HUGE,i,j,k)
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nxnynz[i,j,k]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band

                # update the traveltime for this point
                tmptt = calcttpt_podlec(ttime,rotste,onsrc,slowness,grd,HUGE,i,j,k)
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
function calcttpt_podlec(ttime::Array{Float64,3},rotste::RotoStencils,
                   onsrc::Array{Bool,3},slowness::Array{Float64,3},
                   grd::Grid3DCart,HUGE::Float64,i::Int64,j::Int64,k::Int64)


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
        println("calcttpt_podlec(): on a source node... return 0.0")
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

####################################################################333


"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt_alternative!(ttime::Array{Float64,3},vel::Array{Float64,3},
                                     srcpos::Vector{Float64},grd::Grid3DCart; staggeredgrid::Bool )
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
