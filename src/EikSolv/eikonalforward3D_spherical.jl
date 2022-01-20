

#######################################################
##        Eikonal forward 3D                         ## 
#######################################################

###########################################################################

"""
$(TYPEDSIGNATURES)

Calculate traveltime for 3D velocity models. 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.
The algorithm used is "ttFMM\\_hiord", second order fast marching method. 

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
function traveltime3Dsphere(vel::Array{Float64,3},grd::Grid3DSphere,coordsrc::Array{Float64,2},
                      coordrec::Array{Float64,2}; algo::String="ttFMM_hiord", returntt::Bool=false) 
    
    #println("Check the source/rec to be in bounds!!!")
    @assert size(coordsrc,2)==3
    @assert size(coordrec,2)==3
    @assert all(vel.>0.0)
    @assert all(grd.rinit.<=coordsrc[:,1].<=((grd.nr-1)*grd.Δr+grd.rinit))
    @assert all(grd.θinit.<=coordsrc[:,2].<=((grd.nθ-1)*grd.Δθ+grd.θinit))
    @assert all(grd.φinit.<=coordsrc[:,3].<=((grd.nφ-1)*grd.Δφ+grd.φinit))

    
    ##------------------
    ## parallel version
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
        # in this case velocity and time arrays have the same shape
        ttime = zeros(grd.nr,grd.nθ,grd.nφ,nsrc)
    end

    @sync begin

        if !returntt
            # return ONLY traveltime picks at receivers
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[:,igrs] = remotecall_fetch(ttforwsomesrc3D,wks[s],
                                                          vel,coordsrc[igrs,:],
                                                          coordrec,grd,
                                                          returntt=returntt )
            end
        elseif returntt
            # return both traveltime picks at receivers and at all grid points
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,:,igrs],ttpicks[:,igrs] = remotecall_fetch(ttforwsomesrc3D,wks[s],
                                                                          vel,coordsrc[igrs,:],
                                                                          coordrec,grd,
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
"""
function ttforwsomesrc3D(vel::Array{Float64,3},coordsrc::Array{Float64,2},
                         coordrec::Array{Float64,2},grd::Grid3DSphere ;
                         returntt::Bool=false )
    
    nsrc = size(coordsrc,1)
    nrec = size(coordrec,1)                
    ttpicks = zeros(nrec,nsrc)

    # in this case velocity and time arrays have the same shape
    ttime = zeros(grd.nr,grd.nθ,grd.nφ,nsrc)
    
    ## group of pre-selected sources
    for s=1:nsrc
        ## Compute traveltime and interpolation at receivers in one go for parallelization
        
        ##elseif algo=="ttFMM_hiord"        
        ttime[:,:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grd)
                    
        ## Interpolate at receivers positions
        for i=1:nrec                
            ttpicks[i,s] = trilinear_interp_sph( ttime[:,:,:,s],grd,coordrec[i,1],coordrec[i,2],coordrec[i,3])
        end
    end

     if returntt
        return ttime,ttpicks
     end
    return ttpicks
end

###############################################################################

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

###############################################################################

"""
$(TYPEDSIGNATURES)
"""
function ttFMM_hiord(vel::Array{Float64,3},src::Vector{Float64},grd::Grid3DSphere) 

    ## Sizes
    nr,nθ,nφ=grd.nr,grd.nθ,grd.nφ #size(vel)  ## NOT A STAGGERED GRID!!!

    epsilon = 1e-5
    
    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nr,nθ,nφ)
    ttime[:,:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nr,nθ,nφ)
    status[:,:,:] .= 0   ## set all to far
    
    ##########################################
    ##refinearoundsrc=true

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
    Nmax=nr*nθ*nφ
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## conversion cart to lin indices, old sub2ind
    linid_nrnθnφ = LinearIndices((nr,nθ,nφ))
    ## conversion lin to cart indices, old sub2ind
    cartid_nrnθnφ = CartesianIndices((nr,nθ,nφ))

    ## pre-allocate
    tmptt::Float64 = 0.0 

    ## construct initial narrow band
    @inbounds for l=1:naccinit ##        
        @inbounds for ne=1:6 ## six potential neighbors
            
            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>nr) || (i<1) || (j>nθ) || (j<1) || (k>nφ) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j,k)
                # get handle
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nrnθnφ[i,j,k]
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1

            end            
        end
    end
    
    #-------------------------------
    ## main FMM loop
    totnpts = nr*nθ*nφ 
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        # pop top of heap
        han,tmptt = pop_minheap!(bheap)
        cijka = cartid_nrnθnφ[han]
        ia,ja,ka = cijka[1],cijka[2],cijka[3]
        # set status to accepted
        status[ia,ja,ka] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja,ka] = tmptt

        ## try all neighbors of newly accepted point
        @inbounds for ne=1:6

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if ( (i>nr) || (i<1) || (j>nθ) || (j<1) || (k>nφ) || (k<1) )
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                #print("calc tt  ")
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j,k)
                han = linid_nrnθnφ[i,j,k]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j,k)
                # get handle
                #han = sub2ind((ntx,nty,ntz),i,j,k)
                han = linid_nrnθnφ[i,j,k]
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
"""
function isonbord_sph(ib::Int64,jb::Int64,kb::Int64,nr::Int64,nθ::Int64,nφ::Int64)
    isonb1 = false
    isonb2 = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>nr || jb<1 || jb>nθ || kb<1 || kb>nφ
        isonb1 =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>nr-1 || jb<2 || jb>nθ-1 || kb<2 || kb>nφ-1 
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
function calcttpt_2ndord(ttime::Array{Float64,3},vel::Array{Float64,3},grd::Grid3DSphere,
                         status::Array{Int64,3},i::Int64,j::Int64,k::Int64)

    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    dr = grd.Δr
    dθ = deg2rad(grd.Δθ) ## DEG to RAD !!!!
    dφ = deg2rad(grd.Δφ) ## DEG to RAD !!!!
    # dr2 = dr^2
    # dθ2 = dθ^2
    nr = grd.nr
    nθ = grd.nθ
    nφ = grd.nφ
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
       #
    # Spherical coordinates!! r, θ and φ
    #
    ##################################################
    
    alpha = 0.0
    beta  = 0.0
    gamma = - slowcurpt^2 ## !!!!
    # ish = 0  ## 0, +1, -1
    # jsh = 0  ## 0, +1, -1
    # ksh = 0  ## 0, +1, -1
    HUGE = 1.0e30

    ## 3 directions
    @inbounds for axis=1:3
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## two sides for each direction
        @inbounds for l=1:2

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
            isonb1,isonb2 = isonbord_sph(i+ish,j+jsh,k+ksh,nr,nθ,nφ)
                                    
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
            deltah = dr
        elseif axis==2
            deltah = grd.r[i]*dθ #dθ already in radians
        elseif axis==3
            deltah = grd.r[i]*sind(grd.θ[j])*dφ #dφ already in radians
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

        #@show use1stord,use2ndord,alpha
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
            @inbounds for axis=1:3
                
                use1stord = false
                chosenval1 = HUGE
                
                ## two sides for each direction
                @inbounds for l=1:2

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
                    isonb1,isonb2 = isonbord_sph(i+ish,j+jsh,k+ksh,nr,nθ,nφ)
                    
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
                    deltah = dr
                elseif axis==2
                    deltah = grd.r[i]*dθ #dθ already in radians
                elseif axis==3
                    deltah = grd.r[i]*sind(grd.θ[j])*dφ #dφ already in radians
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
    vel::Array{Float64,3},src::Vector{Float64},grdcoarse::Grid3DSphere,inittt::Float64)
      
    ## downscaling factor
    downscalefactor::Int = 5
    ## extent of refined grid in terms of coarse grid nodes
    noderadius::Int = 5

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    ixsrcglob,iysrcglob,izsrcglob = findclosestnode_sph(src[1],src[2],src[3],grdcoarse.rinit,grdcoarse.θinit,
                                                        grdcoarse.φinit,grdcoarse.Δr,grdcoarse.Δθ,grdcoarse.Δφ)
    
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
    outxmax = i2coarsevirtual>grdcoarse.nr
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>grdcoarse.nθ
    outzmin = k1coarsevirtual<1 
    outzmax = k2coarsevirtual>grdcoarse.nφ

    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=grdcoarse.nr : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=grdcoarse.nθ : j2coarse=j2coarsevirtual
    outzmin ? k1coarse=1            : k1coarse=k1coarsevirtual
    outzmax ? k2coarse=grdcoarse.nφ : k2coarse=k2coarsevirtual

    ##
    ## Refined grid parameters
    ##
    dr = grdcoarse.Δr/downscalefactor
    dθ = grdcoarse.Δθ/downscalefactor
    dφ = grdcoarse.Δφ/downscalefactor

    # fine grid size
    nr = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    nθ = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    nφ = (k2coarse-k1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    # set the origin of fine grid
    rinit = grdcoarse.r[i1coarse] 
    θinit = grdcoarse.θ[j1coarse]
    φinit = grdcoarse.φ[k1coarse]
    # grd
    grdfine = Grid3DSphere(Δr=dr,Δθ=dθ,Δφ=dφ,nr=nr,nθ=nθ,nφ=nφ,rinit=rinit,θinit=θinit,φinit=φinit)

    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nr,nθ,nφ)
    ttime[:,:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nr,nθ,nφ)
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
    velfinegrd = Array{Float64}(undef,nr,nθ,nφ)
    @inbounds for k=1:nφ
        @inbounds for j=1:nθ
            @inbounds for i=1:nr
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
    onsrc = sourceboxloctt_sph!(ttime,velfinegrd,srcfine,grdfine)

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
    Nmax=nr*nθ*nφ
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## conversion cart to lin indices, old sub2ind
    linid_nrnθnφ= LinearIndices((nr,nθ,nφ))
    ## conversion lin to cart indices, old ind2sub
    cartid_nrnθnφ = CartesianIndices((nr,nθ,nφ))
    
    ## construct initial narrow band
    @inbounds for l=1:naccinit ##
        @inbounds for ne=1:6 ## six potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            k = ks[l] + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if (i>nr) || (i<1) || (j>nθ) || (j<1) || (k>nφ) || (k<1)
                continue
            end

            if status[i,j,k]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j,k)

                # get handle
                # han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθnφ[i,j,k]
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
    totnpts = nr*nθ*nφ
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nr,nθ),han)
        cija = cartid_nrnθnφ[han]
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
        #  if (ia==nr) || (ia==1) || (ja==nθ) || (ja==1)
        if (ia==1 && !outxmin) || (ia==nr && !outxmax) || (ja==1 && !outymin) || (ja==nθ && !outymax) || (ka==1 && !outzmin) || (ka==nφ && !outzmax)
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
        @inbounds for ne=1:6 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            k = ka + neigh[ne,3]
            
            ## if the point is out of bounds skip this iteration
            if (i>nr) || (i<1) || (j>nθ) || (j<1) || (k>nφ) || (k<1)
                continue
            end

            if status[i,j,k]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j,k)
                han = linid_nrnθnφ[i,j,k]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j,k]=1                

            elseif status[i,j,k]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j,k)
                # get handle
                han = linid_nrnθnφ[i,j,k]
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



