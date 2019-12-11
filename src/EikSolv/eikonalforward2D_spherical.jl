

#######################################################
##        Eikonal forward 2D                         ## 
#######################################################

## @doc raw because of some backslashes in the string...
@doc raw"""
    traveltime2Dsphere(vel::Array{Float64,2},grd::Grid2DSphere,coordsrc::Array{Float64,2},
                 coordrec::Array{Float64,2} ; returntt::Bool=false ) 

Calculate traveltime for 2D velocity models in a structured spherical grid (polar coordinates). 
Returns the traveltime at receivers and optionally the array(s) of traveltime on the entire gridded model.
The computations are run in parallel depending on the number of workers (nworkers()) available.
The algorithm used is "ttFMM\_hiord".

# Arguments
- `vel`: the 2D velocity model
- `grd`: a struct specifying the geometry and size of the model
- `coordsrc`: the coordinates of the source(s) (x,y), a 2-column array
- `coordrec`: the coordinates of the receiver(s) (x,y), a 2-column array
- `returntt` (optional): whether to return the 3D array(s) of traveltimes for the entire model

# Returns
- `ttpicks`: array(nrec,nsrc) the traveltimes at receivers
- `ttime`: if `returntt==true` additionally return the array(s) of traveltime on the entire gridded model

"""
function traveltime2Dsphere( vel::Array{Float64,2},grd::Grid2DSphere,coordsrc::Array{Float64,2},
                       coordrec::Array{Float64,2} ; returntt::Bool=false ) 
        
    @assert size(coordsrc,2)==2
    @assert size(coordrec,2)==2
    @assert all(vel.>0.0)
    @assert all(grd.rinit.<=coordsrc[:,1].<=((grd.nr-1)*grd.Δr+grd.rinit))
    @assert all(grd.θinit.<=coordsrc[:,2].<=((grd.nθ-1)*grd.Δθ+grd.θinit))
    

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
        ttime = zeros(grd.nr,grd.nθ,nsrc)
    end

    @sync begin

        if !returntt
            # return ONLY traveltime picks at receivers
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttpicks[:,igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                          vel,coordsrc[igrs,:],
                                                          coordrec,grd,
                                                          returntt=returntt )
            end
        elseif returntt
            # return both traveltime picks at receivers and at all grid points
            for s=1:nchu
                igrs = grpsrc[s,1]:grpsrc[s,2]
                @async ttime[:,:,igrs],ttpicks[:,igrs] = remotecall_fetch(ttforwsomesrc2D,wks[s],
                                                                          vel,coordsrc[igrs,:],
                                                                          coordrec,grd,
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
                         coordrec::Array{Float64,2},grdsph::Grid2DSphere ; returntt::Bool=false )
    
    nsrc = size(coordsrc,1)
    nrec = size(coordrec,1)                
    ttpicksGRPSRC = zeros(nrec,nsrc) 

    # in this case velocity and time arrays have the same shape
    ttGRPSRC = zeros(grdsph.nr,grdsph.nθ,nsrc)

    ## group of pre-selected sources
    @inbounds for s=1:nsrc    

        ## Compute traveltime and interpolation at receivers in one go for parallelization
        ttGRPSRC[:,:,s] = ttFMM_hiord(vel,coordsrc[s,:],grdsph)

        ## interpolate at receivers positions
        @inbounds for i=1:nrec
            ## ASSUMING that the function varies linearly between adjacent points in r and θ, along rays and "rings" 
            ttpicksGRPSRC[i,s] = bilinear_interp_sph( ttGRPSRC[:,:,s],grdsph,coordrec[i,1],coordrec[i,2])
        end
    end
    
    if returntt
        return ttGRPSRC,ttpicksGRPSRC
    end
    return ttpicksGRPSRC 
end


#################################################################################

function sourceboxloctt_sph!(ttime::Array{Float64,2},vel::Array{Float64,2},srcpos::Vector{Float64},
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
        @inbounds for lcart in ijsrc
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

#################################################################################
#################################################################################

function ttFMM_hiord(vel::Array{Float64,2},src::Vector{Float64},
                     grd::Grid2DSphere) 
    
    ## Sizes
    nr,nθ=grd.nr,grd.nθ #size(vel)  ## NOT A STAGGERED GRID!!!
    epsilon = 1e-6
    
    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nr,nθ)
    ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nr,nθ)
    status[:,:] .= 0   ## set all to far
    
    ##########################################
    # switch off refinement for debugging...
    refinearoundsrc=true #true
    
    if refinearoundsrc
        ##---------------------------------
        ## 
        ## Refinement around the source      
        ##
        ttaroundsrc_sph!(status,ttime,vel,src,grd,inittt)

    else
        ##-----------------------------------------
        ## 
        ## NO refinement around the source      
        ##
        println("\nttFMM_hiord(): NO refinement around the source! \n")

        ## source location, etc.      
        ## REGULAR grid
        onsrc = sourceboxloctt_sph!(ttime,vel,src,grd )

        ##
        ## Status of nodes
        status[onsrc] .= 2 ## set to accepted on src
        
    end # if refinearoundsrc
    ###################################################### 

    # begin
    #     # for debugging...
    #     println("\n#### ttFMM_hiord(): DELETE using PyPlot at the TOP of eikonalforward2D.jl\n")
    #     figure()
    #     subplot(121)
    #     title("ttime")
    #     imshow(ttime,vmax=30)
    #     colorbar()
    #     subplot(122)
    #     title("status")
    #     imshow(status)
    #     colorbar()
    # end
    
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
    Nmax=nr*nθ
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## conversion cart to lin indices, old sub2ind
    linid_nrnθ = LinearIndices((nr,nθ))
    ## conversion lin to cart indices, old ind2sub
    cartid_nrnθ = CartesianIndices((nr,nθ))
    
    ## construct initial narrow band
    @inbounds for l=1:naccinit ##
        
        @inbounds for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]

            ## if the point is out of bounds skip this iteration
            if (i>nr) || (i<1) || (j>nθ) || (j<1)
                continue
            end

            if status[i,j]==0 ## far

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j)
                # get handle
                # han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθ[i,j]
                # insert into heap
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1
            end
        end
    end

    #-------------------------------
    ## main FMM loop
    totnpts = nr*nθ
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!

        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nr,nθ),han)
        cija = cartid_nrnθ[han]
        ia,ja = cija[1],cija[2]
        #ja = div(han,nr) +1
        #ia = han - nr*(ja-1)
        # set status to accepted
        status[ia,ja] = 2 # 2=accepted
        # set traveltime of the new accepted point
        ttime[ia,ja] = tmptt

        ## try all neighbors of newly accepted point
        @inbounds for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nr) || (i<1) || (j>nθ) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j)
                han = linid_nrnθ[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,vel,grd,status,i,j)
                # get handle
                han = linid_nrnθ[i,j]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------
    end
    
    return ttime
end

##====================================================================##

function isonbord_sph(ib::Int64,jb::Int64,nr::Int64,nθ::Int64)
    isonb1st = false
    isonb2nd = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>nr || jb<1 || jb>nθ
        isonb1st =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>nr-1 || jb<2 || jb>nθ-1
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
                         sphgrd::Grid2DSphere, status::Array{Int64,2},i::Int64,j::Int64)
    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS

    dr = sphgrd.Δr
    dθ = deg2rad(sphgrd.Δθ)  ## DEG to RAD !!!!

    # dx2 = dx^2
    # dy2 = dy^2
    nr = sphgrd.nr
    nθ = sphgrd.nθ
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
    #
    # Spherical coordinates!! r and θ
    # 
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
            isonb1st,isonb2nd = isonbord_sph(i+ish,j+jsh,nr,nθ)
            
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

        if axis==1
            deltah = dr
        elseif axis==2
            deltah = sphgrd.r[i]*dθ
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
                    isonb1st,isonb2nd = isonbord_sph(i+ish,j+jsh,nr,nθ)
                    
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

                if axis==1
                    deltah = dr
                elseif axis==2
                    deltah = sphgrd.r[i]*dθ
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
function ttaroundsrc_sph!(statuscoarse::Array{Int64,2},ttimecoarse::Array{Float64,2},
    vel::Array{Float64,2},src::Vector{Float64},grdcoarse::Grid2DSphere,inittt::Float64)
      
    ##
    ## 2x10 nodes -> 2x50 nodes
    ##
    downscalefactor::Int = 5
    noderadius::Int = 5
    ##println("Set noderadius and downscale factor back to 5!!!")

    ## find indices of closest node to source in the "big" array
    ## ix, iy will become the center of the refined grid
    ixsrcglob,iysrcglob = findclosestnode_sph(src[1],src[2],grdcoarse.rinit,grdcoarse.θinit,grdcoarse.Δr,grdcoarse.Δθ) 
    
    ##
    ## Define chunck of coarse grid
    ##
    i1coarsevirtual = ixsrcglob - noderadius
    i2coarsevirtual = ixsrcglob + noderadius
    j1coarsevirtual = iysrcglob - noderadius
    j2coarsevirtual = iysrcglob + noderadius
    # if hitting borders of the coarse grid truncate the finer
    outxmin = i1coarsevirtual<1
    outxmax = i2coarsevirtual>grdcoarse.nr
    outymin = j1coarsevirtual<1 
    outymax = j2coarsevirtual>grdcoarse.nθ
    outxmin ? i1coarse=1            : i1coarse=i1coarsevirtual
    outxmax ? i2coarse=grdcoarse.nr : i2coarse=i2coarsevirtual
    outymin ? j1coarse=1            : j1coarse=j1coarsevirtual
    outymax ? j2coarse=grdcoarse.nθ : j2coarse=j2coarsevirtual

    ##
    ## Refined grid parameters
    ##
    dr = grdcoarse.Δr/downscalefactor
    dθ = grdcoarse.Δθ/downscalefactor

    # fine grid size
    nr = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    nθ = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number

    # set the origin of fine grid
    ## not a Cartesian grid, so the radius MATTERS for the derivatives
    rinit = grdcoarse.r[i1coarse]
    θinit = grdcoarse.θ[j1coarse]
    # rinit = 0.0 
    # θinit = 0.0

    ###  grdfine = Grid2D(dh,xinit,yinit,nx,ny)
    grdfine = Grid2DSphere(Δr=dr,Δθ=dθ,nr=nr,nθ=nθ,rinit=rinit,θinit=θinit)

    ## 
    ## Time array
    ##
    inittt = 1e30
    ttime = Array{Float64}(undef,nr,nθ)
    ttime[:,:] .= inittt
    ##
    ## Status of nodes
    ##
    status = Array{Int64}(undef,nr,nθ)
    status[:,:] .= 0   ## set all to far
   
    ##
    ## Get the vel around the source on the coarse grid
    ##
    velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)
    
    ##
    ## Reset coordinates to match the fine grid
    ##
    # xorig = ((i1coarse-1)*grdcoarse.Δr+grdcoarse.rinit) 
    # yorig = ((j1coarse-1)*grdcoarse.Δθ+grdcoarse.θinit) 
    xsrc = src[1] #- xorig #- grdcoarse.rinit
    ysrc = src[2] #- yorig #- grdcoarse.θinit
    srcfine = Float64[xsrc,ysrc]

    ##
    ## Nearest neighbor interpolation for velocity on finer grid
    ## 
    velfinegrd = Array{Float64}(undef,nr,nθ)
    @inbounds for j=1:nθ
        @inbounds for i=1:nr
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
    ## REGULAR grid, use "grdfine","finegrd", source position in the fine grid!!
    onsrc = sourceboxloctt_sph!(ttime,velfinegrd,srcfine,grdfine)

    ######################################################
  
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
    Nmax=nr*nθ
    bheap = build_minheap!(Array{Float64}(undef,0),Nmax,Array{Int64}(undef,0))

    ## pre-allocate
    tmptt::Float64 = 0.0 
    
    ## conversion cart to lin indices, old sub2ind
    linid_nrnθ = LinearIndices((nr,nθ))
    ## conversion lin to cart indices, old ind2sub
    cartid_nrnθ = CartesianIndices((nr,nθ))
    
    ## construct initial narrow band
    @inbounds for l=1:naccinit ##        
         @inbounds for ne=1:4 ## four potential neighbors

            i = is[l] + neigh[ne,1]
            j = js[l] + neigh[ne,2]
            
            ## if the point is out of bounds skip this iteration
            if (i>nr) || (i<1) || (j>nθ) || (j<1)
                continue
            end

            if status[i,j]==0 ## far
                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                # get handle
                # han = sub2ind((nr,nθ),i,j)
                han = linid_nrnθ[i,j]
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
    totnpts = nr*nθ
    @inbounds for node=naccinit+1:totnpts ## <<<<===| CHECK !!!!
        
        ## if no top left exit the game...
        if bheap.Nh<1
            break
        end

        han,tmptt = pop_minheap!(bheap)
        #ia,ja = ind2sub((nr,nθ),han)
        cija = cartid_nrnθ[han]
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
        #  if (ia==nr) || (ia==1) || (ja==nθ) || (ja==1)
        if (ia==1 && !outxmin) || (ia==nr && !outxmax) || (ja==1 && !outymin) || (ja==nθ && !outymax) 
            ttimecoarse[ i1coarse:i2coarse,j1coarse:j2coarse]  =  ttime[1:downscalefactor:end,1:downscalefactor:end]
            statuscoarse[i1coarse:i2coarse,j1coarse:j2coarse]  = status[1:downscalefactor:end,1:downscalefactor:end]
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
        @inbounds for ne=1:4 

            i = ia + neigh[ne,1]
            j = ja + neigh[ne,2]
         
            ## if the point is out of bounds skip this iteration
            if (i>nr) || (i<1) || (j>nθ) || (j<1)
                continue
            end

            if status[i,j]==0 ## far, active

                ## add tt of point to binary heap and give handle
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                han = linid_nrnθ[i,j]
                insert_minheap!(bheap,tmptt,han)
                # change status, add to narrow band
                status[i,j]=1                

            elseif status[i,j]==1 ## narrow band                

                # update the traveltime for this point
                tmptt = calcttpt_2ndord(ttime,velfinegrd,grdfine,status,i,j)
                # get handle
                han = linid_nrnθ[i,j]
                # update the traveltime for this point in the heap
                update_node_minheap!(bheap,tmptt,han)

            end
        end
        ##-------------------------------

     end
    error("Ouch...")
end


#######################################################################################

#########################################################
#end                                                    #
#########################################################

