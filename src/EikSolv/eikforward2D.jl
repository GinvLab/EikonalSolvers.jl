

########################################################

function interpolate_receiver(ttime::Array{Float64,2},grd::AbstractGridEik2D,xypt::AbstractVector{Float64} )
    intval = bilinear_interp(ttime,grd,xypt)
    return intval                          
end

########################################################

"""
$(TYPEDSIGNATURES)

Bilinear interpolation.
"""
function bilinear_interp(f::AbstractArray{Float64,2},grd::AbstractGridEik2D,
                         xyzpt::AbstractVector{Float64};
                         outputcoeff::Bool=false)

    xreq,yreq = xyzpt[1],xyzpt[2]

    if typeof(grd)==Grid2DCart
        dx = grd.hgrid
        dy = grd.hgrid
        xinit = grd.xinit
        yinit = grd.yinit

    elseif typeof(grd)==Grid2DSphere
        dx = grd.Δr
        dy = grd.Δθ
        xinit = grd.rinit
        yinit = grd.θinit

    end
    
    nx,ny = size(f)
    ## rearrange such that the coordinates of corners are (0,0), (0,1), (1,0), and (1,1)
    xh=(xreq-xinit)/dx
    yh=(yreq-yinit)/dy
    i=floor(Int64,xh+1) # indices starts from 1
    j=floor(Int64,yh+1) # indices starts from 1
  
    ## if at the edges of domain choose previous square...
    if i==nx
        i=i-1
    elseif i>nx
        i=i-2
    end
    if j==ny
        j=j-1
    elseif j>ny
        j=j-2
    end 

    xd=xh-(i-1) # indices starts from 1
    yd=yh-(j-1) # indices starts from 1

    if outputcoeff
        
        coeff = @SVector[(1.0-xd)*(1.0-yd),
                         (1.0-yd)*xd, 
                         (1.0-xd)*yd,
                         xd*yd]
        
        fcorn = @SVector[f[i,j],
                         f[i+1,j],
                         f[i,j+1],
                         f[i+1,j+1]]

        ijs = @SMatrix[i j;
                       i+1 j;
                       i j+1;
                       i+1 j+1]
        
        # intval = f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd +
        #     f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
        # @show intval,dot(coeff,fcorn) 
        # @assert intval ≈ dot(coeff,fcorn)

        return coeff,fcorn,ijs

    else
        intval = f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd +
            f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
        return intval

    end

    return 
end

#############################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(fmmvars::FMMVars2D,vel::Array{Float64,2},srcpos::AbstractVector,
                         grd::Grid2DCart )

    ## source location, etc.      
    mindistsrc = 10.0*eps()
    xsrc,ysrc=srcpos[1],srcpos[2]

    # get the position and velocity of corners around source
    _,velcorn,ijsrc = bilinear_interp(vel,grd,srcpos,outputcoeff=true)
    
    ## Set srcboxpar
    Ncorn = size(ijsrc,1)
    fmmvars.srcboxpar.ijksrc .= ijsrc
    fmmvars.srcboxpar.xyzsrc .= srcpos
    fmmvars.srcboxpar.velcorn .= velcorn

    ## set ttime around source ONLY FOUR points!!!
    for l=1:Ncorn
        i,j = ijsrc[l,:]

        ## set status = accepted == 2
        fmmvars.status[i,j] = 2

        ## corner position 
        xp = grd.x[i] 
        yp = grd.y[j]

        # set the distance from corner to origin
        distcorn = sqrt((xsrc-xp)^2+(ysrc-yp)^2)
        fmmvars.srcboxpar.distcorn[l] = distcorn

        # set the traveltime to corner
        fmmvars.ttime[i,j] = distcorn / vel[i,j] #velsrc

    end

    return #ijsrc
end 

##########################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order. 
   Two-dimensional Cartesian or spherical grid.
"""
function calcttpt_2ndord!(fmmvars::FMMVars2D,vel::Array{Float64,2},
                          grd::AbstractGridEik2D,ij::MVector{2,Int64},
                          codeD::MVector{2,<:Integer})
    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS
    i = ij[1]
    j = ij[2]

    if typeof(grd)==Grid2DCart
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
    HUGE = typemax(eltype(vel)) #1.0e30
    codeD[:] .= 0 # integers

    ## 2 axis, x and y
    for axis=1:2
        
        use1stord = false
        use2ndord = false
        chosenval1 = HUGE
        chosenval2 = HUGE
        
        ## 2 directions: forward or backward
        for l=1:2

            ## map the 4 cases to an integer as in linear indexing...
            lax = l + 2*(axis-1)
            if lax==1 # axis==1
                ish = 1 #1
                jsh = 0 #0
            elseif lax==2 # axis==1
                ish = -1  #-1
                jsh = 0  #0
            elseif lax==3 # axis==2
                ish = 0 #0
                jsh = 1 #1
            elseif lax==4 # axis==2
                ish = 0 # 0
                jsh = -1 #-1
            end

            ## check if on boundaries
            isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                                    
            ##==== 1st order ================
            if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
                ## first test value
                testval1 = fmmvars.ttime[i+ish,j+jsh]

                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only!!!
                    chosenval1 = testval1
                    use1stord = true

                    # save derivative choices
                    axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                    
                    ##==== 2nd order ================
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    if !isonb2nd && fmmvars.status[i+ish2,j+jsh2]==2 ## 2==accepted
                        # second test value
                        testval2 = fmmvars.ttime[i+ish2,j+jsh2]
                        ## pick the lowest value of the two
                        ##    compare to chosenval 1, *not* 2!!
                        ## This because the direction has already been chosen
                        ##  at the line "testval1<chosenval1"
                        if testval2<chosenval1 ## < only!!!
                            chosenval2 = testval2
                            use2ndord = true 
                            # save derivative choices
                            axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)
                        else
                            chosenval2=HUGE
                            # below in case first direction gets 2nd ord
                            #   but second direction, with a smaller testval1,
                            #   does *not* get a second order
                            use2ndord=false # this is needed!
                            # save derivative choices
                            axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                        end

                    end ##==== END 2nd order ================
                    
                end 
            end ##==== END 1st order ================


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
            gamma += curalpha * tmpa2^2 ## see init of gamma : - slowcurpt^2

        elseif use1stord # first order
            ## curalpha: make sure you multiply only times the
            ##   current alpha for beta and gamma...
            curalpha = 1.0/deltah^2 
            alpha += curalpha
            beta  += ( -2.0*curalpha * chosenval1 )
            gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
        end

    end ## for axis=1:2

    ## compute discriminant 
    sqarg = beta^2-4.0*alpha*gamma

    ## To get a non-negative discriminant, need to fulfil:
    ##    (tx-ty)^2 - 2*s^2/curalpha <= 0
    ##    where tx,ty can be
    ##     t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order
    ##     t? = chosenval1  if 1st order 
    ##

    ##=========================================================================================
    ## If discriminant is negative (probably because of sharp contrasts in
    ##  velocity) revert to 1st order for both x and y
    if sqarg<0.0

        begin    
            codeD[:] .= 0 # integers
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
                    if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
                        testval1 = fmmvars.ttime[i+ish,j+jsh]
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

            if fmmvars.allowfixsqarg==true
            
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
    ##=========================================================================================

    ### roots of the quadratic equation
    tmpsq = sqrt(sqarg)
    soughtt1 =  (-beta + tmpsq)/(2.0*alpha)
    soughtt2 =  (-beta - tmpsq)/(2.0*alpha)

    ## choose the largest solution
    soughtt = max(soughtt1,soughtt2)

    return soughtt
end

################################################################################

