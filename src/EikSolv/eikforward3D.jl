

########################################################

function interpolate_receiver(ttime::Array{Float64,3},grd::AbstractGridEik3D,xypt::AbstractVector{Float64} )
    intval = trilinear_interp(ttime,grd,xypt)
    return intval                          
end

########################################################

"""
$(TYPEDSIGNATURES)

Trinilear interpolation.
"""
function trilinear_interp(f::AbstractArray{Float64,3},grd::AbstractGridEik3D,
                          xyzpt::AbstractVector{Float64}; outputcoeff::Bool=false)
 
    xin,yin,zin = xyzpt[1],xyzpt[2],xyzpt[3]

    if typeof(grd)==Grid3DCart
        dx = grd.hgrid
        dy = grd.hgrid
        dz = grd.hgrid
        xinit,yinit,zinit = grd.cooinit

    elseif typeof(grd)==Grid3DSphere
        dx = grd.Δr
        dy = grd.Δθ
        dz = grd.Δφ
        xinit,yinit,zinit = grd.cooinit

    end
    
    xh = (xin-xinit)/dx
    yh = (yin-yinit)/dy
    zh = (zin-zinit)/dz
    x = xin-xinit  
    y = yin-yinit
    z = zin-zinit
    i = floor(Int64,xh+1) # indices starts from 1
    j = floor(Int64,yh+1) # indices starts from 1
    k = floor(Int64,zh+1) # indices starts from 1

    ## if at the edges of domain choose previous square...
    nx,ny,nz=size(f)

    ## Julia indices start from 1 while i,j,k from 0
 
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
    if k==nz
        k=k-1
    elseif k>nz
        k=k-2
    end

    x0=(i-1)*dx
    y0=(j-1)*dy
    z0=(k-1)*dz
    x1=i*dx
    y1=j*dy
    z1=k*dz

    f000 = f[i,j,  k] 
    f010 = f[i,j+1,k]
    f001 = f[i,j,  k+1]
    f011 = f[i,j+1,k+1]

    f100 = f[i+1,j,  k] 
    f110 = f[i+1,j+1,k]
    f101 = f[i+1,j,  k+1]
    f111 = f[i+1,j+1,k+1]

    xd = (x - x0)/(x1 - x0)
    yd = (y - y0)/(y1 - y0)
    zd = (z - z0)/(z1 - z0)
    

    if outputcoeff

        ##============================
        ## !!! ORDERING MATTERS !!!!
        ##   for gradient computations
        ## ============================

        coeff = @SVector [(1-xd)*(1-yd)*(1-zd) ,
                          xd*(1-yd)*(1-zd) ,
                          (1-xd)*yd*(1-zd) ,
                          (1-xd)*(1-yd)*zd ,
                          xd*yd*(1-zd) ,
                          (1-xd)*yd*zd ,
                          xd*(1-yd)*(1-zd) ,
                          xd*yd*zd ]

        fcorn = @SVector[f000;
                         f100;
                         f010;
                         f001;
                         f110;
                         f011;
                         f101;
                         f111]

        
        ijs = @SMatrix [i   j    k;
                        i+1  j    k;
                        i  j+1   k;
                        i   j    k+1;
                        i+1  j+1  k;                        
                        i  j+1   k+1;
                        i+1  j    k+1;
                        i+1  j+1  k+1 ]
        return coeff,fcorn,ijs

    else

        interpval = f000 * (1-xd)*(1-yd)*(1-zd) +
                    f100 * xd*(1-yd)*(1-zd) +
                    f010 * (1-xd)*yd*(1-zd) +
                    f001 * (1-xd)*(1-yd)*zd +
                    f110 * xd*yd*(1-zd) +
                    f011 * (1-xd)*yd*zd +
                    f101 * xd*(1-yd)*(1-zd) +
                    f111 * xd*yd*zd

        return interpval
    end

    return interpval
end

###########################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order. 
   Three-dimensional Cartesian or spherical grid.
"""
function calcttpt_2ndord!(fmmvars::FMMVars3D,vel::Array{Float64,3},
                          grd::AbstractGridEik3D,ijk::MVector{3,Int64},
                          codeD::MVector{3,<:Integer}) 
    
    #######################################################
    ##  Local solver Sethian et al., Rawlison et al.  ???##
    #######################################################

    # The solution from the quadratic eq. to pick is the larger, see 
    #  Sethian, 1996, A fast marching level set method for monotonically
    #  advancing fronts, PNAS
    i = ijk[1]
    j = ijk[2]
    k = ijk[3]

    # sizes, etc.
    n1,n2,n3 = grd.grsize
    if typeof(grd)==Grid3DCart
        Δh = MVector(grd.hgrid,grd.hgrid,grd.hgrid)
    elseif typeof(grd)==Grid3DSphere
        Δh = MVector(grd.Δr, grd.r[i]*deg2rad(grd.Δθ),grd.r[i]*deg2rad(grd.Δφ) )
    end
    # slowness
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
    HUGE = typemax(eltype(vel)) #1.0e30
    codeD .= 0 # integers

    ## 3 axis, x, y and z
    for axis=1:3
        
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
                ksh = 0
            elseif lax==2 # axis==1
                ish = -1  #-1
                jsh = 0  #0
                ksh = 0
            elseif lax==3 # axis==2
                ish = 0 #0
                jsh = 1 #1
                ksh = 0
            elseif lax==4 # axis==2
                ish = 0 # 0
                jsh = -1 #-1
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
            isonb1st,isonb2nd = isonbord(i+ish,j+jsh,k+ksh,n1,n2,n3)

            ##==== 1st order ================
            if !isonb1st && fmmvars.status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                ## first test value
                testval1 = fmmvars.ttime[i+ish,j+jsh,k+ksh]

                ## pick the lowest value of the two
                if testval1<chosenval1 ## < only!!!
                    chosenval1 = testval1
                    use1stord = true

                    # save derivative choices
                    if axis==1
                        codeD[axis]= -ish
                    elseif axis==2
                        codeD[axis]= -jsh
                    elseif axis==3
                        codeD[axis]= -ksh
                    end
                    
                    ##==== 2nd order ================
                    ish2::Int64 = 2*ish
                    jsh2::Int64 = 2*jsh
                    ksh2::Int64 = 2*ksh
                    if !isonb2nd && fmmvars.status[i+ish2,j+jsh2,k+ksh2]==2 ## 2==accepted
                        # second test value
                        testval2 = fmmvars.ttime[i+ish2,j+jsh2,k+ksh2]
                        ## pick the lowest value of the two
                        ##    compare to chosenval 1, *not* 2!!
                        ## This because the direction has already been chosen
                        ##  at the line "testval1<chosenval1"
                        if testval2 <= chosenval1 
                            chosenval2 = testval2
                            use2ndord = true 
                            # save derivative choices
                            if axis==1
                                codeD[axis]= -2*ish
                            elseif axis==2
                                codeD[axis]= -2*jsh
                            elseif axis==3
                                codeD[axis]= -2*ksh
                            end
                        else
                            chosenval2=HUGE
                            # below in case first direction gets 2nd ord
                            #   but second direction, with a smaller testval1,
                            #   does *not* get a second order
                            use2ndord=false # this is needed!
                            # save derivative choices
                            if axis==1
                                codeD[axis]= -ish
                            elseif axis==2
                                codeD[axis]= -jsh
                            elseif axis==3
                                codeD[axis]= -ksh
                            end
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

    end ## for axis=1:3

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
        @warn "Discriminant is negative (sqarg<0.0), reverting to 1st order."
        begin    
            codeD .= 0 # integers
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
                        ish = 1 #1
                        jsh = 0 #0
                        ksh = 0
                    elseif lax==2 # axis==1
                        ish = -1  #-1
                        jsh = 0  #0
                        ksh = 0
                    elseif lax==3 # axis==2
                        ish = 0 #0
                        jsh = 1 #1
                        ksh = 0
                    elseif lax==4 # axis==2
                        ish = 0 # 0
                        jsh = -1 #-1
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
                    isonb1st,isonb2nd = isonbord(i+ish,j+jsh,k+ksh,n1,n2,n3)
                    
                    ## 1st order
                    if !isonb1st && fmmvars.status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                        testval1 = fmmvars.ttime[i+ish,j+jsh,k+ksh]
                        ## pick the lowest value of the two
                        if testval1 < chosenval1 ## < only
                            chosenval1 = testval1
                            use1stord = true

                            # save derivative choices
                            if axis==1
                                codeD[axis] = -ish
                            elseif axis==2
                                codeD[axis] = -jsh
                            elseif axis==3
                                codeD[axis] = -ksh
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
            
            ## recompute sqarg
            sqarg = beta^2-4.0*alpha*gamma

        end ## begin...
       
    end ## if sqarg<0.0

    ################################################################
    if sqarg<0.0
        ## SECOND time sqarg<0...
        @warn "Discriminant is still negative (sqarg<0.0), dropping to one dimension."
        begin 
            codeD[:] .= 0 # integers
            alpha = 0.0
            beta  = 0.0
            gamma = - slowcurpt^2 ## !!!!

            ## 3 directions
            chosenval1vec = MVector(HUGE,HUGE,HUGE)
            for axis=1:3
                
                use1stord = false
                chosenval1vec[axis] = HUGE
                
                ## two sides for each direction
                for l=1:2

                    ## map the 4 cases to an integer as in linear indexing...
                    lax = l + 2*(axis-1)
                    if lax==1 # axis==1
                        ish = 1 #1
                        jsh = 0 #0
                        ksh = 0
                    elseif lax==2 # axis==1
                        ish = -1  #-1
                        jsh = 0  #0
                        ksh = 0
                    elseif lax==3 # axis==2
                        ish = 0 #0
                        jsh = 1 #1
                        ksh = 0
                    elseif lax==4 # axis==2
                        ish = 0 # 0
                        jsh = -1 #-1
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
                    isonb1st,isonb2nd = isonbord(i+ish,j+jsh,k+ksh,n1,n2,n3)
                    
                    ## 1st order
                    if !isonb1st && fmmvars.status[i+ish,j+jsh,k+ksh]==2 ## 2==accepted
                        testval1 = fmmvars.ttime[i+ish,j+jsh,k+ksh]
                        ## pick the lowest value of the two
                        if testval1 < chosenval1vec[axis] ## < only
                            chosenval1vec[axis] = testval1
                            use1stord = true

                            # save derivative choices
                            if axis==1
                                codeD[axis] = -ish
                            elseif axis==2
                                codeD[axis] = -jsh
                            elseif axis==3
                                codeD[axis] = -ksh
                            end

                        end
                    end
                end # end two sides
            end # end three axes

            # use only one dimension to update the traveltime
            # pick the minimum traveltime from neighbors
            idcv = argmin(chosenval1vec)
            soughtt = (Δh[idcv]*slowcurpt) + chosenval1vec[idcv]
            # update the info about derivatives
            tmpcodeD = codeD[idcv]
            codeD .= 0
            # the only nonzero code is the one corresponding to the used dimension
            codeD[idcv] = tmpcodeD

            return soughtt

        end ## begin...
        ################################################################

        #         println("\n To get a non-negative discriminant, need to fulfil: ")
        #         println(" (tx-ty)^2 - 2*s^2/curalpha <= 0")
        #         println(" where tx,ty can be")
        #         println(" t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order")
        #         println(" t? = chosenval1  if 1st order ")
        
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
