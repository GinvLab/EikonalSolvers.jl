
#####################################################

function test_fwdtt_3D_constvel()

    Ntests = 4
    
    maxerr = Vector{Float64}(undef,Ntests)

    # Create a grid and a velocity model
    grd = Grid3DCart(hgrid=250.0,xinit=0.0,yinit=0.0,zinit=0.0,
                     nx=30,ny=31,nz=32)
    nx,ny,nz = grd.grsize

    coordsrcs = [[grd.hgrid+grd.x[nx÷2]  grd.hgrid+grd.y[ny÷2]  grd.hgrid+grd.z[nz÷2]],
                 [grd.hgrid/2+grd.x[nx÷2]  grd.hgrid/2+grd.y[ny÷2]  grd.hgrid/2+grd.y[nz÷2]]]

    tolerr = [0.062135,
              0.157811,
              0.041310,
              0.023719]

    tolnorm = [7.75,
               1.09,
               5.82,
               3.05]

    r=0
    for coordsrc in coordsrcs

        for refinearoundsrc in [false,true]

            #printstyled("  Traveltime 3-D vs. analytical solution (constant velocity), refin.=$refinearoundsrc.\n",bold=true,color=:cyan)

            ## Traveltime forward
            extrapars = ExtraParams(refinearoundsrc=refinearoundsrc)
            
            constvelmod = 2500.0 .* ones(nx,ny,nz)
            coordrec = [[grd.x[2] grd.y[3] grd.y[3]]]
            
            _,ttime = eiktraveltime(constvelmod,grd,coordsrc,coordrec,
                                    returntt=true,extraparams=extrapars)
            
            # analytical solution for constant velocity
            xsrc,ysrc,zsrc=coordsrc[1,1],coordsrc[1,2],coordsrc[1,3]
            ttanalconstvel = zeros(nx,ny,nz)
            diffttanalconstvel = zeros(nx,ny,nz)
            for k=1:nz
                for j=1:ny
                    for i=1:nx
                        ttanalconstvel[i,j,k] = sqrt((grd.x[i]-xsrc)^2+(grd.y[j]-ysrc)^2+(grd.z[k]-zsrc)^2)/constvelmod[i,j,k]
                        diffttanalconstvel[i,j,k] = ttime[1][i,j,k] - ttanalconstvel[i,j,k]
                    end
                end
            end

            r+=1
            maxerr = maximum(abs.(diffttanalconstvel))
            normerr = norm(abs.(diffttanalconstvel))

            # @show maxerr, tolerr[r]
            # @show normerr, tolnorm[r]

            @test all( (maxerr <= tolerr[r], normerr<=tolnorm[r]) )
        end
    end
    
    return 
end

#############################################


function analyticalsollingrad3D(grd::Grid3DCart,xsrcpos::Float64,ysrcpos::Float64,zsrcpos::Float64)
    ##################################
    ## linear gradient of velocity
    nx,ny,nz = grd.grsize

    ## source must be *on* the top surface
    #@show ysrcpos
    @assert (zsrcpos == 0.0)

    # position of the grid nodes
    xgridpos = grd.x 
    ygridpos = grd.y
    zgridpos = grd.z
    # base velocity
    vel0 = 2.0
    # gradient of velociy
    gr = 0.05
    ## construct the 2D velocity model
    velmod = zeros(nx,ny,nz) 
    for j=1:ny, i=1:nx
        velmod[i,j,:] .= vel0 .+ gr .* (zgridpos .- zsrcpos)
    end

    # https://pubs.geoscienceworld.org/books/book/1011/lessons-in-seismic-computing
    # Lesson No. 41: Linear Distribution of Velocity—V. The Wave-Fronts 

    ## Analytic solution
    ansol2d  = zeros(nx,ny,nz)
    for k=1:nz, j=1:ny, i=1:nx
        x = sqrt( (xgridpos[i]-xsrcpos)^2+(ygridpos[j]-ysrcpos)^2 )
        h = zgridpos[k]-zsrcpos
        ansol2d[i,j,k] = 1/gr * acosh( 1 + (gr^2*(x^2+h^2))/(2*vel0*velmod[i,j,k]) )
    end
    return ansol2d,velmod
end

########################################################f

function test_fwdtt_3D_lingrad()
    

    tolerr = [3.83,
              3.19,
              3.26,
              2.86]
              
    tolnorm = [148.90,
               192.21,
               346.95,
               125.76]
                  
    #--------------------------------------
    # Create a grid and a velocity model
    grd = Grid3DCart(hgrid=15.0,xinit=0.0,yinit=0.0,zinit=0.0,
                     nx=40,ny=41,nz=32)
    nx,ny,nz = grd.grsize

    #########################
    # Analytical solution
    #
    ## source must be *on* the top surface for analytic solution
    coordsrcs = [[grd.x[end÷2]+grd.hgrid  grd.y[end÷2]+grd.hgrid  0.0],
                 [grd.x[end÷2]+0.253*grd.hgrid  grd.x[end÷2]+0.253*grd.hgrid  0.0]]

    coordrec = [[2.0 3.0 2.0]]

    r=1
    for coordsrc in coordsrcs

        ansol3d,velanaly = analyticalsollingrad3D(grd,coordsrc[1,1],coordsrc[1,2],coordsrc[1,3])

        for refinearoundsrc in [false,true]
            
            extrapars = ExtraParams(refinearoundsrc=refinearoundsrc)

            _,ttime = eiktraveltime(velanaly,grd,coordsrc,coordrec,
                                    returntt=true,extraparams=extrapars)

            maxerr = maximum(abs.(ttime[1].-ansol3d))
            normerr = norm(abs.(ttime[1].-ansol3d))

            # @show maxerr,tolerr[r]
            # @show normerr,tolnorm[r]
            @test all((maxerr<=tolerr[r], normerr<=tolnorm[r]))

            r+=1
        end
    end
    return
end

###################################

function test_gradvel_3D()

    # create the Grid2D struct
    grd = Grid3DCart(hgrid=20.0,
                     xinit=0.0,
                     yinit=0.0,
                     zinit=0.0,
                     nx=14,
                     ny=12,
                     nz=12)
    nx,ny,nz = grd.grsize

    # set receivers' positions
    nl = 5
    crec = zeros(nl*nl,3)
    xr = LinRange(grd.x[1],grd.x[end],nl)
    yr = LinRange(grd.y[1],grd.y[end],nl)
    zr = 1.34 * grd.z[1]
    l=1
    for j=1:nl
        for i=1:nl
            crec[l,:] .= (xr[i], yr[j], zr)
            l+=1
        end
    end
    coordrec = [crec]

      
    # velocity model 
    velmod = 2500.0 .* ones(grd.grsize...)
    ## increasing velocity with depth...
    for i=1:grd.grsize[3]
        velmod[:,:,i] = 25.4 * i .+ velmod[:,:,i]
    end


    # source coordinates
    coordsrc1 = [grd.hgrid*(grd.grsize[1]÷2)+0.312*grd.hgrid  grd.hgrid*(grd.grsize[2]÷2)-0.712*grd.hgrid  0.92*grd.z[end]]


    tolerr = [4.11e-6,
              3.90e-6]

    tolnorm = [3.01e-5,
               2.56e-5]

    r=0
    for refinesrc in [false,true]
    
        extraparams=ExtraParams(refinearoundsrc=refinesrc,
                                grdrefpars=GridRefinementPars(downscalefactor=3,noderadius=3),
                                radiussmoothgradsrc = 0 )

        ttpicks = eiktraveltime(velmod,grd,coordsrc1,coordrec,
                                extraparams=extraparams)

        # standard deviation of error on observed data
        stdobs = [0.0001.*ones(size(ttpicks[1])) for i=1:size(coordsrc1,1)]
        # generate a "noise" array to simulate real data
        #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
        # add the noise to the synthetic traveltime data
        dobs = ttpicks #.+ noise

        # # create a guess/"current" model
        vel0 = 2200.0 .* ones(nx,ny,nz)
        ## increasing velocity with depth...
        for i=1:nz
           vel0[:,:,i] = 32.5 * i .+ vel0[:,:,i]
        end

        coordsrc2 = copy(coordsrc1)

        ## calculate the gradient of the misfit function
        gradvel = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                              :gradvel,extraparams=extraparams)

        ##
        dh = 0.001

        N = prod(size(gradvel))
        gradvel_FD = similar(gradvel)
        for k=1:nz
            for j=1:ny                
                for i=1:nx
                    # print some info
                    l = i + nx*(j-1 + (k-1)*ny)
                    if l%10==0 
                        print("\r  Iteration $l of $N")
                    end
                    
                    velpdh = copy(vel0)
                    velpdh[i,j,k] += dh
                    velmdh = copy(vel0)
                    velmdh[i,j,k] -= dh
                    misf_pdh = eikttimemisfit(velpdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                              extraparams=extraparams)
                    misf_mdh = eikttimemisfit(velmdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                              extraparams=extraparams)
                    gradvel_FD[i,j,k] = (misf_pdh-misf_mdh)/(2*dh)
                end
            end
        end
        println()

        r+=1
        normerr = norm(abs.(gradvel.-gradvel_FD))
        maxerr = maximum(abs.(gradvel.-gradvel_FD))

        # @show maxerr,tolerr[r]
        # @show normerr,tolnorm[r]

        @test all((maxerr<=tolerr[r] && normerr<=tolnorm[r] ))
    end

    return
end


###################################

function test_gradsrc_3D()

    # create the Grid2D struct
    grd = Grid3DCart(hgrid=20.0,
                     xinit=0.0,
                     yinit=0.0,
                     zinit=0.0,
                     nx=18,
                     ny=17,
                     nz=16)
    nx,ny,nz = grd.grsize

    # set receivers' positions
    nl = 5
    crec = zeros(nl*nl,3)
    xr = LinRange(grd.x[1],grd.x[end],nl)
    yr = LinRange(grd.y[1],grd.y[end],nl)
    zr = 1.34 * grd.z[1]
    l=1
    for j=1:nl
        for i=1:nl
            crec[l,:] .= (xr[i], yr[j], zr)
            l+=1
        end
    end
    coordrec = [crec]

      
    # velocity model 
    velmod = 2500.0 .* ones(grd.grsize...)
    ## increasing velocity with depth...
    for i=1:grd.grsize[3]
        velmod[:,:,i] = 25.4 * i .+ velmod[:,:,i]
    end


    # source coordinates
    coordsrc1 = [grd.hgrid*(grd.grsize[1]÷2)+0.312*grd.hgrid  grd.hgrid*(grd.grsize[2]÷2)-0.712*grd.hgrid  0.92*grd.z[end]]


    tolerr = [2.74e-5,
              5.71e-4]

    tolnorm = [3.24e-5,
               6.27e-4]


    r=1
    for refinesrc in [false,true]
    
        extraparams=ExtraParams(refinearoundsrc=refinesrc,
                                grdrefpars=GridRefinementPars(downscalefactor=3,noderadius=3),
                                radiussmoothgradsrc = 0 )

        ttpicks = eiktraveltime(velmod,grd,coordsrc1,coordrec,
                                extraparams=extraparams)

        # standard deviation of error on observed data
        stdobs = [0.0001.*ones(size(ttpicks[1])) for i=1:size(coordsrc1,1)]
        # generate a "noise" array to simulate real data
        #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
        # add the noise to the synthetic traveltime data
        dobs = ttpicks #.+ noise

        # # create a guess/"current" model
        vel0 = 2200.0 .* ones(nx,ny,nz)
        ## increasing velocity with depth...
        for i=1:nz
           vel0[:,:,i] = 32.5 * i .+ vel0[:,:,i]
        end

        coordsrc2 = similar(coordsrc1)
        for s=1:size(coordsrc2,1)
            coordsrc2[s,:] = coordsrc1[s,:] .+ [3.45*grd.hgrid, -4.45*grd.hgrid, -2.23*grd.hgrid]
        end

        ## calculate the gradient of the misfit function
        gradsrcloc = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                 :gradsrcloc,extraparams=extraparams)

        
        dh = 0.001

        gradsrcloc_FD = similar(gradsrcloc)
        Ndim = 3
        ∂ψ∂xysrc = zeros(size(coordsrc2,1),Ndim)
        for s=1:size(coordsrc2,1)
            for axis=1:Ndim
                coordsrc_plusdx  = copy(coordsrc2[s:s,:])
                coordsrc_minusdx = copy(coordsrc2[s:s,:])

                coordsrc_plusdx[axis] += dh
                misf_pdx = eikttimemisfit(vel0,dobs,stdobs,coordsrc_plusdx,
                                          coordrec,grd;
                                          extraparams=extraparams)
                coordsrc_minusdx[axis] -= dh
                misf_mdx = eikttimemisfit(vel0,dobs,stdobs,coordsrc_minusdx,
                                          coordrec,grd;
                                          extraparams=extraparams)
                gradsrcloc_FD[s,axis] = (misf_pdx-misf_mdx)/(2*dh)
            end
        end
               
        # @show extrema(gradsrcloc)
        # @show extrema(gradsrcloc_FD)


        normerr = norm(abs.(gradsrcloc.-gradsrcloc_FD))
        maxerr = maximum(abs.(gradsrcloc.-gradsrcloc_FD))

        # @show maxerr,tolerr[r]
        # @show normerr,tolnorm[r]

        @test all((maxerr<=tolerr[r] && normerr<=tolnorm[r] ))
        r+=1
    end

    return
end

###################################

# function test_fwdtt_2D_alternative()
    
#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod2D()

#     #--------------------------------------
#     ## Traveltime forward
#     ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
#     for ttalgo in ttalgos
#         println("Traveltime 2D using $ttalgo")
#         ttpicks = traveltime2Dalt(velmod,grd,coordsrc,coordrec,ttalgo=ttalgo)
#     end
    
#     #########################
#     # Analytical solution
#     # 
#     ## source must be *on* the top surface for analytic solution
#     coordsrc=[31.0 0.0;]
#     # @show coordsrc
#     ansol2d,velanaly = analyticalsollingrad2D(grd,coordsrc[1,1],coordsrc[1,2])
#     ## contour(permutedims(ansol2d),colors="black")

#     ## Numerical traveltime
#     mae = Dict()
#     colors = ["red","blue","green"]
#     ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
#     i=0
#     for ttalgo in ttalgos
#         i+=1
#         println("Traveltime 2D using $ttalgo versus analytic solution")
#         ttpicks,ttime = traveltime2Dalt(velanaly,grd,coordsrc[1:1,:],coordrec[1:1],ttalgo=ttalgo,returntt=true)
#         # mean average error
#         if ttalgo in ["ttFS_podlec","ttFMM_podlec"]
#             ttime = (ttime[1:end-1,1:end-1].+ttime[2:end,1:end-1] .+
#                 ttime[1:end-1,2:end].+ttime[2:end,2:end]) ./ 4.0
#         end        
#         mae[ttalgo] = (sum(abs.(ttime[:,:,1]-ansol2d)))/length(ansol2d)
        
#     end

#     @show mae
#     cerr = [er for er in values(mae)]
#     if all(cerr.<=[0.21,0.24,0.24])
#         return true
#     else
#         return false
#     end
# end 

# #############################################


# function test_fwdtt_3D_FMM2ndord()

#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod3D()
    
#     #--------------------------------------
#     ## Traveltime forward
#     println("Traveltime 3D using default algo")
#     ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)
                           
#     return true
# end

# #############################################

# function test_fwdtt_3D_alternative()

#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod3D()
    
#     #--------------------------------------
#     ## Traveltime forward
#     ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
#     for ttalgo in ttalgos
#         println("Traveltime 3D using $ttalgo")
#         ttpicks = traveltime3Dalt(velmod,grd,coordsrc,coordrec,ttalgo=ttalgo)
#     end
    
#     return true
# end 

# #############################################

# function test_gradtt_2D_FMM2ndord()

#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod2D()

#     #--------------------------------------
#     # Gradient of misfit
#     ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
#     stdobs = deepcopy(ttpicks)
#     dobs = deepcopy(ttpicks)
#     for i=1:length(ttpicks)
#         stdobs[i] .= 0.15
#         dobs[i] = ttpicks[i] .+ stdobs[i].^2 .* randn(size(ttpicks[i]))
#     end
#     flatmod = 2.8 .+ zeros(nx,ny) 
    
#     println("Gradient 2D using default algo")
#     grad = gradttime2D(flatmod,grd,coordsrc,coordrec,dobs,stdobs)

#     return true
# end


# #############################################

# function test_gradtt_2D_alternative()

#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod2D()

#     #--------------------------------------
#     # Gradient of misfit
#     ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
#     stdobs = deepcopy(ttpicks)
#     dobs = deepcopy(ttpicks)
#     for i=1:length(ttpicks)
#         stdobs[i] .= 0.15
#         dobs[i] = ttpicks[i] .+ stdobs[i].^2 .* randn(size(ttpicks[i]))
#     end
#     flatmod = 2.8 .+ zeros(nx,ny) 
    
#     gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
#     for gradalgo in gradalgos
#         println("Gradient 2D using $gradalgo")
#         grad = gradttime2Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
#     end

#     return true
# end 

# #############################################
    
# function test_gradtt_3D_FMM2ndord()

#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod3D()

#     #--------------------------------------
#     # Gradient of misfit
#     ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

#     nsrc = 3
#     stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
#     noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
#     dobs = ttpicks .+ noise
    
#     flatmod = 2.8 .+ zeros(nx,ny,grd.nz) 
    
#     println("Gradient 3D using default algo")
#     grad = gradttime3D(flatmod,grd,coordsrc,coordrec,dobs,stdobs)
    
#     return true
# end 

# #############################################
    
# function test_gradtt_3D_alternative()

#     #--------------------------------------
#     # Create a grid and a velocity model
#     grd,coordsrc,coordrec,velmod = creategridmod3D()

#     #--------------------------------------
#     # Gradient of misfit
#     ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

#     nsrc = 3
#     stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
#     noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
#     dobs = ttpicks .+ noise
    
#     flatmod = 2.8 .+ zeros(nx,ny,grd.nz) 
    
#     gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
#     for gradalgo in gradalgos
#         println("Gradient 3D using $gradalgo")
#         grad = gradttime3Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
#     end

#     return true
# end

# ######################################################
