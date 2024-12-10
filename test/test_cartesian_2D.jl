
#####################################################

function test_fwdtt_2D_constvel()

    Ntests = 4
    Ndim = 2
    
    maxerr = Vector{Float64}(undef,Ntests)

    # Create a grid and a velocity model
    grd = Grid2DCart(hgrid=250.0,cooinit=(0.0,0.0),grsize=(111,121))
    nx,ny = grd.grsize

    coordsrcs = [[grd.hgrid+grd.x[nx÷2]  grd.hgrid+grd.y[ny÷2]],
                 [grd.hgrid/2+grd.x[nx÷2]  grd.hgrid/2+grd.y[ny÷2]]]

    tolerr = [0.032895,
              0.012284,
              0.053850,
              0.026170]

    tolnorm = [2.37,
               0.87,
               5.15, #3.24,
               2.14] #1.53]               

    r=0
    for coordsrc in coordsrcs

        for refinearoundsrc in [false,true]
            
            extrapars = ExtraParams(refinearoundsrc=refinearoundsrc)
            
            #--------------------------------------
            ## Traveltime forward
            #printstyled("  Traveltime $(Ndim)-D vs. analytical solution (constant velocity), refin.=$refinearoundsrc.\n",bold=true,color=:cyan)

            constvelmod = 2500.0 .* ones(nx,ny)
            coordrec = [[grd.x[2] grd.y[3]]]
            
            _,ttime = eiktraveltime(constvelmod,grd,coordsrc,coordrec,
                                    returntt=true,extraparams=extrapars)
            
            # analytical solution for constant velocity
            xsrc,ysrc=coordsrc[1,1],coordsrc[1,2]
            ttanalconstvel = zeros(nx,ny)
            diffttanalconstvel = zeros(nx,ny)
            for j=1:ny
                for i=1:nx
                    ttanalconstvel[i,j] = sqrt((grd.x[i]-xsrc)^2+(grd.y[j]-ysrc)^2)/constvelmod[i,j]
                    diffttanalconstvel[i,j] = ttime[1][i,j] - ttanalconstvel[i,j]
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


function analyticalsollingrad2D(grd::Grid2DCart,xsrcpos::Float64,ysrcpos::Float64)
    ##################################
    ## linear gradient of velocity
    nx,ny = grd.grsize

    ## source must be *on* the top surface
    #@show ysrcpos
    @assert (ysrcpos == 0.0)

    # position of the grid nodes
    xgridpos = grd.x #[(i-1)*grd.hgrid for i=1:nx] .+ grd.xinit
    ygridpos = grd.y #[(i-1)*grd.hgrid for i=1:ny] .+ grd.yinit
    # base velocity
    vel0 = 2.0
    # gradient of velociy
    gr = 0.05
    ## construct the 2D velocity model
    velmod = zeros(nx,ny) 
    for i=1:nx
        velmod[i,:] = vel0 .+ gr .* (ygridpos .- ysrcpos)
    end

    # https://pubs.geoscienceworld.org/books/book/1011/lessons-in-seismic-computing
    # Lesson No. 41: Linear Distribution of Velocity—V. The Wave-Fronts 

    ## Analytic solution
    ansol2d  = zeros(nx,ny)
    for j=1:ny, i=1:nx
        x = xgridpos[i]-xsrcpos
        h = ygridpos[j]-ysrcpos
        ansol2d[i,j] = 1/gr * acosh( 1 + (gr^2*(x^2+h^2))/(2*vel0*velmod[i,j]) )
    end
    return ansol2d,velmod
end

########################################################f

function test_fwdtt_2D_lingrad()

    
    Ntests = 4
    Ndim = 2
    
    maxerr = Vector{Float64}(undef,Ntests)

    tolerr = [1.69,
              2.65,
              2.65,
              2.46]

    tolnorm = [67.0,
               67.5,
               182.0,
               52.6]
                  
    #--------------------------------------
    # Create a grid and a velocity model
    grd = Grid2DCart(hgrid=15.0,cooinit=(0.0,0.0),grsize=(111,121))
    nx,ny = grd.grsize

    #########################
    # Analytical solution
    #
    ## source must be *on* the top surface for analytic solution
    coordsrcs = [[grd.x[end÷2]+grd.hgrid  0.0],
                 [grd.x[end÷2]+0.253*grd.hgrid  0.0]]

    coordrec = [[2.0 3.0]]

    r=0
    for coordsrc in coordsrcs

        ansol2d,velanaly = analyticalsollingrad2D(grd,coordsrc[1,1],coordsrc[1,2])

        for refinearoundsrc in [false,true]
            
            extrapars = ExtraParams(refinearoundsrc=refinearoundsrc)

            _,ttime = eiktraveltime(velanaly,grd,coordsrc,coordrec,
                                    returntt=true,extraparams=extrapars)

            r+=1
            maxerr = maximum(abs.(ttime[1].-ansol2d))
            normerr = norm(abs.(ttime[1].-ansol2d))

            # @show r,maxerr,tolerr[r]
            # @show r,normerr,tolnorm[r]
            @test all((maxerr<=tolerr[r], normerr<=tolnorm[r]))
        end
    end
    return
end

###################################

function test_gradvel_2D()

    # create the Grid2D struct
    grd = Grid2DCart(hgrid=20.0,
                     cooinit=(0.0,0.0),
                     grsize=(50,30))
    nx,ny = grd.grsize

    nrec = 7
    coordrec = [[grd.hgrid.*LinRange(3.73,nx-4.34,nrec)  grd.hgrid.*3.7.*ones(nrec);
                 grd.hgrid*(nx÷2)+0.6*grd.hgrid/2  grd.hgrid*(ny÷2)+0.5*grd.hgrid/2;
                 grd.hgrid.*LinRange(2.73,nx-3.34,nrec)  grd.hgrid.*(ny-4)*ones(nrec)]]

    velmod = 2500.0 .* ones(nx,ny)
    for i=1:ny
        velmod[:,i] = 23.6 * i .+ velmod[:,i]
    end

    coordsrc1 = [grd.hgrid*(nx÷2)+0.23*grd.hgrid  grd.hgrid*(ny÷2)-0.6*grd.hgrid]

    tolerr = [7.04e-6,
              7.57e-6]

    tolnorm = [4.02e-5
               4.35e-5]

    r=0
    for refinesrc in [false,true]
    
        extraparams=ExtraParams(refinearoundsrc=refinesrc,
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
        vel0 = 2200.0 .* ones(nx,ny)
        ## increasing velocity with depth...
        for i=1:ny
           vel0[:,i] = 32.5 * i .+ vel0[:,i]
        end

        coordsrc2 = copy(coordsrc1)

        ## calculate the gradient of the misfit function
        gradvel,misf1 = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                :gradvel,extraparams=extraparams)

        misf2 = eikttimemisfit(vel0,grd,coordsrc2,
                               coordrec,dobs,stdobs;
                               extraparams=extraparams)
        dh = 0.001

        gradvel_FD = similar(gradvel)
        for j=1:ny
            for i=1:nx
                velpdh = copy(vel0)
                velpdh[i,j] += dh
                velmdh = copy(vel0)
                velmdh[i,j] -= dh
                misf_pdh = eikttimemisfit(velpdh,grd,coordsrc2,coordrec,dobs,stdobs;
                                        extraparams=extraparams)
                misf_mdh = eikttimemisfit(velmdh,grd,coordsrc2,coordrec,dobs,stdobs;
                                        extraparams=extraparams)
                gradvel_FD[i,j] = (misf_pdh-misf_mdh)/(2*dh)
            end
        end

        r+=1
        normerr = norm(abs.(gradvel.-gradvel_FD))
        maxerr = maximum(abs.(gradvel.-gradvel_FD))
        # @show normerr
        # @show maxerr
        
        @test misf1≈misf2
        @test all((maxerr<=tolerr[r] && normerr<=tolnorm[r] ))
    end

    return
end


###################################

function test_gradsrc_2D()

    # create the Grid2D struct
    grd = Grid2DCart(hgrid=20.0,
                     cooinit=(0.0,0.0),
                     grsize=(50,30))
    nx,ny = grd.grsize

    nrec = 7
    coordrec = [[grd.hgrid.*LinRange(3.73,nx-4.34,nrec)  grd.hgrid.*3.7.*ones(nrec);
                 grd.hgrid*(nx÷2)+0.6*grd.hgrid/2  grd.hgrid*(ny÷2)+0.5*grd.hgrid/2;
                 grd.hgrid.*LinRange(2.73,nx-3.34,nrec)  grd.hgrid.*(ny-4)*ones(nrec)]]

    velmod = 2500.0 .* ones(nx,ny)

    for i=1:ny
        velmod[:,i] = 23.6 * i .+ velmod[:,i]
    end

    coordsrc1 = [grd.hgrid*(nx÷2)+0.23*grd.hgrid  grd.hgrid*(ny÷2)-0.6*grd.hgrid]

    tolerr = [3.83e-4,
              2.26e-4]

    tolnorm = [5.23e-4
               3.13e-4]

    r=0
    for refinesrc in [false,true]
    
        extraparams=ExtraParams(refinearoundsrc=refinesrc,
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
        vel0 = 2200.0 .* ones(nx,ny)
        ## increasing velocity with depth...
        for i=1:ny
           vel0[:,i] = 32.5 * i .+ vel0[:,i]
        end

        coordsrc2 = similar(coordsrc1)
        for s=1:size(coordsrc2,1)
            coordsrc2[s,:] = coordsrc1[s,:] .+ [3.45*grd.hgrid, -9.45*grd.hgrid]
        end

        ## calculate the gradient of the misfit function
        gradsrcloc,misf1 = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                   :gradsrcloc,extraparams=extraparams)

        misf2 = eikttimemisfit(vel0,grd,coordsrc2,
                                  coordrec,dobs,stdobs;
                                  extraparams=extraparams)


        dh = 0.0001

        gradsrcloc_FD = similar(gradsrcloc)
        Ndim = 2
        ∂ψ∂xysrc = zeros(size(coordsrc2,1),Ndim)
        for s=1:size(coordsrc2,1)
            for axis=1:2
                coordsrc_plusdx  = copy(coordsrc2[s:s,:])
                coordsrc_minusdx = copy(coordsrc2[s:s,:])

                coordsrc_plusdx[axis] += dh
                misf_pdx = eikttimemisfit(vel0,grd,coordsrc_plusdx,
                                          coordrec,dobs,stdobs;
                                          extraparams=extraparams)
                coordsrc_minusdx[axis] -= dh
                misf_mdx = eikttimemisfit(vel0,grd,coordsrc_minusdx,
                                          coordrec,dobs,stdobs;
                                          extraparams=extraparams)
                gradsrcloc_FD[s,axis] = (misf_pdx-misf_mdx)/(2*dh)
            end
        end
               

        r+=1
        normerr = norm(abs.(gradsrcloc.-gradsrcloc_FD))
        maxerr = maximum(abs.(gradsrcloc.-gradsrcloc_FD))

        # @show gradsrcloc
        # @show gradsrcloc_FD
        # @show normerr #-tolnorm[r]
        # @show maxerr #-tolerr[r]

        @test misf1≈misf2
        @test all((maxerr<=tolerr[r] && normerr<=tolnorm[r] ))
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
