
###################################

function test_fwdtt_2D_constvel()

    Ntests = 4
    Ndim = 2
    
    maxerr = Vector{Float64}(undef,Ntests)

    # Create a grid and a velocity model
    grd = Grid2DCart(hgrid=250.0,xinit=0.0,yinit=0.0,nx=111,ny=121)

    coordsrcs = [[grd.hgrid+grd.x[grd.nx÷2]  grd.hgrid+grd.y[grd.ny÷2]],
                 [grd.hgrid/2+grd.x[grd.nx÷2]  grd.hgrid/2+grd.y[grd.ny÷2]]]

    tolerr = [0.032895,
              0.012284,
              0.053837,
              0.025700]

    r=0
    for coordsrc in coordsrcs

        for refinearoundsrc in [false,true]
            
            extrapars = ExtraParams(refinearoundsrc=refinearoundsrc)
            
            #--------------------------------------
            ## Traveltime forward
            #printstyled("  Traveltime $(Ndim)-D vs. analytical solution (constant velocity), refin.=$refinearoundsrc.\n",bold=true,color=:cyan)

            constvelmod = 2500.0 .* ones(grd.nx,grd.ny)
            coordrec = [[grd.x[2] grd.y[3]]]
            
            _,ttime = eiktraveltime(constvelmod,grd,coordsrc,coordrec,
                                    returntt=true,extraparams=extrapars)
            
            # analytical solution for constant velocity
            xsrc,ysrc=coordsrc[1,1],coordsrc[1,2]
            ttanalconstvel = zeros(grd.nx,grd.ny)
            diffttanalconstvel = zeros(grd.nx,grd.ny)
            for j=1:grd.ny
                for i=1:grd.nx
                    ttanalconstvel[i,j] = sqrt((grd.x[i]-xsrc)^2+(grd.y[j]-ysrc)^2)/constvelmod[i,j]
                    diffttanalconstvel[i,j] = ttime[1][i,j] - ttanalconstvel[i,j]
                end
            end

            r+=1
            maxerr[r] = maximum(abs.(diffttanalconstvel))

            # @show tolerr[r],maxerr[r]
            # @show tolerr[r]-maxerr[r]
            # @show maxerr[r]<= tolerr[r]

            @test maxerr[r]<= tolerr[r]
        end
    end
    
    if all( maximum.(maxerr) .<= tolerr )
        return true
    else
        return false
    end
    return 
end

#############################################


function analyticalsollingrad2D(grd::Grid2DCart,xsrcpos::Float64,ysrcpos::Float64)
    ##################################
    ## linear gradient of velocity
     
    ## source must be *on* the top surface
    #@show ysrcpos
    @assert (ysrcpos == 0.0)

    # position of the grid nodes
    xgridpos = grd.x #[(i-1)*grd.hgrid for i=1:grd.nx] .+ grd.xinit
    ygridpos = grd.y #[(i-1)*grd.hgrid for i=1:grd.ny] .+ grd.yinit
    # base velocity
    vel0 = 2.0
    # gradient of velociy
    gr = 0.05
    ## construct the 2D velocity model
    velmod = zeros(grd.nx,grd.ny) 
    for i=1:grd.nx
        velmod[i,:] = vel0 .+ gr .* (ygridpos .- ysrcpos)
    end

    # https://pubs.geoscienceworld.org/books/book/1011/lessons-in-seismic-computing
    # Lesson No. 41: Linear Distribution of Velocity—V. The Wave-Fronts 

    ## Analytic solution
    ansol2d  = zeros(grd.nx,grd.ny)
    for j=1:grd.ny, i=1:grd.nx
        x = xgridpos[i]-xsrcpos
        h = ygridpos[j]-ysrcpos
        ansol2d[i,j] = 1/gr * acosh( 1 + (gr^2*(x^2+h^2))/(2*vel0*velmod[i,j]) )
    end
    return ansol2d,velmod
end

########################################################f

function test_fwdtt_2D_lingrad()

    
    Ntests = 2
    Ndim = 2
    
    maxerr = Vector{Float64}(undef,Ntests)

    tolerr = [58.05695,
              73.78549]
                  
    #--------------------------------------
    # Create a grid and a velocity model
    grd = Grid2DCart(hgrid=250.0,xinit=0.0,yinit=0.0,nx=111,ny=121)
    

    #########################
    # Analytical solution
    #
    ## source must be *on* the top surface for analytic solution
    coordsrc=[grd.x[end÷2]+grd.hgrid*2.423  0.0]
    ansol2d,velanaly = analyticalsollingrad2D(grd,coordsrc[1,1],coordsrc[1,2])

    coordrec = [[2.0 3.0]]

    r=0
    for refinearoundsrc in [false,true]
        
        extrapars = ExtraParams(refinearoundsrc=refinearoundsrc)

        _,ttime = eiktraveltime(velanaly,grd,coordsrc,coordrec,
                                returntt=true,extraparams=extrapars)

        @show extrema(ansol2d)
        @show extrema(ttime[1])

        r+=1
        maxerr[r] = maximum(abs.(ttime[1].-ansol2d))
        @show maxerr[r],tolerr[r]
        
        @test maxerr[r]<=tolerr[r]
    end

    if all( maximum.(maxerr) .<= tolerr )
        return true
    else
        return false
    end
    return
end

#     ## Numerical traveltime
#     #mae = Dict()
#     # colors = ["red","blue","green"]
#     println("Traveltime 2D using default algo versus analytic solution")
#     ttpicks,ttime2 = eiktraveltime(velanaly,grd,coordsrc[1:1,:],coordrec[1:1],returntt=true)

#     display(ttime2[1])
#     display(ansol2d)
#     # mean average error
#     mae = (sum(abs.(ttime2[1]-ansol2d)))/length(ansol2d)

#     @show (abs.(ttime2[1].-ansol2d))./abs.(ansol2d) * 100
#     @show mae
#     if mae<=0.21
#         return true
#     else
#         return false
#     end
# end

# ###################################

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
#     flatmod = 2.8 .+ zeros(grd.nx,grd.ny) 
    
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
#     flatmod = 2.8 .+ zeros(grd.nx,grd.ny) 
    
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
    
#     flatmod = 2.8 .+ zeros(grd.nx,grd.ny,grd.nz) 
    
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
    
#     flatmod = 2.8 .+ zeros(grd.nx,grd.ny,grd.nz) 
    
#     gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
#     for gradalgo in gradalgos
#         println("Gradient 3D using $gradalgo")
#         grad = gradttime3Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
#     end

#     return true
# end

# ######################################################
