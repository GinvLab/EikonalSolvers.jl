
###################################

function test_fwdtt_2D_constvel()
    

    # Create a grid and a velocity model
    grd = Grid2DCart(hgrid=250.0,xinit=0.0,yinit=0.0,nx=101,ny=121)

    #--------------------------------------
    ## Traveltime forward
    println("Traveltime 2D using default algo - FMM 2nd order")

    constvelmod = 2500.0 .* ones(grd.nx,grd.ny)
    coordrec = [[grd.x[2] grd.y[3]]]

    coordsrc = [grd.hgrid/5+grd.x[grd.nx÷2]  grd.hgrid/1.2+grd.y[grd.ny÷2]]
   
    extrapars = ExtraParams(refinearoundsrc=false)
    println("Refinement around source? $(extrapars.refinearoundsrc) ")
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

    @show coordsrc
    @show extrema(ttanalconstvel)
    @show extrema(diffttanalconstvel)
    
    return ttanalconstvel,ttime[1],grd,coordsrc,extrapars
end

###################################

function test_fwdtt_2D_lingrad()

    #########################
    # Analytical solution
    #
    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2D()

                                       # single source
    ## source must be *on* the top surface for analytic solution
    coordsrc=[31.734 0.0]
    # @show coordsrc
    ansol2d,velanaly = analyticalsollingrad2D(grd,coordsrc[1,1],coordsrc[1,2])
    ## contour(permutedims(ansol2d),colors="black")

    ## Numerical traveltime
    #mae = Dict()
    # colors = ["red","blue","green"]
    println("Traveltime 2D using default algo versus analytic solution")
    ttpicks,ttime2 = eiktraveltime(velanaly,grd,coordsrc[1:1,:],coordrec[1:1],returntt=true)

    display(ttime2[1])
    display(ansol2d)
    # mean average error
    mae = (sum(abs.(ttime2[1]-ansol2d)))/length(ansol2d)

    @show (abs.(ttime2[1].-ansol2d))./abs.(ansol2d) * 100
    @show mae
    if mae<=0.21
        return true
    else
        return false
    end
end

###################################

function test_fwdtt_2D_alternative()
    
    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2D()

    #--------------------------------------
    ## Traveltime forward
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    for ttalgo in ttalgos
        println("Traveltime 2D using $ttalgo")
        ttpicks = traveltime2Dalt(velmod,grd,coordsrc,coordrec,ttalgo=ttalgo)
    end
    
    #########################
    # Analytical solution
    # 
    ## source must be *on* the top surface for analytic solution
    coordsrc=[31.0 0.0;]
    # @show coordsrc
    ansol2d,velanaly = analyticalsollingrad2D(grd,coordsrc[1,1],coordsrc[1,2])
    ## contour(permutedims(ansol2d),colors="black")

    ## Numerical traveltime
    mae = Dict()
    colors = ["red","blue","green"]
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    i=0
    for ttalgo in ttalgos
        i+=1
        println("Traveltime 2D using $ttalgo versus analytic solution")
        ttpicks,ttime = traveltime2Dalt(velanaly,grd,coordsrc[1:1,:],coordrec[1:1],ttalgo=ttalgo,returntt=true)
        # mean average error
        if ttalgo in ["ttFS_podlec","ttFMM_podlec"]
            ttime = (ttime[1:end-1,1:end-1].+ttime[2:end,1:end-1] .+
                ttime[1:end-1,2:end].+ttime[2:end,2:end]) ./ 4.0
        end        
        mae[ttalgo] = (sum(abs.(ttime[:,:,1]-ansol2d)))/length(ansol2d)
        
    end

    @show mae
    cerr = [er for er in values(mae)]
    if all(cerr.<=[0.21,0.24,0.24])
        return true
    else
        return false
    end
end 

#############################################


function test_fwdtt_3D_FMM2ndord()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()
    
    #--------------------------------------
    ## Traveltime forward
    println("Traveltime 3D using default algo")
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)
                           
    return true
end

#############################################

function test_fwdtt_3D_alternative()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()
    
    #--------------------------------------
    ## Traveltime forward
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    for ttalgo in ttalgos
        println("Traveltime 3D using $ttalgo")
        ttpicks = traveltime3Dalt(velmod,grd,coordsrc,coordrec,ttalgo=ttalgo)
    end
    
    return true
end 

#############################################

function test_gradtt_2D_FMM2ndord()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2D()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
    stdobs = deepcopy(ttpicks)
    dobs = deepcopy(ttpicks)
    for i=1:length(ttpicks)
        stdobs[i] .= 0.15
        dobs[i] = ttpicks[i] .+ stdobs[i].^2 .* randn(size(ttpicks[i]))
    end
    flatmod = 2.8 .+ zeros(grd.nx,grd.ny) 
    
    println("Gradient 2D using default algo")
    grad = gradttime2D(flatmod,grd,coordsrc,coordrec,dobs,stdobs)

    return true
end


#############################################

function test_gradtt_2D_alternative()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2D()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
    stdobs = deepcopy(ttpicks)
    dobs = deepcopy(ttpicks)
    for i=1:length(ttpicks)
        stdobs[i] .= 0.15
        dobs[i] = ttpicks[i] .+ stdobs[i].^2 .* randn(size(ttpicks[i]))
    end
    flatmod = 2.8 .+ zeros(grd.nx,grd.ny) 
    
    gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
    for gradalgo in gradalgos
        println("Gradient 2D using $gradalgo")
        grad = gradttime2Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
    end

    return true
end 

#############################################
    
function test_gradtt_3D_FMM2ndord()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

    nsrc = 3
    stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
    noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    dobs = ttpicks .+ noise
    
    flatmod = 2.8 .+ zeros(grd.nx,grd.ny,grd.nz) 
    
    println("Gradient 3D using default algo")
    grad = gradttime3D(flatmod,grd,coordsrc,coordrec,dobs,stdobs)
    
    return true
end 

#############################################
    
function test_gradtt_3D_alternative()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

    nsrc = 3
    stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
    noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    dobs = ttpicks .+ noise
    
    flatmod = 2.8 .+ zeros(grd.nx,grd.ny,grd.nz) 
    
    gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
    for gradalgo in gradalgos
        println("Gradient 3D using $gradalgo")
        grad = gradttime3Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
    end

    return true
end

######################################################
