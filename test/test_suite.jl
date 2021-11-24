


using Distributed
using EikonalSolvers

# using PyPlot

###################################
######################################################
##         Cartesian coordinates                    ##
######################################################

function creategridmod2D()

    hgrid = 2.5 
    xinit = 0.0
    yinit = 0.0
    nx = 100 
    ny = 70 
    grd = Grid2D(hgrid=hgrid,xinit=xinit,yinit=yinit,nx=nx,ny=ny)
        
    xmax = xinit + (nx-1)*hgrid
    ymax = yinit + (ny-1)*hgrid
    println("Grid size (velocity): $nx x $ny from ($xinit,$yinit) to  ($xmax,$ymax) ")
    println("Model parameters: $(nx*ny)")
    
    ######################################
    nsrc = 3
    coordsrc = [hgrid*LinRange(5.5,nx-15,nsrc)  (ny*hgrid-5.6).+(hgrid*LinRange(-1.0,1.0,nsrc))]

    nrec = 10
    coordrec = [[hgrid*LinRange(3.0,nx-2,nrec)  1.75*hgrid*LinRange(2.0,2.0,nrec)] for i=1:nsrc]
       
    ######################################
        
    velmod = 2.5 .+ zeros(nx,ny) 
    for i=1:ny
        velmod[:,i] = 0.01 * i .+ velmod[:,i]
    end

    return grd,coordsrc,coordrec,velmod
end


###################################

function creategridmod3D()

    hgrid = 2.5 
    xinit = 0.0
    yinit = 0.0
    zinit = 0.0
    nx = 30 
    ny = 30
    nz = 20
    grd = Grid3D(hgrid=hgrid,xinit=xinit,yinit=yinit,zinit=zinit,nx=nx,ny=ny,nz=nz)
        
    xmax = xinit + (nx-1)*hgrid
    ymax = yinit + (ny-1)*hgrid
    zmax = zinit + (nz-1)*hgrid
    println("Grid size (velocity): $nx x $ny x $nz from ($xinit,$yinit,$zinit) to  ($xmax,$ymax,$zmax) ")
    println("Model parameters: $(nx*ny*nz)")
    
    ######################################
    nsrc = 3
    coordsrc = [hgrid*LinRange(3.5,nx-15,nsrc)  hgrid*LinRange(3.5,ny-15,nsrc)  (nz*hgrid-6).+(hgrid*LinRange(-1.0,1.0,nsrc))]

    nrec = 10
    coordrec = [hgrid*LinRange(2.0,nx-2,nrec)  1.75*hgrid*LinRange(2.0,2.0,nrec) 1.75*hgrid*LinRange(2.0,2.0,nrec)] 


    ######################################
        
    velmod = 2.5 .+ zeros(nx,ny,nz) 
    for i=1:nz
        velmod[:,:,i] = 0.01 * i .+ velmod[:,:,i]
    end

    return  grd,coordsrc,coordrec,velmod
end

#############################################
#############################################

function analyticalsollingrad2D(grd::Grid2D,xsrcpos::Float64,ysrcpos::Float64)
    ##################################
    ## linear gradient of velocity
     
    ## source must be *on* the top surface
    #@show ysrcpos
    @assert (ysrcpos == 0.0)

    # position of the grid nodes
    xgridpos = [(i-1)*grd.hgrid for i=1:grd.nx] .+ grd.xinit
    ygridpos = [(i-1)*grd.hgrid for i=1:grd.ny] .+ grd.yinit
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

###################################
###################################

function test_fwdtt_2D()
    
    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2D()

    #--------------------------------------
    ## Traveltime forward
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    for ttalgo in ttalgos
        println("Traveltime 2D using $ttalgo")
        ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec,ttalgo=ttalgo)
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
        ttpicks,ttime = traveltime2D(velanaly,grd,coordsrc[1:1,:],coordrec[1:1],ttalgo=ttalgo,returntt=true)
        # mean average error
        if ttalgo in ["ttFS_podlec","ttFMM_podlec"]
            ttime = (ttime[1:end-1,1:end-1].+ttime[2:end,1:end-1] .+
                ttime[1:end-1,2:end].+ttime[2:end,2:end]) ./ 4.0
        end        
        mae[ttalgo] = (sum(abs.(ttime[:,:,1]-ansol2d)))/length(ansol2d)
        
        ## contour(permutedims(ttime[:,:,1]),colors=colors[i])
    end

    cerr = [er for er in values(mae)]
    if all(cerr.<=[0.04,0.24,0.24])
        return true
    else
        return false
    end
end 

#############################################

function test_fwdtt_3D()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()
    
    #--------------------------------------
    ## Traveltime forward
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    for ttalgo in ttalgos
        println("Traveltime 3D using $ttalgo")
        ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec,ttalgo=ttalgo)
    end
    
    return true
end 

#############################################

function test_gradtt_2D()

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
        grad = gradttime2D(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
    end

    return true
end 

#############################################
    
function test_gradtt_3D()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()

    #--------------------------------------
    # Gradient of misfit
    stdobs = 0.15 .* ones(size(coordrec,1),size(coordsrc,1))
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)
    noise = stdobs.^2 .* randn(size(ttpicks))
    dobs = ttpicks .+ noise
    flatmod = 2.8 .+ zeros(grd.nx,grd.ny,grd.nz) 
    
    gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
    for gradalgo in gradalgos
        println("Gradient 3D using $gradalgo")
        grad = gradttime3D(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
    end

    return true
end 

#############################################
#############################################
#############################################







######################################################
##         Spherical coordinates                    ##
######################################################

###################################

function creategridmod2Dsphere()

    Δr = 2.0
    Δθ = 0.2
    nr = 40
    nθ = 70
    rinit = 500.0
    θinit = 0.0    
    grd = Grid2DSphere(Δr=Δr,Δθ=Δθ,nr=nr,nθ=nθ,rinit=rinit,θinit=θinit)
        
    rmax = grd.r[end]
    θmax = grd.θ[end]
    println("Grid size (velocity): $nr x $nθ from ($rinit,$θinit) to  ($rmax,$θmax) ")
    println("Model parameters: $(nr*nθ)")
    
    ######################################
    nsrc = 3
    coordsrc = [LinRange(rinit+5,rmax-15,nsrc)  (nθ*Δθ-5.6).+(Δθ*LinRange(-1.0,1.0,nsrc))]

    nrec = 10
    coordrec = [LinRange(rinit+3.0,rmax-2,nrec)  1.75*Δθ*LinRange(2.0,2.0,nrec)]
    
    ######################################
        
    velmod = 2.7 .+ zeros(nr,nθ) 
    for i=1:nr
        velmod[i,:] = -0.01 * i .+ velmod[i,:]
    end

    return grd,coordsrc,coordrec,velmod
end


###################################

function creategridmod3Dsphere()

    Δr = 2.0
    Δθ = 1.2
    Δφ = 1.0
    rinit = 500.0
    θinit = 60.0
    φinit = 3.0
    nr = 30 
    nθ = 30
    nφ = 20
    grd = Grid3DSphere(Δr=Δr,Δθ=Δθ,Δφ=Δφ,rinit=rinit,θinit=θinit,φinit=φinit,nr=nr,nθ=nθ,nφ=nφ)
        
    rmax = grd.r[end]
    θmax = grd.θ[end]
    φmax = grd.φ[end]

    println("Grid size (velocity): $nr x $nθ x $nφ from ($rinit,$θinit,$φinit) to  ($rmax,$θmax,$φmax) ")
    println("Model parameters: $(nr*nθ*nφ)")
    
    ######################################
    nsrc = 3
    coordsrc = [LinRange(rinit+3.5,rmax-15,nsrc)  LinRange(θinit+3.5,θmax-15,nsrc)  (nφ*Δφ-6).+(Δφ*LinRange(-1.0,1.0,nsrc))]

    nrec = 10
    coordrec = [LinRange(rinit+2.0,rmax-2,nrec)  θinit.+1.75*Δθ*LinRange(2.0,2.0,nrec) 1.75*Δφ*LinRange(2.0,2.0,nrec)]
    
    ######################################
        
    velmod = 2.7 .+ zeros(nr,nθ,nφ) 
    for i=1:nr
        velmod[i,:,:] = - 0.01 * i .+ velmod[i,:,:]
    end

    return  grd,coordsrc,coordrec,velmod
end

#############################################

function test_fwdtt_2Dsphere()
    
    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2Dsphere()

    #--------------------------------------
    ## Traveltime forward
    println("Traveltime 2D in spherical coordinates")
    ttpicks = traveltime2Dsphere(velmod,grd,coordsrc,coordrec)

    return true
end 

#############################################

function test_fwdtt_3Dsphere()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3Dsphere()
    
    #--------------------------------------
    ## Traveltime forward
    println("Traveltime 3D in spherical coordinates")
    ttpicks = traveltime3Dsphere(velmod,grd,coordsrc,coordrec)
    
    return true
end 

#############################################

function test_gradtt_2Dsphere()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2Dsphere()

    #--------------------------------------
    # Gradient of misfit
    stdobs = 0.15.*ones(size(coordrec,1),size(coordsrc,1))
    ttpicks = traveltime2Dsphere(velmod,grd,coordsrc,coordrec)
    noise = stdobs.^2 .* randn(size(ttpicks))
    dobs = ttpicks .+ noise
    flatmod = 2.8 .+ zeros(grd.nr,grd.nθ) 
    
    println("Gradient 2D in spherical coordinates")
    grad = gradttime2Dsphere(flatmod,grd,coordsrc,coordrec,dobs,stdobs)

    return true
end 

#############################################
    
function test_gradtt_3Dsphere()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3Dsphere()

    #--------------------------------------
    # Gradient of misfit
    stdobs = 0.15 .* ones(size(coordrec,1),size(coordsrc,1))
    ttpicks = traveltime3Dsphere(velmod,grd,coordsrc,coordrec)
    noise = stdobs.^2 .* randn(size(ttpicks))
    dobs = ttpicks .+ noise
    flatmod = 2.8 .+ zeros(grd.nr,grd.nθ,grd.nφ) 
    
    println("Gradient 3D using in spherical coordinates")
    grad = gradttime3Dsphere(flatmod,grd,coordsrc,coordrec,dobs,stdobs)
    
    return true
end 

#############################################

