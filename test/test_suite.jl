


using Distributed
using EikonalSolvers


###################################

function creategridmod2D()

    hgrid = 2.5 
    xinit = 0.0
    yinit = 0.0
    nx = 100 
    ny = 70 
    grd = Grid2D(hgrid,xinit,yinit,nx,ny)
        
    xmax = xinit + (nx-1)*hgrid+xinit
    ymax = yinit + (ny-1)*hgrid+yinit
    println("Grid size (velocity): $nx x $ny from ($xinit,$yinit) to  ($xmax,$ymax) ")
    println("Model parameters: $(nx*ny)")
    
    ######################################
    nsrc = 3
    coordsrc = [hgrid*LinRange(5.5,nx-15,nsrc)  (ny*hgrid-5.6).+(hgrid*LinRange(-1.0,1.0,nsrc))]

    nrec = 10
    coordrec = [hgrid*LinRange(3.0,nx-2,nrec)  1.75*hgrid*LinRange(2.0,2.0,nrec)]
    
    ######################################
        
    velmod = 2.5 .+ ones(nx,ny) 
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
    grd = Grid3D(hgrid,xinit,yinit,zinit,nx,ny,nz)
        
    xmax = xinit + (nx-1)*hgrid+xinit
    ymax = yinit + (ny-1)*hgrid+yinit
    zmax = zinit + (nz-1)*hgrid+zinit
    println("Grid size (velocity): $nx x $ny x $nz from ($xinit,$yinit,$zinit) to  ($xmax,$ymax,$zmax) ")
    println("Model parameters: $(nx*ny*nz)")
    
    ######################################
    nsrc = 3
    coordsrc = [hgrid*LinRange(3.5,nx-15,nsrc)  hgrid*LinRange(3.5,ny-15,nsrc)  (nz*hgrid-6).+(hgrid*LinRange(-1.0,1.0,nsrc))]

    nrec = 10
    coordrec = [hgrid*LinRange(2.0,nx-2,nrec)  1.75*hgrid*LinRange(2.0,2.0,nrec) 1.75*hgrid*LinRange(2.0,2.0,nrec)]
    
    ######################################
        
    velmod = 2.5 .+ ones(nx,ny,nz) 
    for i=1:nz
        velmod[:,:,i] = 0.01 * i .+ velmod[:,:,i]
    end

    return  grd,coordsrc,coordrec,velmod
end

#############################################
#############################################

function test_fwdtt_2D()
    
    @show nprocs()
    @show nworkers()

    #--------------------------------------
    # Create a grid and a velocity model
     grd,coordsrc,coordrec,velmod = creategridmod2D()

    #--------------------------------------
    ## Traveltime forward
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    for ttalgo in ttalgos
        println("Traveltime 2D using $ttalgo")
        ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec,algo=ttalgo)
    end
    
    return true
end 

#############################################

function test_fwdtt_3D()
    
    @show nprocs()
    @show nworkers()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()
    
    #--------------------------------------
    ## Traveltime forward
    ttalgos = ["ttFS_podlec","ttFMM_podlec","ttFMM_hiord"]
    for ttalgo in ttalgos
        println("Traveltime 3D using $ttalgo")
        ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec,algo=ttalgo)
    end
    
    return true
end 

#############################################

function test_gradtt_2D()
    
    @show nprocs()
    @show nworkers()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2D()

    #--------------------------------------
    # Gradient of misfit
    stdobs = 0.15
    ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
    noise = stdobs^2*randn(size(ttpicks))
    dobs = ttpicks .+ noise
    flatmod = 2.8 .+ ones(grd.nx,grd.ny) 
    
    gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
    for gradalgo in gradalgos
        println("Gradient 2D using $gradalgo")
        grad = gradttime2D(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
    end

    return true
end 

#############################################
    
function test_gradtt_3D()
    
    @show nprocs()
    @show nworkers()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3D()

    #--------------------------------------
    # Gradient of misfit
    stdobs = 0.15
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)
    noise = stdobs^2*randn(size(ttpicks))
    dobs = ttpicks .+ noise
    flatmod = 2.8 .+ ones(grd.nx,grd.ny,grd.nz) 
    
    gradalgos = ["gradFS_podlec","gradFMM_podlec","gradFMM_hiord"]
    for gradalgo in gradalgos
        println("Gradien 3D using $ttalgo")
        @time  grad = gradttime2D(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo=gradalgo)
    end

    return true
end 

#############################################


