


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
    coordrec = [[LinRange(rinit+3.0,rmax-2,nrec)  1.75*Δθ*LinRange(2.0,2.0,nrec)] for i=1:nsrc]
    
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
    coordrec = [[LinRange(rinit+2.0,rmax-2,nrec)  θinit.+1.75*Δθ*LinRange(2.0,2.0,nrec) 1.75*Δφ*LinRange(2.0,2.0,nrec)] for i=1:nsrc]
    
    ######################################
        
    velmod = 2.7 .+ zeros(nr,nθ,nφ) 
    for i=1:nr
        velmod[i,:,:] = - 0.01 * i .+ velmod[i,:,:]
    end

    return  grd,coordsrc,coordrec,velmod
end

#############################################

function test_fwdtt_2Dspher_FMM2ndord()
    
    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2Dsphere()

    #--------------------------------------
    ## Traveltime forward
    println("Traveltime 2D in spherical coordinates")
    ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)

    return true
end 

#############################################

function test_fwdtt_3Dspher_FMM2ndord()
    
    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3Dsphere()

    #--------------------------------------
    ## Traveltime forward
    println("Traveltime 2D in spherical coordinates")
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

    return true
end

#############################################

function test_gradtt_2Dspher_FMM2ndord()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2Dsphere()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)

    nsrc = 3
    stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
    noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    dobs = ttpicks .+ noise

    flatmod = 2.8 .+ zeros(grd.nr,grd.nθ) 
    
    println("Gradient 2D in spherical coordinates")
    grad = gradttime2D(flatmod,grd,coordsrc,coordrec,dobs,stdobs)

    return true
end

#############################################

function test_gradtt_2Dspher_alternative()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod2Dsphere()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)

    nsrc = 3
    stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
    noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    dobs = ttpicks .+ noise

    flatmod = 2.8 .+ zeros(grd.nr,grd.nθ) 
    
    println("Gradient 2D in spherical coordinates")
    grad = gradttime2Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo="gradFMM_hiord")

    return true
end 

#############################################
    
function test_gradtt_3Dspher_FMM2ndord()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3Dsphere()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

    nsrc = 3
    stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
    noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    dobs = ttpicks .+ noise

    flatmod = 2.8 .+ zeros(grd.nr,grd.nθ,grd.nφ) 
    
    println("Gradient 3D using in spherical coordinates")
    grad = gradttime3D(flatmod,grd,coordsrc,coordrec,dobs,stdobs)
    
    return true
end

#############################################
    
function test_gradtt_3Dspher_alternative()

    #--------------------------------------
    # Create a grid and a velocity model
    grd,coordsrc,coordrec,velmod = creategridmod3Dsphere()

    #--------------------------------------
    # Gradient of misfit
    ttpicks = traveltime3D(velmod,grd,coordsrc,coordrec)

    nsrc = 3
    stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
    noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    dobs = ttpicks .+ noise

    flatmod = 2.8 .+ zeros(grd.nr,grd.nθ,grd.nφ) 
    
    println("Gradient 3D using in spherical coordinates")
    grad = gradttime3Dalt(flatmod,grd,coordsrc,coordrec,dobs,stdobs,gradttalgo="FMM_hiord")
    
    return true
end 

#############################################

