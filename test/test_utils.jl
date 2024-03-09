


# using Distributed
# using EikonalSolvers

# # using PyPlot

# ###################################
# ######################################################
# ##         Cartesian coordinates                    ##
# ######################################################

# function creategridmod2D()

#     hgrid = 125.0 
#     xinit = 0.0
#     yinit = 0.0
#     nx = 100 
#     ny = 70 
#     grd = Grid2DCart(hgrid=hgrid,xinit=xinit,yinit=yinit,nx=nx,ny=ny)
        
#     xmax = xinit + (nx-1)*hgrid
#     ymax = yinit + (ny-1)*hgrid
#     println("Grid size (velocity): $nx x $ny from ($xinit,$yinit) to  ($xmax,$ymax) ")
#     println("Model parameters: $(nx*ny)")
    
#     ######################################
#     nsrc = 3
#     coordsrc = [hgrid*LinRange(5.545,nx-14.75,nsrc)  (ny*hgrid-5.6).+(hgrid*LinRange(-1.0,1.0,nsrc))]

#     nrec = 10
#     coordrec = [[hgrid*LinRange(3.0,nx-2,nrec)  1.75*hgrid*LinRange(2.0,2.0,nrec)] for i=1:nsrc]
       
#     ######################################
        
#     velmod = 2500.0 .+ zeros(nx,ny) 
#     for i=1:ny
#         velmod[:,i] = 10.0 * i .+ velmod[:,i]
#     end

#     return grd,coordsrc,coordrec,velmod
# end


# ###################################

# function creategridmod3D()

#     hgrid = 2.5 
#     xinit = 0.0
#     yinit = 0.0
#     zinit = 0.0
#     nx = 30 
#     ny = 30
#     nz = 20
#     grd = Grid3DCart(hgrid=hgrid,xinit=xinit,yinit=yinit,zinit=zinit,nx=nx,ny=ny,nz=nz)
        
#     xmax = xinit + (nx-1)*hgrid
#     ymax = yinit + (ny-1)*hgrid
#     zmax = zinit + (nz-1)*hgrid
#     println("Grid size (velocity): $nx x $ny x $nz from ($xinit,$yinit,$zinit) to  ($xmax,$ymax,$zmax) ")
#     println("Model parameters: $(nx*ny*nz)")
    
#     ######################################
#     nsrc = 3
#     coordsrc = [hgrid*LinRange(3.5,nx-15,nsrc)  hgrid*LinRange(3.5,ny-15,nsrc)  (nz*hgrid-6).+(hgrid*LinRange(-1.0,1.0,nsrc))]

#     nrec = 10
#     coordrec = [[hgrid*LinRange(2.0,nx-2,nrec)  1.75*hgrid*LinRange(2.0,2.0,nrec) 1.75*hgrid*LinRange(2.0,2.0,nrec)]  for i=1:nsrc]


#     ######################################
        
#     velmod = 2.5 .+ zeros(nx,ny,nz) 
#     for i=1:nz
#         velmod[:,:,i] = 0.01 * i .+ velmod[:,:,i]
#     end

#     return  grd,coordsrc,coordrec,velmod
# end

# #############################################
# #############################################

# function analyticalsollingrad2D(grd::Grid2DCart,xsrcpos::Float64,ysrcpos::Float64)
#     ##################################
#     ## linear gradient of velocity
     
#     ## source must be *on* the top surface
#     #@show ysrcpos
#     @assert (ysrcpos == 0.0)

#     # position of the grid nodes
#     xgridpos = grd.x #[(i-1)*grd.hgrid for i=1:grd.nx] .+ grd.xinit
#     ygridpos = grd.y #[(i-1)*grd.hgrid for i=1:grd.ny] .+ grd.yinit
#     # base velocity
#     vel0 = 2.0
#     # gradient of velociy
#     gr = 0.05
#     ## construct the 2D velocity model
#     velmod = zeros(grd.nx,grd.ny) 
#     for i=1:grd.nx
#         velmod[i,:] = vel0 .+ gr .* (ygridpos .- ysrcpos)
#     end

#     # https://pubs.geoscienceworld.org/books/book/1011/lessons-in-seismic-computing
#     # Lesson No. 41: Linear Distribution of Velocity—V. The Wave-Fronts 

#     ## Analytic solution
#     ansol2d  = zeros(grd.nx,grd.ny)
#     for j=1:grd.ny, i=1:grd.nx
#         x = xgridpos[i]-xsrcpos
#         h = ygridpos[j]-ysrcpos
#         ansol2d[i,j] = 1/gr * acosh( 1 + (gr^2*(x^2+h^2))/(2*vel0*velmod[i,j]) )
#     end
#     return ansol2d,velmod
# end

# ###################################
# ###################################






