# EikonalSolvers.jl

A library to perform *seismic* traveltime computations by solving the eikonal equation in two (2D) and three dimensions (3D) with the possibility of computing the gradient of a misfit function with respect to the velocity model.
Different eikonal solvers are available, based either on the fast marching (FMM) method (1st and 2nd order) or the fast sweeping (FS) method for global updates with different local stencils. 

Both forward and gradient (adjoint) computations are parallelised using Julia's distributed computing functions. The  parallelisation is "by source", distributing calculations for different seismic sources to different processors.


## Example of forward calculations in 2D

```
using EikonalSolvers
hgrid = 5.0                   # grid spacing
xinit,yinit = 0.0, 0.0        # grid origin coordinates
nx,ny = 300, 250              # grid size
grd = Grid2D(hgrid,xinit,yinit,nx,ny)                          # create the Grid2D struct
coordsrc = [LinRange(10.0,200.0,4)  LinRange(200.0,250.0,4)]   # coordinates of the sources (4 sources)
coordrec = [LinRange(10.0,200.0,10)  LinRange(200.0,250.0,10)] # coordinates of the receivers (10 receivers)
velmod = 2.5 .* ones(grd.nx,grd.ny)                            # velocity model

# run the traveltime computation with default algorithm ("ttFMM_hiord")
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
```
