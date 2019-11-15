# EikonalSolvers.jl

[![Build Status](https://travis-ci.com/inverseproblem/EikonalSolvers.jl.svg?branch=master)](https://travis-ci.com/inverseproblem/EikonalSolvers.jl)
[![CodeCov](https://codecov.io/gh/inverseproblem/EikonalSolvers.jl/branch/master/graph/badge.svg)](https://codecov.io/inverseproblem/EikonalSolvers.jl)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://inverseproblem.github.io/EikonalSolvers.jl/stable)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://inverseproblem.github.io/EikonalSolvers.jl/latest)

Docs: <https://inverseproblem.github.io/EikonalSolvers.jl/latest>

A library to perform __seismic traveltime__ computations by solving the eikonal equation in two (__2D__) and three dimensions (__3D__) with the possibility of computing the __gradient of a misfit function__ with respect to the velocity model.
Different eikonal solvers are available, based either on the fast marching (FMM) method (1st and 2nd order) or the fast sweeping (FS) method for global updates with different local stencils. 

Both forward and gradient (adjoint) computations are parallelised using Julia's distributed computing functions. The  parallelisation is "by source", distributing calculations for different seismic sources to different processors.


## Example of forward calculations in 2D

Here below an example of how to calculate traveltimes at receiver stations in 2D, given a grid geometry and positions of sources and receivers.
```julia
using EikonalSolvers
grd = Grid2D(hgrid=0.5,xinit=0.0,yinit=0.0,nx=300,ny=220)      # create the Grid2D struct
coordsrc = [LinRange(10.0,200.0,4)  LinRange(200.0,250.0,4)]   # coordinates of the sources (4 sources)
coordrec = [LinRange(10.0,200.0,10)  LinRange(200.0,250.0,10)] # coordinates of the receivers (10 receivers)
velmod = 2.5 .* ones(grd.nx,grd.ny)                            # velocity model

# run the traveltime computation with default algorithm ("ttFMM_hiord")
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
```

![velmodttpicks](docs/src/images/velmod-ttpicks.png)

Optionally it is possible to retrieve also the traveltime at all grid points.
```julia
ttimepicks,ttimegrid = traveltime2D(velmod,grd,coordsrc,coordrec,returntt=true)
```
![ttarrays](docs/src/images/ttime-arrays.png)


## Example of gradient calculations in 2D

The gradient of the misfit functional (see documentation) with respect to velocity can be calculated as following. A set of observed traveltimes, error on the measuraments and a reference velocity model are also required.
```julia
# calculate the gradient of the misfit function
grad = gradttime2D(vel0,grd,coordsrc,coordrec,dobs,stdobs)
```
![ttarrays](docs/src/images/gradient.png)

# Calculations in 3D 

Calculations in 3D for both Cartesian and spherical coordinates, are analogous to the 2D function, see the documentation at <https://inverseproblem.github.io/EikonalSolvers.jl/stable>.

<img src="docs/src/images/examplegrad3D.png" alt="Example gradient 3D" height="350"/>
<img src="docs/src/images/examplegrad3Dsphere.png" alt="Example gradient 3D in spherical coordinates" height="350"/>


