


# Contents

```@contents
Pages = ["index.md","publicapi.md"]
Depth = 2
```

# EikonalSolvers's documentation

```@meta
Author = "Andrea Zunino"
```
A library to perform seismic traveltime computations by solving the eikonal equation in two and three dimensions with the possibility of computing the gradient of the misfit function (see below) with respect to the velocity model.  
Both forward and gradient computations are parallelised using Julia's distributed computing functions and the parallelisation is "by source", distributing calculations for different sources to different processors (see below).

## Installation

?????????????????????  
?????????????????????  
?????????????????????  
?????????????????????  

## Theoretical background

The eikonal equation is given by:

```math
\frac{n!}{k!(n - k)!} = \binom{n}{k} \, , 
```

In the numerical solution to the eikonal equation, there are two major components, the global scheme, which defines the strategy to update to the traveltime on the grid, i.e., fast sweeping or fast marching method and the local scheme, providing the finite difference stencils.

## Numerical implementation

Three different implementations of the solver for the eikonal equation and the related  are provided, explained in the following.
1. A second order fast marching method ([Osher](@ref)) using traditional stencils  ([Rawlison](@ref)) referred to as "ttFMM\_hiord" in the code, with an additional refinement of the grid around the source.
2. A fast marching method ([Osher](@ref)) using Podvin-Lecomte stencils  ([Podvin](@ref)), referred to as "ttFMM\_podlec" in the code. 
3. A fast sweeping method ([Leung](@ref)) using Podvin-Lecomte stencils ([Podvin](@ref)), referred to as "ttFS\_podlec" in the code. 
The default method is the first ("ttFMM\_hiord"), where an additional refinement of the grid around the source is performed, improving the accuracy of traveltimes in that region.

Regarding the gradient of the misfit function with respect to velocity, the misfit function is defined as
```math
\frac{n!}{k!(n - k)!} = \binom{n}{k} \, , 
```
and the gradient as
```math
\frac{n!}{k!(n - k)!} = \binom{n}{k} \, . 
```
The gradient computations are based on the _adjoint_ state method ([Leung](@ref)) and thus take into the non-linearity of the forward problem - _no_ linearisation of the forward model and _no_ rays are employed. The result is are a more "diffuse" sensitivity around the theoretical ray (see an example in the following).

Analogously to the forward routines, the gradient functions provide three different implementations of the forward and adjoint calculations:
1. A second order fast marching method ([Osher](@ref)) using traditional stencils ([Rawlison](@ref)) for forward calculations and a higher order fast marching method ([Osher](@ref)) using traditional stencils (custom made) for adjoint calculations, referred to as "ttFMM\_hiord" in the code, with an additional refinement of the grid around the source.
2. A fast marching method ([Osher](@ref)) using Podvin-Lecomte stencils  ([Podvin](@ref)) for forward caoculations and a fast marching method ([Osher](@ref)) using ([Leung](@ref),[Taillandier](@ref)) stencils for adjoint calculations, referred to as "ttFMM\_podlec" in the code. 
3. A fast sweeping method ([Leung](@ref)) using Podvin-Lecomte stencils ([Podvin](@ref)) for forward calculations and a fast sweeping method ([Leung](@ref)) using  ([Leung](@ref),[Taillandier](@ref)) stencils for adjoint calculations, referred to as "ttFS\_podlec" in the code.


The following sets of functions are exported by `EikonalSolvers`.  
For two-dimensional (2D) problems:
* [`Grid2D`](@ref), a struct describing the geometry and size of the 2D grid;
* [`traveltime2D`](@ref), which computes the traveltimes in a 2D model; 
* [`gradttime2D`](@ref), which computes the gradient of the misfit function with respect to velocity in 2D.

For three-dimensional (3D) problems:
* [`Grid3D`](@ref), a struct describing the geometry and size of the 3D grid;
* [`traveltime3D`](@ref), which computes the traveltimes in a 3D model; 
* [`gradttime3D`](@ref), which computes the gradient of the misfit function with respect to velocity in 3D.

Units are arbitrary but must be consistent.

## Parallelisation
Both forward and gradient computations are parallelised using Julia's distributed computating functions. The parallelisation is "by source", meaning that traveltimes (or gradients) for different sources are computed in parallel. Therefore if there are \$N\$ input sources and \$P\$ processors, each processor will perform computations for about \$N \over P\$ sources. The number of processors corresponds to the number of "workers" (`nworkers()`) available to Julia when the computations are run.  

To get more than one processor, Julia can either be started with `julia -p <N>` where `N` is the desired number of processors or using `addprocs(<N>)` _before_ loading the module.

## Example of forward calculations
As an illustration in the following it is shown how to calculate traveltimes at receivers in 2D. 
Let's start with a complete example:
```@example full
using EikonalSolvers
# grid spacing
hgrid = 5.0
# grid origin coordinates
xinit,yinit = 0.0, 0.0 
# grid size
nx,ny = 300, 250
# create the Grid2D struct
grd = Grid2D(hgrid,xinit,yinit,nx,ny) 
# coordinates of the sources (4 sources)
coordsrc = [LinRange(10.0,200.0,4)  LinRange(200.0,250.0,4)]
# coordinates of the receivers (10 receivers)
coordrec = [LinRange(10.0,200.0,10)  LinRange(200.0,250.0,10)]
# velocity model
velmod = 2.5 .* ones(grd.nx,grd.ny) 
# run the traveltime computation with default algorithm ("ttFMM_hiord")
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```
The output will be a 10\$\times\$4 array (number of receivers times number of sources):
```@example full
ttimepicks
```
Now let's analyse more in details the various components. 
First of we have to import the module and define the parameters of the grid using the struct `Grid2D`:
```@example parts
using EikonalSolvers

hgrid = 5.0            # grid spacing
xinit,yinit = 0.0, 0.0 # grid origin coordinates
nx,ny = 300, 250       # grid size
grd = Grid2D(hgrid,xinit,yinit,nx,ny) # create the Grid2D struct
```
Then define the coordinates of the sources, a two-column array (since we are in 2D) representing the \$x\$ and \$y\$ coordinates
```@example parts
coordsrc = [LinRange(10.0,200.0,4)  LinRange(200.0,250.0,4)]
```
and the receivers, again a two-column array (since we are in 2D) representing the \$x\$ and \$y\$ coordinates
```@example parts
coordrec = [LinRange(10.0,200.0,10)  LinRange(200.0,250.0,10)]
```
The velocity model is defined as a 2D array with size (`grd.nx` \$\times\$ `grd.ny`)
```@example parts
velmod = 2.5 .* ones(grd.nx,grd.ny) 
# increasing velocity with depth...
for i=1:nx 
  velmod[i,:] = 0.034 * i .+ velmod[i,:] 
end
```
Finally, the traveltime at receivers is computed, where the default algorithm is used ("ttFMM\_hiord")
```@example parts
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
ttimepicks
```
To use a diffent algorithm and to additionally return the traveltime everywhere on the grid do
```@example parts
ttalgo = "ttFMM_podlec"
ttimepicks,ttimegrid = traveltime2D(velmod,grd,coordsrc,coordrec,algo=ttalgo,returntt=true)
nothing # hide
```
The resulting traveltime array on the grid is returned as a three-dimensional array, containing a set of two-dimensional arrays, one for each source
```@example parts
ttimegrid
```	


## Example of gradient calculations	



