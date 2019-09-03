


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

To install the package simple enter into the package manager mode in Julia by typing "`]`" at the REPL prompt and then use `add`, i.e.,
```
(v1.2) pkg> add EikonalSolvers
```
The package will be automatically downloaded from the web and installed.


## Theoretical background

The eikonal equation in 3D is given by:
```math
| \nabla \tau |^2 = \left( \dfrac{\partial \tau}{\partial x} \right)^2 +
\left( \dfrac{\partial \tau}{\partial y} \right)^2  +
\left( \dfrac{\partial \tau}{\partial z} \right)^2 =
\dfrac{1}{v^2(x,y,z)}
```
where ``\tau`` is the travel time, ``x,y,z`` the spatial coordinates and ```v``` the velocity.

In the numerical solution to the eikonal equation, there are two major components, the global scheme, which defines the strategy to update to the traveltime on the grid, i.e., fast sweeping or fast marching method and the local scheme, providing the finite difference stencils.

The gradient computations are based on the adjoint state method Leung ...

## Numerical implementation

Three different implementations of the solver for the eikonal equation and the related  are provided, explained in the following.
1. A second order fast marching method ([^Sethian1996], [^SethianPopovici1999]) using traditional stencils ([^RawlinsonSambridge2004]) referred to as "ttFMM\_hiord" in the code, with an additional refinement of the grid around the source.
2. A fast marching method ([^Sethian1996], [^SethianPopovici1999]) using Podvin-Lecomte stencils  ([^PodvinLecomte1991]), referred to as "ttFMM\_podlec" in the code. 
3. A fast sweeping method ([^LeungQian2006]) using Podvin-Lecomte stencils ([^PodvinLecomte1991]), referred to as "ttFS\_podlec" in the code. 
The default method is the first ("ttFMM\_hiord"), where an additional refinement of the grid around the source is performed, improving the accuracy of traveltimes in that region.

Regarding the gradient of the misfit function with respect to velocity, the misfit function (without considering the prior) is defined as
```math
S = \dfrac{1}{2} \sum_i \dfrac{\left( \mathbf{\tau}_i^{\rm{calc}}(\mathbf{v})-\mathbf{\tau}_i^{\rm{obs}} \right)^2}{\sigma_i^2} \, .
```
The gradient of the above functional with respect to the velocity model ``\dfrac{\partial S}{\partial \mathbf{v}}`` can be calculated efficiently using the _adjoint_ state method (e.g., [^LeungQian2006]). 
```math
\dfrac{\partial S}{\partial \mathbf{v}} = - \sum_j \dfrac{\lambda_j(\mathbf{x})}{\mathbf{v}^3} %+ {\rm grad}(\rho(\mathbf{v})) \, ,
```
where the adjoint state variable ``\lambda`` is computed by solving ([^Bretaudeauetal2014]):
```math
\nabla \cdot ( \lambda(\mathbf{x}) \, \nabla \tau (\mathbf{x}) ) = - \sum R^t \dfrac{\Delta \tau}{\sigma^2} \, , 
```
with boundary condition ``\mathbf{n} \cdot \nabla \tau = \sum_k \dfrac{\Delta \tau_k}{\sigma^2_k}``, with ``\mathbf{n}`` the unit outward vector of the surface of the model.
Gradient computations performed in this code using the adjoint state method take into the non-linearity of the forward problem - _no_ linearisation of the forward model and _no_ rays are employed. The result is are a more "diffuse" sensitivity around the theoretical ray (see an example in the following).

Analogously to the forward routines, the gradient functions provide three different implementations of the forward and adjoint calculations:
1. A second order fast marching method ([^Sethian1996], [^SethianPopovici1999]) using traditional stencils ([^RawlinsonSambridge2004]) for forward calculations and a higher order fast marching method ([^Sethian1996], [^SethianPopovici1999]) using traditional stencils (custom made) for adjoint calculations, referred to as "ttFMM\_hiord" in the code, with an additional refinement of the grid around the source.
2. A fast marching method ([^Sethian1996], [^SethianPopovici1999]) using Podvin-Lecomte stencils  ([^PodvinLecomte1991]) for forward caoculations and a fast marching method ([^Sethian1996], [^SethianPopovici1999]) using ([^LeungQian2006],[^Taillandieretal2009]) stencils for adjoint calculations, referred to as "ttFMM\_podlec" in the code. 
3. A fast sweeping method ([^LeungQian2006]) using Podvin-Lecomte stencils ([^PodvinLecomte1991]) for forward calculations and a fast sweeping method ([^LeungQian2006]) using  ([^LeungQian2006],[^Taillandieretal2009]) stencils for adjoint calculations, referred to as "ttFS\_podlec" in the code.


The following sets of functions are exported by `EikonalSolvers`.  
For two-dimensional (2D) problems:
* [`Grid2D`](@ref), a `struct` describing the geometry and size of the 2D grid;
* [`traveltime2D`](@ref), which computes the traveltimes in a 2D model; 
* [`gradttime2D`](@ref), which computes the gradient of the misfit function with respect to velocity in 2D.

For three-dimensional (3D) problems:
* [`Grid3D`](@ref), a `struct` describing the geometry and size of the 3D grid;
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
hgrid = 0.5                   # grid spacing
xinit,yinit = 0.0, 0.0        # grid origin coordinates
nx,ny = 300, 220              # grid size
grd = Grid2D(hgrid,xinit,yinit,nx,ny)                              # create the Grid2D struct
coordsrc = [hgrid.*LinRange(10.0,290.0,4)   hgrid.*200.0.*ones(4)] # coordinates of the sources (4 sources)
coordrec = [hgrid.*LinRange(8.0,200.0,10)  hgrid.*20.0.*ones(10)] # coordinates of the receivers (10 receivers)
velmod = 2.5 .* ones(grd.nx,grd.ny)                                # velocity model

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

hgrid = 0.5            # grid spacing
xinit,yinit = 0.0, 0.0 # grid origin coordinates
nx,ny = 300, 220       # grid size
grd = Grid2D(hgrid,xinit,yinit,nx,ny) # create the Grid2D struct
```
Then define the coordinates of the sources, a two-column array (since we are in 2D) representing the \$x\$ and \$y\$ coordinates
```@example parts
coordsrc = [hgrid.*LinRange(10.0,290.0,4)   hgrid.*200.0.*ones(4)] 
```
and the receivers, again a two-column array (since we are in 2D) representing the \$x\$ and \$y\$ coordinates
```@example parts
coordrec = [hgrid.*LinRange(8.0,200.0,10)  hgrid.*20.0.*ones(10)] 
```
The velocity model is defined as a 2D array with size (`grd.nx` \$\times\$ `grd.ny`)
```@example parts
velmod = 2.5 .* ones(grd.nx,grd.ny) 
# increasing velocity with depth...
for i=1:ny 
  velmod[:,i] = 0.034 * i .+ velmod[:,i] 
end
```
Finally, the traveltime at receivers is computed, where the default algorithm is used ("ttFMM\_hiord")
```@example parts
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```
![velmodttpicks](./images/velmod-ttpicks.png)
```@example parts
ttimepicks
```
To use a diffent algorithm and to additionally return the traveltime everywhere on the grid do
```@example parts
ttalgo = "ttFMM_podlec"
ttimepicks,ttimegrid = traveltime2D(velmod,grd,coordsrc,coordrec,algo=ttalgo,returntt=true)
nothing # hide
```
![ttarrays](./images/ttime-arrays.png)
The resulting traveltime array on the grid is returned as a three-dimensional array, containing a set of two-dimensional arrays, one for each source.
```@example parts
ttimegrid
```	


## Example of gradient calculation	

Here a synthetic example of 2D gradient computations is illustrated. In reality, traveltime data are "measured" from recorded seismograms, however, here we first create some synthetic "observed" traveltimes using a synthetic velocity model. First the grid and velocity model are set up, then forward calculations are performed, as in the section above [Example of forward calculations](@ref).
```@example grad1
using EikonalSolvers
hgrid = 0.5                   # grid spacing
xinit,yinit = 0.0, 0.0        # grid origin coordinates
nx,ny = 300, 220              # grid size
grd = Grid2D(hgrid,xinit,yinit,nx,ny)                              # create the Grid2D struct
coordsrc = [hgrid.*LinRange(10.0,290.0,4)   hgrid.*200.0.*ones(4)] # coordinates of the sources (4 sources)
coordrec = [hgrid.*LinRange(8.0,200.0,10)  hgrid.*20.0.*ones(10)] # coordinates of the receivers (10 receivers)
velmod = 2.5 .* ones(grd.nx,grd.ny)                                # velocity model
# increasing velocity with depth...
for i=1:ny 
  velmod[:,i] = 0.034 * i .+ velmod[:,i] 
end

# run the traveltime computation with default algorithm ("ttFMM_hiord")
ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```	
Then the "observed" traveltime data are created by adding some Gaussian noise to the traveltimes computed above to simulate real measurements.
```@example grad1
# standard deviation of error on observed data
stdobs = 0.15.*ones(size(coordrec,1))
# generate a "noise" array to simulate real data
noise = stdobs.^2 .* randn(size(ttpicks))
# add the noise to the synthetic traveltime data
dobs = ttpicks .+ noise
nothing # hide
```

Now we can finally compute the gradient of the misfit functional (see above) at a given point, i.e., the gradient is computed at a given velocity model: ``\dfrac{\partial S}{\partial \mathbf{v}} \Big|_{\mathbf{v}_0}``.
```@example grad1
# create a guess/"current" model 
vel0 = 2.8 .+ zeros(grd.nx,grd.ny) 
    
# calculate the gradient of the misfit function
grad = gradttime2D(vel0,grd,coordsrc,coordrec,dobs,stdobs)
nothing # hide
```	
![ttarrays](./images/gradient.png)
The calculated gradient is an array with the same shape than the velocity model.
```@example grad1
grad
```

# References

[^Bretaudeauetal2014]: Bretaudeau, F., Brossier, R., Virieux, J. and Métivier, L. [2014] First-arrival delayed tomography using 1st and 2nd order adjoint-state method. In: SEG Technical Program Expanded Abstracts 2014, Society of Exploration Geophysicists, 4757-4762.

[^LeungQian2006]: Leung, S. and Qian, J. (2006). An adjoint state method for three-dimensional transmission traveltime tomography using first-arrivals. Communications in Mathematical Sciences, 4(1), 249-266.

[^PodvinLecomte1991]: Podvin, P. and Lecomte, I. (1991). Finite difference computation of traveltimes in very contrasted velocity models: a massively parallel approach and its associated tools. Geophysical Journal International, 105, 271-284.

[^RawlinsonSambridge2004]: Rawlinson, N. and Sambridge, M. (2004). Wave front evolution in strongly heterogeneous layered media using the fast marching method. Geophys. J. Int., 156(3), 631-647.

[^Sethian1996]: Sethian A. J. (1996). A fast marching level set method for monotonically advancing fronts. Proceedings of the National Academy of Sciences Feb 1996, 93 (4) 1591-1595. 

[^SethianPopovici1999]: Sethian A. J. and Popovici A. (1999). Three dimensional traveltimes computation using the Fast Marching Method. Geophysics. 64. 516-523. 

[^Taillandieretal2009]: Taillandier, C., Noble, M., Chauris, H. and Calandra, H. (2009). First arrival travel time tomography based on the adjoint state methods. Geophysics, 74(6), WCB57–WCB66.
