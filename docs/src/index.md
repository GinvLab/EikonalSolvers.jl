

```@meta
EditURL = "https://gitlab.com/JuliaGeoph/EikonalSolvers.jl/-/tree/main/docs/src/"
Author = "Andrea Zunino"
```
# Contents

```@contents
Pages = ["index.md","publicapi.md","privatestuff.md"]
Depth = 2
```

# EikonalSolvers's documentation

A library to perform seismic traveltime computations by solving the eikonal equation in two (2D) and three dimensions (3D) with the possibility of computing the gradient of a misfit function with respect to the velocity model. The coordinate system can be either regular Cartesian or spherical. The forward algorithm is based on a fast marching (FMM) method (2nd order) with a refinement of the grid around the source location. The computation of the gradient relies on the discrete adjoint method.

For historical reasons, alternative solvers are available (poorly maintained), including a first order FMM method and the fast sweeping (FS) method for global updates with different kinds of local stencils. Additionally, a continuos adjoint method to calculate the gradient is also provided.

Both forward and gradient (adjoint) computations can be run in paralle using either Julia's distributed computing functions for distributed memory or threads for multicore processor. The parallelisation scheme is "by source", distributing calculations for different seismic sources to different processors.

This code is part of a larger project `HMCLab` ([^ZuninoGebraadFichtner2023]) targeting probabilistic geophysical inverse problems.

## Installation

To install the package simple enter into the package manager mode in Julia by typing "`]`" at the REPL prompt and then use `add`, i.e.,
```
(v1.8) pkg> add EikonalSolvers
```
The package will be automatically downloaded from the web and installed.

Alternatively, use the path where the directory of the package is located, be it remote (GitLab)
```
(v1.8) pkg> add https://gitlab.com/JuliaGeoph/EikonalSolvers.jl
```
or local
```
(v1.8) pkg> add /path/to/EikonalSolvers.jl
```


## Theoretical background

The eikonal equation in 3D is given by:
```math
| \nabla \tau |^2 = \left( \dfrac{\partial \tau}{\partial x} \right)^2 +
\left( \dfrac{\partial \tau}{\partial y} \right)^2  +
\left( \dfrac{\partial \tau}{\partial z} \right)^2 =
\dfrac{1}{v^2(x,y,z)}
```
where ``\tau`` is the travel time, ``x,y,z`` the spatial coordinates and ``v`` the velocity.

In the numerical solution to the eikonal equation, there are two major components, the global scheme, which defines the strategy to update to the traveltime on the grid, i.e., the fast marching method and the local scheme, providing the finite difference stencils.

The gradient computations are based on the adjoint state method (see below).

## Numerical implementation

The solver for the eikonal equation ([^ZuninoGebraadetal2023]) is a second-order fast marching method (FMM) ([^Sethian1996], [^SethianPopovici1999]) using traditional stencils (e.g., [^RawlinsonSambridge2004]), with an additional refinement of the grid around the source.

Regarding the gradient of the misfit function with respect to velocity, the misfit function (without considering the prior) is defined as
```math
S = \dfrac{1}{2} \sum_i \dfrac{\left( \mathbf{\tau}_i^{\rm{calc}}(\mathbf{v})-\mathbf{\tau}_i^{\rm{obs}} \right)^2}{\sigma_i^2} \, .
```
The gradient of the above functional with respect to the velocity model ``\dfrac{\partial S}{\partial \mathbf{v}}`` can be calculated efficiently using the _adjoint_ state method (e.g., [^LeungQian2006]). 

Gradient computations performed in this code using the __discrete adjoint state method__ , which takes into account the non-linearity of the forward problem - _no_ linearisation of the forward model and _no_ rays are employed. The computational cost is almost independent of the number of receivers and, because of the discretization, the result is a "diffuse" sensitivity around the theoretical rays (see an example in the following). 


## Exported functions / API

The following sets of functions are exported by `EikonalSolvers`. Units are arbitrary but must be *consistent*.

For two-dimensional (2D) problems, Cartesian coordinates (rectilinear grids) or spherical/polar coordinates (curvilinear grids):
* [`Grid2D`](@ref), a `struct` describing the geometry and size of the 2D Cartesian grid;
* [`Grid2DSphere`](@ref), a `struct` describing the geometry and size of the 2D spherical grid;
To compute traveltimes and gradients in 2D:
* [`traveltime2D`](@ref), which computes the traveltimes in a 2D model; 
* [`gradttime2D`](@ref), which computes the 2D gradient of the misfit function with respect to velocity.

For three-dimensional (3D) problems Cartesian coordinates (rectilinear grids):
* [`Grid3D`](@ref), a `struct` describing the geometry and size of the 3D  Cartesian grid;
* [`Grid2DSphere`](@ref), a `struct` describing the geometry and size of the 2D spherical grid;
To compute traveltimes and gradients in 3D:
* [`traveltime3D`](@ref), which computes the traveltimes in a 3D model; 
* [`gradttime3D`](@ref), which computes the 3D gradient of the misfit function with respect to velocity.

Misfit functional calculation:
* [`ttmisfitfunc`](@ref), computes the value (scalar) of the misfit functional for given observed traveltimes and velocity modl.

Moreover, a convenience module `HMCTraveltimes` (see [`EikonalSolvers.HMCtraveltimes`](@ref)) is provided to facilitate the use of `EikonalSolvers` within the framework of Hamiltonian Monte Carlo inversion (see e.g. [^ZuninoMosegaard2018]) by employing the package `HMCsampler`. 

Additional functions are provided to perform forward and inverse calculations using different methods than the defaults. These functions are there for historical reasons and are pooly maintained. Do not rely on them.
* [`traveltime2Dalt`](@ref) and [`traveltime3Dalt`](@ref) which compute traveltimes using either a fast marching method ([^Sethian1996], [^SethianPopovici1999]) using Podvin-Lecomte stencils ([^PodvinLecomte1991]) or a fast sweeping method (FSM) ([^LeungQian2006]) using Podvin-Lecomte stencils ([^PodvinLecomte1991]) ;
* [`gradttime2Dalt`](@ref) and [`gradttime3Dalt`](@ref), which computes the 3D gradient of the misfit function with respect to velocity using either FMM or FSM ([^LeungQian2006],[^Taillandieretal2009]) and different stencils depending on user choices.

[`ExtraParams`](@ref) also may play an important role in certain user cases.


## Parallelisation

Both forward and gradient computations are parallelised using Julia's distributed computating functions. The parallelisation is "by source", meaning that traveltimes (or gradients) for different sources are computed in parallel. Therefore if there are ``N`` input sources and ``P`` processors, each processor will perform computations for about ``N \over P`` sources. The number of processors corresponds to the number of "threads" or "workers" (`Threads.nthreads()` or `nworkers()`) available to Julia when the computations are run. To enable parallel calculations the field `parallelkind` in the [`ExtraParams`](@ref) `struct` must be set either to `:sharedmem` or `:distribmem"` and passed to the forward or inverse routines. Default is `:serial`.

To get more than one thread for shared memory computing, Julia can either be started with `julia -t <N>` where `N` is the desired number of threads. To get more than one processor for distributed computing, instead, Julia can either be started with `julia -p <N>` where `N` is the desired number of processors or using `addprocs(<N>)` _before_ loading the module. 


## Example of forward calculations

### Cartesian coordinates

As an illustration in the following it is shown how to calculate traveltimes at receivers in 2D in Cartesian coordinates. 
Let's start with a complete example:

```@example full
using EikonalSolvers
grd = Grid2D(hgrid=0.5,xinit=0.0,yinit=0.0,nx=300,ny=220)         # create the Grid2D struct
nsrc = 4
nrec = 10
coordsrc = [grd.hgrid.*LinRange(10.0,290.0,nsrc)  grd.hgrid.*200.0.*ones(nsrc)] # coordinates of the sources (4 sources)
coordrec = [ [grd.hgrid.*LinRange(8.0,200.0,nrec) grd.hgrid.*20.0.*ones(nrec)] for i=1:nsrc] # coordinates of the receivers (10 receivers)
velmod = 2.5 .* ones(grd.nx,grd.ny)                                # velocity model

# run the traveltime computation with default algorithm ("ttFMM_hiord")
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```
The output will be a vector of vectors with 4 elements (number of sources) each of which being a 10-element array (number of receivers):
```@example full
ttimepicks
```
Now let's analyse more in details the various components. 
First of we have to import the module and define the parameters of the grid using the struct `Grid2D`:
```@example parts
using EikonalSolvers
# hgrid: grid spacing
# xinit,yinit: grid origin coordinates
# nx,ny: grid size
grd = Grid2D(hgrid=0.5,xinit=0.0,yinit=0.0,nx=300,ny=220)  # create the Grid2D struct
```
Then define the coordinates of the sources, a two-column array (since we are in 2D) representing the \$x\$ and \$y\$ coordinates
```@example parts
nsrc = 4
coordsrc = [grd.hgrid.*LinRange(10.0,290.0,4)   grd.hgrid.*200.0.*ones(4)] 
```
and the receivers, a "vector of arrays", i.e., a vector where each element (one element per each source) is a two-column array (since we are in 2D) representing the \$x\$ and \$y\$ coordinates
```@example parts
nrec = 10
coordrec = [ [grd.hgrid.*LinRange(8.0,200.0,nrec) grd.hgrid.*20.0.*ones(nrec)] for i=1:nsrc] # coordinates of the receivers (10 receivers)
```
The velocity model is defined as a 2D array with size (`grd.nx` \$\times\$ `grd.ny`)
```@example parts
velmod = 2.5 .* ones(grd.nx,grd.ny) 
# increasing velocity with depth...
for i=1:grd.ny 
  velmod[:,i] = 0.034 * i .+ velmod[:,i] 
end
```
Finally, the traveltime at receivers is computed
```@example parts
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```
![velmodttpicks](./images/velmod-ttpicks.png)
```@example parts
ttimepicks
```
To additionally return the traveltime everywhere on the grid do
```@example parts
ttimepicks,ttimegrid = traveltime2D(velmod,grd,coordsrc,coordrec,returntt=true)
nothing # hide
```
![ttarrays](./images/ttime-arrays.png)
The resulting traveltime array on the grid is returned as a three-dimensional array, containing a set of two-dimensional arrays, one for each source.
```@example parts
size(ttimegrid)
```	

### Spherical coordinates

Here we show an example of traveltime calculation in spherical coordinates in 2D. The grid is defined in terms of the radius `r` and the angle `θ`, representing the *co*-latitude. In 3D there is the additional angle `φ`, representing the longitude.
Remark: for spherical coordinates some Greek symbols are used. You can render them in Julia as following:

Symbol | How to render
--- | --- 
Δ | \Delta^TAB 
θ | \theta^TAB
φ | \varphi^TAB

The function for traveltimes in spherical coordinates, analogously to the Cartesian case, is `traveltime2Dsphere()`.
The grid setup and forward computations are carried out as shown in the following script. 
```@example fullsph
using EikonalSolvers
grd = Grid2DSphere(Δr=2.0,Δθ=0.2,nr=40,nθ=70,rinit=500.0,θinit=10.0) # create the Grid2DSphere struct
coordsrc = [grd.rinit.+grd.Δr*3  grd.θinit.+grd.Δθ*(grd.nθ-5)] # coordinates of the sources (1 source)
coordrec = [[grd.rinit.+grd.Δr*(grd.nr-3) grd.θinit.+grd.Δθ*3] ] # coordinates of the receivers (1 receiver), vector of arrays
velmod = 2.5 .* ones(grd.nr,grd.nθ)                           # velocity model

# run the traveltime computation 
ttimepicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```
The following picture shows an example of computed traveltime and gradient in spherical coordinates in 2D.
![sph2dttgrad](./images/sph2dttgrad.png)


## Example of gradient calculation	

### Cartesian coordinates

Here a synthetic example of 2D gradient computations in Cartesian coordinates is illustrated. In reality, traveltime data are "measured" from recorded seismograms, however, here we first create some synthetic "observed" traveltimes using a synthetic velocity model. First the grid and velocity model are set up, then forward calculations are performed, as in the section above [Example of forward calculations](@ref).
```@example grad1
using EikonalSolvers
hgrid=0.5
grd = Grid2D(hgrid=hgrid,xinit=0.0,yinit=0.0,nx=300,ny=220)         # create the Grid2D struct
nsrc = 4
nrec = 10
coordsrc = [hgrid.*LinRange(10.0,290.0,nsrc)   hgrid.*200.0.*ones(nsrc)] # coordinates of the sources (4 sources)
coordrec = [[hgrid.*LinRange(8.0,200.0,nrec)  hgrid.*20.0.*ones(nrec)] for i=1:nsrc] # coordinates of the receivers (10 receivers)
velmod = 2.5 .* ones(grd.nx,grd.ny)                                # velocity model
# increasing velocity with depth...
for i=1:grd.ny 
  velmod[:,i] = 0.034 * i .+ velmod[:,i] 
end

# run the traveltime computation with default algorithm ("ttFMM_hiord")
ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)
nothing # hide
```	
Then the "observed" traveltime data are created by adding some Gaussian noise to the traveltimes computed above to simulate real measurements.
```@example grad1
# standard deviation of error on observed data
stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
# generate a "noise" array to simulate real data
noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
# add the noise to the synthetic traveltime data
dobs = ttpicks .+ noise
nothing # hide
```

Now we can finally compute the gradient of the misfit functional (see above) at a given point, i.e., the gradient is computed at a given velocity model: ``\dfrac{\partial S}{\partial \mathbf{v}} \Big|_{\mathbf{v}_0}``.
```@example grad1
# create a guess/"current" model 
vel0 = 2.3 .* ones(grd.nx,grd.ny) 
# increasing velocity with depth...
for i=1:grd.ny
   vel0[:,i] = 0.015 * i .+ vel0[:,i]
end
    
# calculate the gradient of the misfit function
grad = gradttime2D(vel0,grd,coordsrc,coordrec,dobs,stdobs)
nothing # hide
```	
![ttarrays](./images/gradient.png)
The calculated gradient is an array with the same shape than the velocity model.
```@example grad1
size(grad)
```

### Spherical coordinates

Here a synthetic example of 2D gradient computations in spherical coordinates is shown.
```@example grad1sph
using EikonalSolvers
grd = Grid2DSphere(Δr=2.0,Δθ=0.2,nr=40,nθ=70,rinit=500.0,θinit=10.0) # create the Grid2DSphere struct
coordsrc = [grd.rinit.+grd.Δr*3  grd.θinit.+grd.Δθ*(grd.nθ-5)] # coordinates of the sources (1 source)
nsrc = size(coordsrc,1)
coordrec = [[grd.rinit.+grd.Δr*(grd.nr-3) grd.θinit.+grd.Δθ*3] for i=1:nsrc] # coordinates of the receivers (1 receiver)

# velocity model
velmod = 2.5 .* ones(grd.nr,grd.nθ) 
# run the traveltime computation
ttpicks = traveltime2D(velmod,grd,coordsrc,coordrec)

# standard deviation of error on observed data
stdobs = [0.15.*ones(size(ttpicks[1])) for i=1:nsrc]
# generate a "noise" array to simulate real data
noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
# add the noise to the synthetic traveltime data
dobs = ttpicks .+ noise

# create a guess/"current" model 
vel0 = 3.0 .* ones(grd.nr,grd.nθ)

# calculate the gradient of the misfit function
grad = gradttime2D(vel0,grd,coordsrc,coordrec,dobs,stdobs)
nothing # hide
```	
An example of a (thresholded) sensitivity kernel and contouring of traveltimes in 3D, using spherical coordinates, is depicted in the following plot:
![grad3D](./images/examplegrad3Dcarsph.png)


# References

[^ZuninoGebraadetal2023]: Zunino A., Gebraad, L., Ghirotto, A. and Fichtner, A., (2023). HMCLab a framework for solving diverse geophysical inverse problems using the Hamiltonian Monte Carlo algorithm, submitted, JGR.

[^LeungQian2006]: Leung, S. and Qian, J. (2006). An adjoint state method for three-dimensional transmission traveltime tomography using first-arrivals. Communications in Mathematical Sciences, 4(1), 249-266.

[^PodvinLecomte1991]: Podvin, P. and Lecomte, I. (1991). Finite difference computation of traveltimes in very contrasted velocity models: a massively parallel approach and its associated tools. Geophysical Journal International, 105, 271-284.

[^RawlinsonSambridge2004]: Rawlinson, N. and Sambridge, M. (2004). Wave front evolution in strongly heterogeneous layered media using the fast marching method. Geophys. J. Int., 156(3), 631-647.

[^Sethian1996]: Sethian A. J. (1996). A fast marching level set method for monotonically advancing fronts. Proceedings of the National Academy of Sciences Feb 1996, 93 (4) 1591-1595. 

[^SethianPopovici1999]: Sethian A. J. and Popovici A. (1999). Three dimensional traveltimes computation using the Fast Marching Method. Geophysics. 64. 516-523. 

[^Taillandieretal2009]: Taillandier, C., Noble, M., Chauris, H. and Calandra, H. (2009). First arrival travel time tomography based on the adjoint state methods. Geophysics, 74(6), WCB57–WCB66.

[^ZuninoMosegaard2018]: Zunino A., Mosegaard K. (2018), Integrating Gradient Information with Probabilistic Traveltime Tomography Using the Hamiltonian Monte Carlo Algorithm, 80th EAGE Conference & Exhibition, Copenhagen.
