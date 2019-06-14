# EikonalSolvers.jl

A library to perform seismic traveltime computations by solving the eikonal equation in two and three dimensions with the possibility of computing the gradient of a misfit function with respect to the velocity model.
Different eikonal solvers are available, based either on the fast marching method (1st and 2nd order) or the fast sweeping method for global updates with different local stencils. 

Both forward and gradient (adjoint) computations are parallelised using Julia's distributed computing functions. Tje  parallelisation is "by source", distributing calculations for different seismic sources to different processors.
