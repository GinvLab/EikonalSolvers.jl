
################################################################
################################################################

module HMCtraveltimes

using EikonalSolvers

export EikonalProb

#################################################################

## create the problem type for traveltime tomography
Base.@kwdef struct EikonalProb
    mstart::Vector{Float64} # required
    grd::Union{Grid2D,Grid3D}
    dobs::Array{Float64,2}
    stdobs::Array{Float64,2}
    coordsrc::Array{Float64,2}
    coordrec::Array{Float64,2}
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

## make the type callable
function (eikprob::EikonalProb)(vecvel::Vector{Float64},kind::String)

    if typeof(eikprob.grd)==Grid2D
        # reshape vector to 2D array
        velnd = reshape(vecvel,eikprob.grd.nx,eikprob.grd.ny)
    elseif typeof(eikprob.grd)==Grid3D
        # reshape vector to 3D array
        velnd = reshape(vecvel,eikprob.grd.nx,eikprob.grd.ny,eikprob.grd.nz)
    else
        error("typeof(eikprob.grd)== ?? ")
    end

    if kind=="logpdf"
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        #println("logpdf")
        misval = misfitfunc(velnd,eikprob.dobs,eikprob.stdobs,eikprob.coordsrc,
                            eikprob.coordrec,eikprob.grd) .+ nlogprior()
        return misval

    elseif kind=="gradlogpdf"
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        if typeof(eikprob.grd)==Grid2D
            grad = gradttime2D(velnd,eikprob.grd,eikprob.coordsrc,
                               eikprob.coordrec,eikprob.dobs,eikprob.stdobs)
        elseif typeof(eikprob.grd)==Grid3D
            grad = gradttime3D(velnd,eikprob.grd,eikprob.coordsrc,
                               eikprob.coordrec,eikprob.dobs,eikprob.stdobs)
        end
        # flatten traveltime array
        vecgrad = vec(grad) .+ gradnlogprior()
        # return flattened gradient
        return vecgrad
        
    else
        error("Wrong argument 'kind'...")
    end
end

#################################################################

## prior functional
function nlogprior(  )

    return 0.0
end

## prior functional
function gradnlogprior(  )

    return 0.0
end


#################################################################

end # module
#################################################################

####################################
#  A template
#
# struct MyProblem
#     field1::<type>
#     field2::<type>
#     ...
# end
#
# function (myprob::MyProblem)(m::Vector{Float64},kind::String)
#
#     if kind=="logpdf"
#
#         [...]
#         return logpdf  # must return a scalar Float64
#
#     elseif kind=="gradlogpdf"
#
#         [....]
#         return gradlogpdf  # must return an array Vector{Float64}
#
#     else
#         error("Wrong kind...")
#     end
# end
####################################################
