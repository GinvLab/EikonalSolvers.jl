
#
# MIT License
# Copyright (c)  Andrea Zunino
# 

################################################################
################################################################

"""
HMCtraveltimes

A convenience module to facilitate the use of `EikonalSolvers` within the 
 framework of Hamiltonian Monte Carlo inversion by employing the package `HMCsampler`. 

# Exports

$(EXPORTS)
"""
module HMCtraveltimes

using DocStringExtensions
using EikonalSolvers

export EikonalProbVel,EikonalProbSrcLoc,EikonalProbVelAndSrcLoc


#################################################################

## create the problem type for traveltime tomography
"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
Base.@kwdef struct EikonalProbVel
    grd::Union{Grid2DCart,Grid3DCart}
    dobs::Vector{Vector{Float64}}
    stdobs::Vector{Vector{Float64}}
    coordsrc::Array{Float64,2}
    coordrec::Vector{Array{Float64,2}}
    whichgrad::Symbol
    logVel::Bool=false
    extraparams::Union{ExtraParams,Nothing}=nothing
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

## make the type callable
"""
$(TYPEDSIGNATURES)

"""
function (eikprob::EikonalProbVel)(inpvecvel::Vector{Float64},kind::Symbol)

    @assert eikprob.whichgrad==:gradvel

    if eikprob.logVel==true 
        vecvel = exp.(inpvecvel)
    else
        vecvel = inpvecvel
    end

    velnd = reshape(vecvel,eikprob.grd.grsize...)

    # if typeof(eikprob.grd)==Grid2DCart
    #     # reshape vector to 2D array
    #     velnd = reshape(vecvel,eikprob.grd.grsize[1],eikprob.grd.grsize[2])
    # elseif typeof(eikprob.grd)==Grid3DCart
    #     # reshape vector to 3D array
    #     velnd = reshape(vecvel,eikprob.grd.grsize[1],eikprob.grd.grsize[2],
    #                     eikprob.grd.grsize[3])
    # else
    #     error("Wrong type of eikprob.grd.")
    # end

    if kind==:nlogpdf
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        #println("HMCtraveltimes logpdf")
        misval = eikttimemisfit(velnd,eikprob.grd,
                                eikprob.coordsrc,
                                eikprob.coordrec,
                                eikprob.dobs,
                                eikprob.stdobs,
                                extraparams=eikprob.extraparams) 
        return misval
        

    elseif kind==:gradnlogpdf
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        grad,_ = eikgradient(velnd,eikprob.grd,eikprob.coordsrc,
                             eikprob.coordrec,eikprob.dobs,eikprob.stdobs,
                             eikprob.whichgrad,
                             extraparams=eikprob.extraparams)
        if eikprob.logVel==true
            # derivative of ln(vel)
            grad .= (1.0./velnd) .* grad
        end

        # flatten traveltime array
        vecgrad = vec(grad)
        # return flattened gradient
        return  vecgrad
        
    else
        error("eikprob::EikonalVelProb(): Wrong argument 'kind': $kind...")
    end
end

#################################################################

## create the problem type for traveltime tomography
"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
Base.@kwdef struct EikonalProbSrcLoc
    grd::Union{Grid2DCart,Grid3DCart}
    dobs::Vector{Vector{Float64}}
    stdobs::Vector{Vector{Float64}}
    coordrec::Vector{Array{Float64,2}}
    velmod::Array{Float64,2}
    whichgrad::Symbol
    extraparams::Union{ExtraParams,Nothing}=nothing
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

## make the type callable
"""
$(TYPEDSIGNATURES)

"""
function (eikprob::EikonalProbSrcLoc)(inpcoosrc::Vector{Float64},kind::Symbol)

    @assert eikprob.whichgrad==:gradsrcloc

    coosrcnd = reshape(inpcoosrc,:,ndims(eikprob.velmod))
   
    if kind==:nlogpdf
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        #println("HMCtraveltimes logpdf")
        misval = eikttimemisfit(eikprob.velmod,eikprob.grd,
                                coosrcnd,
                                eikprob.coordrec,
                                eikprob.dobs,
                                eikprob.stdobs,
                                extraparams=eikprob.extraparams) 
        ###############      
        # scal = 2.0
        # println("\n --- Scaling misfit by $scal ---\n")
        # misval *= scal
        ###############
        return misval
        

    elseif kind==:gradnlogpdf
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        gradsrcloc,mis = eikgradient(eikprob.velmod,eikprob.grd,coosrcnd,
                                     eikprob.coordrec,eikprob.dobs,eikprob.stdobs,
                                     eikprob.whichgrad,
                                     extraparams=eikprob.extraparams)

        #@show mis
        # flatten traveltime array
        vecgradsrcloc = vec(gradsrcloc)
        # return flattened gradient

        ##################
        # scal = 0.5 
        # println("Scaling gradient by $scal")
        # vecgradsrcloc *= scal
        ##################
        return vecgradsrcloc
        
    else
        error("eikprob::EikonalSrcLocProb(): Wrong argument 'kind': $kind...")
    end
end


#################################################################

"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
Base.@kwdef struct EikonalProbVelAndSrcLoc
    grd::Union{Grid2DCart,Grid3DCart}
    dobs::Vector{Vector{Float64}}
    stdobs::Vector{Vector{Float64}}
    coordrec::Vector{Array{Float64,2}}
    whichgrad::Symbol
    logVel::Bool=false
    extraparams::Union{ExtraParams,Nothing}=nothing
end


"""
$(TYPEDSIGNATURES)

"""
function (eikprob::EikonalProbVelAndSrcLoc)(inppars::Vector{Float64},kind::Symbol)

    @assert eikprob.whichgrad==:gradvelandsrcloc
    
    ngridpt = prod(eikprob.grd.grsize)
    inpvecvel = inppars[1:ngridpt]
    inpcoosrc = inppars[ngridpt+1:end]

    if eikprob.logVel==true 
        vecvel = exp.(inpvecvel)
    else
        vecvel = inpvecvel
    end
    # reshape velocity
    velnd = reshape(vecvel,eikprob.grd.grsize...)
    # reshape coordinates
    coosrcnd = reshape(inpcoosrc,:,ndims(velnd))

    #display(coosrcnd)

    if kind==:nlogpdf
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        #println("HMCtraveltimes logpdf")
        misval = eikttimemisfit(velnd,eikprob.grd,
                                coosrcnd,
                                eikprob.coordrec,
                                eikprob.dobs,
                                eikprob.stdobs,
                                extraparams=eikprob.extraparams) 

        return misval
        

    elseif kind==:gradnlogpdf
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        gradvel,gradsrcloc,_ = eikgradient(velnd,eikprob.grd,coosrcnd,
                                           eikprob.coordrec,eikprob.dobs,eikprob.stdobs,
                                           eikprob.whichgrad,
                                           extraparams=eikprob.extraparams)

        if eikprob.logVel==true
            # derivative of ln(vel)
            gradvel .= (1.0./velnd) .* gradvel
        end

        # flatten traveltime array
        vecgrad = [vec(gradvel); vec(gradsrcloc)]        
        # return flattened gradient
        return vecgrad
        
    else
        error("eikprob::EikonalVelAndSrcLocProb(): Wrong argument 'kind': $kind...")
    end
end


#################################
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
