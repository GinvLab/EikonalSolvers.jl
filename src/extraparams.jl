

##--------------------------------------------------------------
## allowfixsqarg: if true, try to brute-force fix problems with
##                negative sqarg.
##    Use is discouraged...
Base.@kwdef mutable struct ExtraParams
    ## brute-force fix negative sqarg
    allowfixsqarg::Bool
    ## refine grid around source?
    refinearoundsrc::Bool
end

##--------------------------
function setextraparams!(extrapars::ExtraParams ;
                         allowfixsqarg::Bool=false, refinearoundsrc::Bool=true)
    extrapars.allowfixsqarg = allowfixsqarg
    extrapars.refinearoundsrc = refinearoundsrc
    warningextrapar(extrapars)
    return 
end
##--------------------------
function warningextrapar(extrapars::ExtraParams)
    if extrapars.allowfixsqarg==true
        @warn("ExtraParams: allowfixsqarg==true, brute-force fixing of negative discriminant allowed.")
    end
    if extrapars.refinearoundsrc==false
        @warn("ExtraParams: refinearoundsrc==false, no grid refinement around source.")
    end
    return 
end

##--------------------------------------------------------------
