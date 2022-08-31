

##--------------------------------------------------------------
## allowfixsqarg: if true, try to brute-force fix problems with
##                negative sqarg.
##    Use is discouraged...
"""
$(TYPEDEF)

# Fields 

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct ExtraParams
    "brute-force fix negative sqarg"
    allowfixsqarg::Bool
    "refine grid around source?"
    refinearoundsrc::Bool
    "trigger GC manually at selected points"
    manualGCtrigger::Bool
end

##--------------------------
"""
$(TYPEDSIGNATURES)

Set some additional parameter for EikonalSolvers for testing in special conditions.

- `allowfixsqarg`: brute-force fix negative sqarg. Default `false`.
- `refinearoundsrc`: refine grid around source? Default `true`.
- `manualGCtrigger`: trigger explicitply the garbage collector at selected points? Default `true`.

"""
function setextraparams!(extrapars::ExtraParams ;
                         allowfixsqarg::Bool=false, refinearoundsrc::Bool=true, 
                         manualGCtrigger::Bool=true)
    extrapars.allowfixsqarg = allowfixsqarg
    extrapars.refinearoundsrc = refinearoundsrc
    extrapars.manualGCtrigger = manualGCtrigger
    warningextrapar(extrapars)
    return 
end
##--------------------------
"""
$(TYPEDSIGNATURES)

"""
function warningextrapar(extrapars::ExtraParams)
    if extrapars.allowfixsqarg==true
        @warn("ExtraParams: allowfixsqarg==true, brute-force fixing of negative discriminant allowed.")
    end
    if extrapars.refinearoundsrc==false
        @warn("ExtraParams: refinearoundsrc==false, no grid refinement around source.")
    end
    if extrapars.manualGCtrigger==false
        @warn("ExtraParams: manualGCtrigger==false, manual trigger for garbage collection disabled.")
    end
    return 
end

##--------------------------------------------------------------
