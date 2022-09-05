
#########################################################################

"""
$(TYPEDSIGNATURES)

Find closest node on a 2D grid to a given point in spherical/polar coordinates.
"""
function findclosestnode_sph(r::Float64,θ::Float64,rinit::Float64,θinit::Float64,
                             Δr::Float64,Δθ::Float64) 
    # xini ???
    # yini ???
    ir = floor((r-rinit)/Δr)
    iθ = floor((θ-θinit)/Δθ)

    rx = r-(ir*Δr+rinit)
    ry = θ-(iθ*Δθ+θinit)

    middler = Δr/2.0
    middleθ = Δθ/2.0

    if rx>=middler
        ir = ir+1
    end
    if ry>=middleθ  
        iθ = iθ+1
    end
    return Int(ir+1),Int(iθ+1) # julia
end


#####################################################################

"""
$(TYPEDSIGNATURES)

Find closest node on a 3D grid to a given point in spherical coordinates.
"""
function findclosestnode_sph(r::Float64,θ::Float64,φ::Float64,rinit::Float64,θinit::Float64,
                             φinit::Float64,Δr::Float64,Δθ::Float64,Δφ::Float64)
    # rini ???
    # θini ???
    ir = floor((r-rinit)/Δr)
    iθ = floor((θ-θinit)/Δθ)
    iφ = floor((φ-φinit)/Δφ)
    rr = r-(ir*Δr+rinit)
    rθ = θ-(iθ*Δθ+θinit)
    rφ = φ-(iφ*Δφ+φinit)
 
    middler = Δr/2.0
    middleθ = Δθ/2.0
    middleφ = Δφ/2.0

    if rr>=middler 
        ir = ir+1
    end
    if rθ>=middleθ
        iθ = iθ+1
    end
    if rφ>=middleφ
        iφ = iφ+1
    end
    return Int(ir+1),Int(iθ+1),Int(iφ+1) # julia
end

#############################################################
