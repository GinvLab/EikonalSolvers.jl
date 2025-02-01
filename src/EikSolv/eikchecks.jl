


function checksrcrecposition(grd::Grid2DCart,coordsrc,coordrec)
    @assert size(coordsrc,1)==length(coordrec) 
    @assert all(grd.x[1].<=coordsrc[:,1].<=grd.x[end]) 
    @assert all(grd.y[1].<=coordsrc[:,2].<=grd.y[end])
    for r=1:length(coordrec)
        @assert all(grd.x[1].<=coordrec[r][:,1].<=grd.x[end])
        @assert all(grd.y[1].<=coordrec[r][:,2].<=grd.y[end])
    end
    return 
end

function checksrcrecposition(grd::Grid3DCart,coordsrc,coordrec)
    @assert size(coordsrc,1)==length(coordrec)
    @assert all(grd.x[1].<=coordsrc[:,1].<=grd.x[end])
    @assert all(grd.y[1].<=coordsrc[:,2].<=grd.y[end])
    @assert all(grd.z[1].<=coordsrc[:,3].<=grd.z[end])
    for r=1:length(coordrec)
        @assert all(grd.x[1].<=coordrec[r][:,1].<=grd.x[end])
        @assert all(grd.y[1].<=coordrec[r][:,2].<=grd.y[end])
        @assert all(grd.z[1].<=coordrec[r][:,3].<=grd.z[end])
    end
    return 
end

function checksrcrecposition(grd::Grid2DSphere,coordsrc,coordrec)
    @assert size(coordsrc,1)==length(coordrec)
    @assert all(grd.r[1].<=coordsrc[:,1].<=grd.r[end])
    @assert all(grd.θ[1].<=coordsrc[:,2].<=grd.θ[end])
    for r=1:length(coordrec)
        @assert all(grd.r[1].<=coordrec[r][:,1].<=grd.r[end])
        @assert all(grd.θ[1].<=coordrec[r][:,2].<=grd.θ[end])
    end
    return 
end

function checksrcrecposition(grd::Grid3DSphere,coordsrc,coordrec)
    @assert size(coordsrc,1)==length(coordrec)
    @assert all(grd.r[1].<=coordsrc[:,1].<=grd.r[end])
    @assert all(grd.θ[1].<=coordsrc[:,2].<=grd.θ[end])
    @assert all(grd.φ[1].<=coordsrc[:,3].<=grd.φ[end])
    for r=1:length(coordrec)
        @assert all(grd.r[1].<=coordrec[r][:,1].<=grd.r[end])
        @assert all(grd.θ[1].<=coordrec[r][:,2].<=grd.θ[end])
        @assert all(grd.φ[1].<=coordrec[r][:,3].<=grd.φ[end])
    end
    return 
end
