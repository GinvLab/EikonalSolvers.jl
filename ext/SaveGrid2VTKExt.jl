
module SaveGrid2VTKExt

using EikonalSolvers
using WriteVTK
using DocStringExtensions

#export savemodelvtk

#######################################################

"""
$(TYPEDSIGNATURES)

Save a model to VTK file format for Grid2DCart grids.
"""
function EikonalSolvers.savemodelvtk(flname::String,grd::Grid2DCart,datadict::Dict; kind::String="points")

    if kind=="cells"

        nx = grd.nx
        ny = grd.ny
        h2 = grd.hgrid/2.0

        # cell vertices N+1 long
        x = [grd.x.-h2..., grd.x[end].+h2]
        y = [grd.y.-h2..., grd.y[end].+h2]
        
        ### create vtk file
        vtkfile = vtk_grid(flname, x,y)  # 2-D

        # add cell data
        for (key,arr) in datadict
            if ndims(arr)>2
                for i=1:size(arr,3)
                    vtk_cell_data(vtkfile, arr[:,:,i], string(key,"  #$i"))
                end
            else
                vtk_cell_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)


    elseif kind=="points"

        ### create vtk file
        vtkfile = vtk_grid(flname, grd.x,grd.y)  # 2-D

        # add point data
        for (key,arr) in datadict
            if ndims(arr)>2
                for i=1:size(arr,3)
                    vtk_point_data(vtkfile, arr[:,:,i], string(key,"  #$i"))
                end
            else
                vtk_point_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)

    else
        
        error("savevtk(): Wrong argument 'kind'.")   
    end
    return
end

#######################################################


"""
$(TYPEDSIGNATURES)

Save a model to VTK file format for Grid3DCart grids.
"""
function EikonalSolvers.savemodelvtk(flname::String,grd::Grid3DCart,datadict::Dict; kind::String="points")


    if kind=="cells"

        nx = grd.nx
        ny = grd.ny
        nz = grd.nz
        h2 = grd.hgrid/2.0

        # cell vertices N+1 long
        x = [grd.x.-h2..., grd.x[end].+h2]
        y = [grd.y.-h2..., grd.y[end].+h2]
        z = [grd.z.-h2..., grd.z[end].+h2]
        
        ### create vtk file
        vtkfile = vtk_grid(flname, x,y,z)  # 3-D

        # add cell data
        for (key,arr) in datadict
            if ndims(arr)>3
                for i=1:size(arr,4)
                    vtk_cell_data(vtkfile, arr[:,:,:,i], string(key,"  #$i"))
                end
            else
                vtk_cell_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)


    elseif kind=="points"

        ### create vtk file
        vtkfile = vtk_grid(flname, grd.x,grd.y,grd.z)  # 3-D

        # add point data
        for (key,arr) in datadict
            if ndims(arr)>3
                for i=1:size(arr,4)
                    vtk_point_data(vtkfile, arr[:,:,:,i], string(key,"  #$i"))
                end
            else
                vtk_point_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)

    else
        
        error("savevtk(): Wrong argument 'kind'.")   
    end
    return
end

#######################################################

"""
$(TYPEDSIGNATURES)

Save a model to VTK file format for Grid2DSphere grids.
"""
function EikonalSolvers.savemodelvtk(flname::String,grd::Grid2DSphere,datadict::Dict; kind::String="points")
    
    Δr = grd.Δr
    Δθ = grd.Δθ
    nr = grd.nr
    nθ = grd.nθ

    ######################################
    ### VTK file

    if kind=="cells"

        # transform to Cartesian coord for cell vertices
        xy = zeros(2,nr+1,nθ+1)

        iθ=1
        for ir=1:nr
            ## minus Δ
            xy[1,ir,iθ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) 
            xy[2,ir,iθ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
            if ir==nr # last point
                xy[1,ir+1,iθ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) 
                xy[2,ir+1,iθ] = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
            end
        end


        for iθ=1:nθ
            for ir=1:1
                ## minus Δ
                xy[1,ir,iθ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) 
                xy[2,ir,iθ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                if iθ==nθ # last point
                    xy[1,ir,iθ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) 
                    xy[2,ir,iθ+1] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                end
            end
        end
        
        for iθ=1:nθ
            for ir=1:nr            
                ## plus Δ
                x = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) 
                y = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                xy[:,ir+1,iθ+1] = (x,y)
            end
        end

        ### create vtk file
        vtkfile = vtk_grid(flname, xy)  # 2-D

        # add cell data
        for (key,arr) in datadict
            if ndims(arr)>2
                for i=1:size(arr,3)
                    vtk_cell_data(vtkfile, arr[:,:,i], string(key,"  #$i"))
                end
            else
                vtk_cell_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)

        
    elseif kind=="points"

        ## point grid
        xypts = zeros(2,nr,nθ)      
        for iθ=1:nθ
            for ir=1:nr            
                x = (grd.r[ir]) * sind(grd.θ[iθ]) 
                y = (grd.r[ir]) * cosd(grd.θ[iθ])
                xypts[:,ir,iθ] = (x,y)
            end
        end

        ### create vtk file
        vtkfile = vtk_grid(flname, xypts)  # 2-D

        # add point data
        for (key,arr) in datadict
            if ndims(arr)>2
                for i=1:size(arr,3)
                    vtk_point_data(vtkfile, arr[:,:,i], string(key,"  #$i"))
                end
            else
                vtk_point_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)
        
    end

    return
end

#######################################################


"""
$(TYPEDSIGNATURES)

Save a model to VTK file format for Grid3DSphere grids.
"""
function EikonalSolvers.savemodelvtk(flname::String,grd::Grid3DSphere,datadict::Dict; kind::String="points")
    
    Δr = grd.Δr
    Δθ = grd.Δθ
    Δφ = grd.Δφ 
    nr = grd.nr
    nθ = grd.nθ
    nφ = grd.nφ

    ######################################
    ### VTK file

    if kind=="cells"
        
        # transform to Cartesian coord for cell vertices
        xyz = zeros(3,nr+1,nθ+1,nφ+1)
        
        ## take care of r at "lower" boundary
        ir=1
        for iφ=1:nφ
            for iθ=1:nθ
                ## minus Δθ, "left" edge
                xyz[1,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                xyz[2,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                xyz[3,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                if iθ==nθ #&& iφ<nφ # last point, "right" edge
                    xyz[1,ir,iθ+1,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                    xyz[2,ir,iθ+1,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                    xyz[3,ir,iθ+1,iφ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                end
                if iφ==nφ #&& iθ<nθ # last point
                    xyz[1,ir,iθ,iφ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]+grd.Δφ/2) 
                    xyz[2,ir,iθ,iφ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]+grd.Δφ/2)
                    xyz[3,ir,iθ,iφ+1] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                end
                if iφ==nφ && iθ==nθ # corner point
                    xyz[1,ir,iθ+1,iφ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * cosd(grd.φ[iφ]+grd.Δφ/2) 
                    xyz[2,ir,iθ+1,iφ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * sind(grd.φ[iφ]+grd.Δφ/2)
                    xyz[3,ir,iθ+1,iφ+1] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                end
                
            end
        end

        ## take care of θ at "left" boundary
        iθ=1
        for iφ=1:nφ
            for ir=1:nr
                ## minus Δr, lower edge
                xyz[1,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                xyz[2,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                xyz[3,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                if ir==nr # last point, add top edge
                    xyz[1,ir+1,iθ,iφ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                    xyz[2,ir+1,iθ,iφ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                    xyz[3,ir+1,iθ,iφ] = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                end
                if iφ==nφ # last point, add edge
                    xyz[1,ir,iθ,iφ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]+grd.Δφ/2) 
                    xyz[2,ir,iθ,iφ+1] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]+grd.Δφ/2)
                    xyz[3,ir,iθ,iφ+1] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                end
                if iφ==nφ && ir==nr # corner point
                    xyz[1,ir+1,iθ,iφ+1] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]+grd.Δφ/2) 
                    xyz[2,ir+1,iθ,iφ+1] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]+grd.Δφ/2)
                    xyz[3,ir+1,iθ,iφ+1] = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                end
            end
        end

        ## take care of r at minimum φ boundary
        iφ=1
        for ir=1:nr
            for iθ=1:nθ
                ## minus Δφ,minimum φ edge
                xyz[1,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                xyz[2,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                xyz[3,ir,iθ,iφ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                if ir==nr # last point
                    xyz[1,ir+1,iθ,iφ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                    xyz[2,ir+1,iθ,iφ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]-grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                    xyz[3,ir+1,iθ,iφ] = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]-grd.Δθ/2)
                end
                if iθ==nθ # last point, max ϕ
                    xyz[1,ir,iθ+1,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                    xyz[2,ir,iθ+1,iφ] = (grd.r[ir]-grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                    xyz[3,ir,iθ+1,iφ] = (grd.r[ir]-grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                end
                if ir==nr && ir==nr # corner point
                    xyz[1,ir+1,iθ+1,iφ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * cosd(grd.φ[iφ]-grd.Δφ/2) 
                    xyz[2,ir+1,iθ+1,iφ] = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * sind(grd.φ[iφ]-grd.Δφ/2)
                    xyz[3,ir+1,iθ+1,iφ] = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                end
            end
        end

        ## fill all remaining cells
        for iφ=1:nφ
            for iθ=1:nθ
                for ir=1:nr            
                    ## plus Δ
                    x = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * cosd(grd.φ[iφ]+grd.Δφ/2) 
                    y = (grd.r[ir]+grd.Δr/2) * sind(grd.θ[iθ]+grd.Δθ/2) * sind(grd.φ[iφ]+grd.Δφ/2)
                    z = (grd.r[ir]+grd.Δr/2) * cosd(grd.θ[iθ]+grd.Δθ/2)
                    xyz[:,ir+1,iθ+1,iφ+1] = [x,y,z]
                end
            end
        end

        ### create vtk file
        vtkfile = vtk_grid(flname, xyz)  # 3-D

        # add cell data
        for (key,arr) in datadict
            if ndims(arr)>3
                for i=1:size(arr,4)
                    vtk_cell_data(vtkfile, arr[:,:,:,i], string(key,"  #$i"))
                end
            else
                vtk_cell_data(vtkfile, arr, key)
            end
        end

        # save the file
        outfiles = vtk_save(vtkfile)


    elseif kind=="points"

        xyzpts = zeros(3,nr,nθ,nφ)
        for iφ=1:nφ
            for iθ=1:nθ
                for ir=1:nr            
                    ## plus Δ
                    x = (grd.r[ir]) * sind(grd.θ[iθ]) * cosd(grd.φ[iφ]) 
                    y = (grd.r[ir]) * sind(grd.θ[iθ]) * sind(grd.φ[iφ])
                    z = (grd.r[ir]) * cosd(grd.θ[iθ])
                    xyzpts[:,ir,iθ,iφ] = [x,y,z]
                end
            end
        end

        # vtk file
        vtkfile = vtk_grid(flname, xyzpts)  # 3-D

        # add point data
        for (key,arr) in datadict
            if ndims(arr)>3
                for i=1:size(arr,4)
                    vtk_point_data(vtkfile, arr[:,:,:,i], string(key,"  #$i"))
                end
            else
                vtk_point_data(vtkfile, arr, key)
            end
        end
        # save data
        vtk_save(vtkfile)
   end

   return
end

###########################################################


end # module

