
using Revise
using EikonalSolvers
using GLMakie
using HDF5


#################################################

function rungradvel2d()
    
    # create the Grid2D struct
    hgrid = 50.0 
    grd = Grid2DCart(hgrid=hgrid,
                     xinit=0.0,
                     yinit=0.0,
                     nx=50,
                     ny=30) 
   
    nrec = 7
    coordrec = [[hgrid.*LinRange(3.73,grd.nx-4.34,nrec)  hgrid.*3.7.*ones(nrec);
                 hgrid*(grd.nx÷2)+0.6*grd.hgrid/2  hgrid*(grd.ny÷2)+0.5*grd.hgrid/2;
                 hgrid.*LinRange(2.73,grd.nx-3.34,nrec)  hgrid.*(grd.ny-4)*ones(nrec)]]
    
    #@show coordrec

    velmod = 2500.0 .* ones(grd.nx,grd.ny)                    
    # ## increasing velocity with depth...
    for i=1:grd.ny
        velmod[:,i] = 23.6 * i .+ velmod[:,i]
    end

    srcongrid = false
    if srcongrid
        println("  Source on a grid point")
        coordsrc1 = [hgrid*(grd.nx÷2)  hgrid*(grd.ny÷2)]
    else
        println("  Source NOT on a grid point")
        coordsrc1 = [hgrid*(grd.nx÷2)+grd.hgrid/2  hgrid*(grd.ny÷2)+grd.hgrid]
    end
    @show coordsrc1

    compare_adj_FD = true
    refinesrc = [false,true]
    gradvel_all = Vector{Any}(undef,2)
    gradvel_FD_all = [similar(velmod), similar(velmod)]

    for col=1:2
    
        extraparams=ExtraParams(refinearoundsrc=refinesrc[col],
                                radiussmoothgradsrc = 0 )

        println("\n-------------- forward  ----------------")
        # run the traveltime computation with default algorithm ("ttFMM_hiord")
        ttpicks,tt_ref = eiktraveltime(velmod,grd,coordsrc1,coordrec,
                                       returntt=true,extraparams=extraparams)

        # standard deviation of error on observed data
        nsrc = size(coordsrc1,1)
        stdobs = [0.0001.*ones(size(ttpicks[1])) for i=1:nsrc]
        # generate a "noise" array to simulate real data
        #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
        # add the noise to the synthetic traveltime data
        dobs = ttpicks #.+ noise

        # # create a guess/"current" model
        vel0 = 2200.0 .* ones(grd.nx,grd.ny)
        ## increasing velocity with depth...
        for i=1:grd.ny
           vel0[:,i] = 32.5 * i .+ vel0[:,i]
        end

        coordsrc2 = copy(coordsrc1) 
        #coordsrc = [hgrid*5.4 hgrid*(grd.ny-3.2)]


        println("\n-------------- gradient  ----------------")
        ## calculate the gradient of the misfit function
        gradvel_all[col] = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                     :gradvel,extraparams=extraparams)

   

        ## Compare with brute-force finite differences calculation
        dh = 0.001
        @show dh
        gradvel_FD = similar(gradvel_all[col])
        for j=1:grd.ny
            for i=1:grd.nx            
                if i%10 == 0
                    print("\ri: $i, j: $j      ")
                end
                velpdh = copy(vel0)
                velpdh[i,j] += dh
                velmdh = copy(vel0)
                velmdh[i,j] -= dh
                misf_pdh = eikttimemisfit(velpdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                        extraparams=extraparams)
                misf_mdh = eikttimemisfit(velmdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                        extraparams=extraparams)
                gradvel_FD_all[col][i,j] = (misf_pdh-misf_mdh)/(2*dh)
            end
        end
        println()
       
        
    end

    
    #######################################################

    fig = Figure(size=(1200,1100))

    for col=1:2
        
        var1 = gradvel_all[col]
        var2 = gradvel_FD_all[col]
        var3 = gradvel_all[col].-gradvel_FD_all[col]
        refsrc = refinesrc[col]

                
        ax1 = Axis(fig[1,col][1,1],title="Refin. $refsrc, grad adj.")
        vmax = maximum(abs.(var1))
        colorrange = (-vmax,vmax)
        hm1 = heatmap!(ax1,grd.x,grd.y,var1,colormap=:seismic,
                       colorrange=colorrange)
        Colorbar(fig[1,col][1,2], hm1)
        ax1.yreversed = true
        scatter!(ax1,coordsrc1[1],coordsrc1[2],
                 marker=:cross,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="source")
        scatter!(ax1,coordrec[1][:,1],coordrec[1][:,2],
                 marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="receiver")
        axislegend(ax1)


        ax2 = Axis(fig[2,col][1,1],title="Refinement $refsrc, grad FD")
        vmax = maximum(abs.(var2))
        colorrange = (-vmax,vmax)
        hm2 = heatmap!(ax2,grd.x,grd.y,var2,colormap=:seismic,
                       colorrange=colorrange)
        Colorbar(fig[2,col][1,2], hm2)
        ax2.yreversed = true
        scatter!(ax2,coordsrc1[1],coordsrc1[2],
                 marker=:cross,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="source")
        scatter!(ax2,coordrec[1][:,1],coordrec[1][:,2],
                 marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="receiver")
        axislegend(ax2)

        
        vmax = maximum(abs.(var3))
        ax3 = Axis(fig[3,col][1,1],title="Refin. $refsrc, gradvel-grad_FD")
        colorrange = (-vmax,vmax)
        hm3 = heatmap!(ax3,grd.x,grd.y,var3,colormap=:seismic,
                       colorrange=colorrange)
        Colorbar(fig[3,col][1,2], hm3)
        ax3.yreversed = true
        scatter!(ax3,coordsrc1[1],coordsrc1[2],
                 marker=:cross,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="source")
        scatter!(ax3,coordrec[1][:,1],coordrec[1][:,2],
                 marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="receiver")
        axislegend(ax3)

        @show extrema(gradvel_all[col].-gradvel_FD_all[col])

    end

    save("testgrad.png",fig)
    #display(GLMakie.Screen(),fig)
    display(fig)
    #sleep(5)

    return
end # function
