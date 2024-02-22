
using Revise
using EikonalSolvers
using GLMakie



# create the Grid2D struct
hgrid = 30.00
grd = Grid2DCart(hgrid=hgrid,xinit=0.0,yinit=0.0,nx=80,ny=60) 
# nsrc = 4
# coordsrc = [hgrid.*LinRange(10.0,190.0,nsrc)   hgrid.*100.0.*ones(nsrc)] # coordinates of the sources (4 sources)

nsrc = 1
nrec = 5
coordrec = [[hgrid.*LinRange(3.73,grd.nx-4.34,nrec)  hgrid.*3.7.*ones(nrec);
                 hgrid*(grd.nx÷2)+0.6*grd.hgrid/2  hgrid*(grd.ny÷2)+0.5*grd.hgrid/2;
                 hgrid.*LinRange(2.73,grd.nx-3.34,nrec)  hgrid.*(grd.ny-4)*ones(nrec)]]

@show coordrec

velmod = 2500.0 .* ones(grd.nx,grd.ny)                                # velocity model
if true
    # ## increasing velocity with depth...
    for i=1:grd.ny
        velmod[:,i] = 0.034 * i .+ velmod[:,i]
    end
    # velmod = velmod[:,end:-1:1]
    # velmod .+= 1.0.*rand(size(velmod))
    # for i=1:grd.nx
    #     velmod[i,:] = 0.034 * i .+ velmod[i,:]
    # end
else
    println("\n=== Constant velocity model ===\n")
end

gradtype = :gradvelandsrcloc #:gradvel  # :gradvelandsrcloc  # :gradvel
parallelkind = :serial #:sharedmem
compare_adj_FD = true



for refinesrc in [false,true]

    extraparams=ExtraParams(parallelkind=parallelkind,
                            refinearoundsrc=refinesrc,
                            radiussmoothgradsrc=0)

    println("\n######## Refinement of the source region?  $refinesrc  ###########")

    srcongrid = false
    if srcongrid
        println("  Source on a grid point")
        coordsrc1 = [hgrid*(grd.nx÷2)  hgrid*(grd.ny÷2)]
    else
        println("  Source NOT on a grid point")
        coordsrc1 = [hgrid*(grd.nx÷2)+0.99*grd.hgrid/2  hgrid*(grd.ny÷2)+0.99*grd.hgrid/2]
    end

    println("\n-------------- forward  ----------------")
    # run the traveltime computation with default algorithm ("ttFMM_hiord")
    ttpicks = eiktraveltime(velmod,grd,coordsrc1,coordrec,
                            extraparams=extraparams)


    # standard deviation of error on observed data
    stdobs = [0.05.*ones(size(ttpicks[1])) for i=1:nsrc]
    # generate a "noise" array to simulate real data
    #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    # add the noise to the synthetic traveltime data
    dobs = ttpicks #.+ noise


    # # create a guess/"current" model
    #vel0 = copy(velmod)
    vel0 = copy(velmod)
    @assert velmod==vel0
    # vel0 = 2.3 .* ones(grd.nx,grd.ny)
    # increasing velocity with depth...
    for i=1:grd.ny
       vel0[:,i] = 0.015 * i .+ vel0[:,i]
    end
    # nsrc = 1
    coordsrc2 = copy(coordsrc1) 
    #coordsrc2 = [hgrid*22.53 hgrid*(grd.ny-38.812)]



    println("\n-------------- gradient  ----------------")
    ## calculate the gradient of the misfit function
    gradvel,∂χ∂xysrc = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                   gradtype,extraparams=extraparams)



    println("\n-------------- misfit  ----------------")
    misf_ref = ttmisfitfunc(vel0,dobs,stdobs,coordsrc2,coordrec,grd;
                            extraparams=extraparams)

    dh = 0.0001

    coordsrc_plusdx = coordsrc2 .+ [dh 0.0]
    misf_pdx = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_plusdx,coordrec,grd;
                            extraparams=extraparams)

    coordsrc_minusdx = coordsrc2 .+ [-dh 0.0]
    misf_mdx = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_minusdx,coordrec,grd;
                            extraparams=extraparams)
    
    coordsrc_plusdy = coordsrc2 .+ [0.0 dh]
    misf_pdy = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_plusdy,coordrec,grd;
                            extraparams=extraparams)
    
    coordsrc_minusdy = coordsrc2 .+ [0.0 -dh]
    misf_mdy = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_minusdy,coordrec,grd;
                            extraparams=extraparams)


    ∂χ∂x_src_FD = (misf_pdx-misf_mdx)/(2*dh)
    ∂χ∂y_src_FD = (misf_pdy-misf_mdy)/(2*dh)
    ∂χ∂xysrc_FD = [∂χ∂x_src_FD ∂χ∂y_src_FD]

    # @show misf_ref
    # @show misf_pdx,misf_mdx
    # @show misf_pdy,misf_mdy

    if ∂χ∂xysrc!=nothing
        @show ∂χ∂xysrc
        @show ∂χ∂xysrc_FD
        @show ∂χ∂xysrc.-∂χ∂xysrc_FD

    else
        println("No ∂χ∂xysrc requested.")
    end



    ##########################################

    # standard deviation of error on observed data
    stdobs = [0.01.*ones(size(ttpicks[1])) for i=1:nsrc]
    # generate a "noise" array to simulate real data
    #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
    # add the noise to the synthetic traveltime data
    dobs = ttpicks #.+ noise


    # # create a guess/"current" model
    #vel0 = copy(velmod)
    #vel0 = copy(velmod)
    vel0 = 2200.0 .* ones(grd.nx,grd.ny)
    ## increasing velocity with depth...
    for i=1:grd.ny
        vel0[:,i] = 32.5 * i .+ vel0[:,i]
    end
    # nsrc = 1
    coordsrc2 = copy(coordsrc1) #[hgrid*22.53 hgrid*(grd.ny-38.812)]
    #coordsrc = [hgrid*5.4 hgrid*(grd.ny-3.2)]


    println("\n-------------- gradient  ----------------")
    ## calculate the gradient of the misfit function
    gradvel,∂χ∂xysrc = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                   gradtype,extraparams=extraparams)

    
    if compare_adj_FD

        dh = 0.00001
        @show dh
        gradvel_FD = similar(gradvel)
        for j=1:grd.ny
            for i=1:grd.nx            

                if i%10 == 0
                    print("\ri: $i, j: $j      ")
                end
                
                velpdh = copy(vel0)
                velpdh[i,j] += dh
                velmdh = copy(vel0)
                velmdh[i,j] -= dh
                
                misf_pdh = ttmisfitfunc(velpdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                        extraparams=extraparams)

                misf_mdh = ttmisfitfunc(velmdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                        extraparams=extraparams)

                gradvel_FD[i,j] = (misf_pdh-misf_mdh)/(2*dh)
            end
        end
        println()
    end     

    
    if true

        if compare_adj_FD
            fig = Figure(size=(800,1200))
        else
            fig = Figure(size=(1000,700))
        end

        var1 = gradvel #abs.(gradvel).>0.0
        ax1 = Axis(fig[1,1],title="Refinement $refinesrc, grad adj.")
        vmax = maximum(abs.(var1))
        colorrange = (-vmax,vmax)
        hm1 = heatmap!(ax1,grd.x,grd.y,var1,colormap=:seismic,
                       colorrange=colorrange)
        Colorbar(fig[1,2], hm1)
        ax1.yreversed = true
        scatter!(ax1,coordsrc1[1],coordsrc1[2],
                 marker=:cross,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="source")
        scatter!(ax1,coordrec[1][:,1],coordrec[1][:,2],
                 marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                 color=:orange,markersize=20,label="receiver")
        axislegend(ax1)
        # scatter!(ax1,srcpoints,
        #          marker=:circle,#strokecolor=:green,strokewidth=3,
        #          color=:green,markersize=10,label="source points")


        if compare_adj_FD
            var2 = gradvel_FD #abs.(gradvel_FD).>0.0
            ax2 = Axis(fig[2,1],title="Refinement $refinesrc, grad FD")
            vmax = maximum(abs.(var2))
            colorrange = (-vmax,vmax)
            hm2 = heatmap!(ax2,grd.x,grd.y,var2,colormap=:seismic,
                           colorrange=colorrange)
            Colorbar(fig[2,2], hm2)
            ax2.yreversed = true
            scatter!(ax2,coordsrc1[1],coordsrc1[2],
                     marker=:cross,#strokecolor=:green,strokewidth=3,
                     color=:orange,markersize=20,label="source")
            scatter!(ax2,coordrec[1][:,1],coordrec[1][:,2],
                     marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                     color=:orange,markersize=20,label="receiver")
            axislegend(ax2)
            # scatter!(ax2,srcpoints,
            #          marker=:circle,#strokecolor=:green,strokewidth=3,
            #          color=:green,markersize=10,label="soure points")

            var3 = gradvel.-gradvel_FD
            scalamp = 1e0
            vmax = scalamp*maximum(abs.(var3))
            ax3 = Axis(fig[3,1],title="Refinement $refinesrc, gradvel - grad_FD, clipped at $scalamp * max val")
            colorrange = (-vmax,vmax)
            hm3 = heatmap!(ax3,grd.x,grd.y,var3,colormap=:seismic,
                           colorrange=colorrange)
            Colorbar(fig[3,2], hm3)
            ax3.yreversed = true
            scatter!(ax3,coordsrc1[1],coordsrc1[2],
                     marker=:cross,#strokecolor=:green,strokewidth=3,
                     color=:orange,markersize=20,label="source")
            scatter!(ax3,coordrec[1][:,1],coordrec[1][:,2],
                     marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                     color=:orange,markersize=20,label="receiver")
            axislegend(ax3)
            # scatter!(ax3,srcpoints,
            #          marker=:circle,#strokecolor=:green,strokewidth=3,
            #          color=:green,markersize=10,label="source points")

            
            @show extrema(gradvel_FD)
            @show extrema(gradvel.-gradvel_FD)
        end
        save("testgrad.png",fig)
        #display(GLMakie.Screen(),fig)
        display(fig)
        #sleep(5)
    end



  

end


