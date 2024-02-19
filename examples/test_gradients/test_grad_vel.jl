
using Revise
using EikonalSolvers
using GLMakie
using HDF5



function runtest()
    
    # create the Grid2D struct
    hgrid = 40.0 #122.73
    grd = Grid2D(hgrid=hgrid,
                 xinit=0.0,
                 yinit=0.0,
                 nx=50,
                 ny=30) 
    # nsrc = 4
    # coordsrc = [hgrid.*LinRange(10.0,190.0,nsrc)   hgrid.*100.0.*ones(nsrc)] # coordinates of the sources (4 sources)

    nsrc = 1 
    nrec = 7
    # coordrec = [[hgrid.*LinRange(3.73,grd.nx-4.34,nrec)  hgrid.*3.7.*ones(nrec)] for i=1:nsrc]
    # coordrec = [vcat(coordrec...,
    #                  [[hgrid.*LinRange(2.73,grd.nx-3.34,nrec)  hgrid.*(grd.ny-4)*ones(nrec)] for i=1:nsrc]... )]
    # coordrec = [vcat(coordrec...,
    #                  [hgrid*(grd.nx÷2)+0.6*grd.hgrid/2  hgrid*(grd.ny÷2)+0.5*grd.hgrid/2])]
    coordrec = [[hgrid.*LinRange(3.73,grd.nx-4.34,nrec)  hgrid.*3.7.*ones(nrec);
                 hgrid*(grd.nx÷2)+0.6*grd.hgrid/2  hgrid*(grd.ny÷2)+0.5*grd.hgrid/2;
                 hgrid.*LinRange(2.73,grd.nx-3.34,nrec)  hgrid.*(grd.ny-4)*ones(nrec)]]
    
    #@show coordrec

    velmod = 2500.0 .* ones(grd.nx,grd.ny)                                # velocity model
    if true
        # ## increasing velocity with depth...
        for i=1:grd.ny
            velmod[:,i] = 23.6 * i .+ velmod[:,i]
        end
        # velmod = velmod[:,end:-1:1]
        # velmod .+= 1.0.*rand(size(velmod))
        # for i=1:grd.nx
        #     velmod[i,:] = 0.034 * i .+ velmod[i,:]
        # end
    else
        println("\n=== Constant velocity model ===\n")
    end

    gradtype = :gradvel #:gradvel  # :gradvelandsrcloc  # :gradvel
    parallelkind = :serial #:sharedmem


    misf_plus = similar(velmod)
    misf_less = similar(velmod)


    
    compare_adj_FD = true

    refinesrc = true

    extraparams=ExtraParams(parallelkind=parallelkind,
                            refinearoundsrc=refinesrc,
                            radiussmoothgradsrc = 0 )

    @show extraparams.grdrefpars.downscalefactor
    @show extraparams.grdrefpars.noderadius


    println("\n######## Refinement of the source region?  $refinesrc  ###########")

    println("\n  Smoothing around source: $(extraparams.radiussmoothgradsrc)")

    srcongrid = false
    if srcongrid
        println("  Source on a grid point")
        coordsrc1 = [hgrid*(grd.nx÷2)  hgrid*(grd.ny÷2)]
    else
        println("  Source NOT on a grid point")
        coordsrc1 = [hgrid*(grd.nx÷2)+0.99*grd.hgrid/2  hgrid*(grd.ny÷2)+0.99*grd.hgrid/2]
    end
    @show coordsrc1

    println("\n-------------- forward  ----------------")
    # run the traveltime computation with default algorithm ("ttFMM_hiord")
    ttpicks,tt_ref = traveltime2D(velmod,grd,coordsrc1,coordrec,
                                  returntt=true,extraparams=extraparams)

    srcpoints = h5read("srcpoints.h5","srcpts")
    @show srcpoints


    #     gradmi_ij = zeros(grd.nx,grd.ny)
    #     tt_analytic = zeros(grd.nx,grd.ny)

    #     for j=1:grd.ny
    #         for i=1:grd.nx

    #             dh = 0.001
    #             velpdh = copy(velmod)
    #             velpdh[i,j] += dh
    #             velmdh = copy(velmod)
    #             velmdh[i,j] -= dh

    #             tp1,tt_pdh = traveltime2D(velpdh,grd,coordsrc1,coordrec,
    #                                       returntt=true,extraparams=extraparams)

    #             tp2,tt_mdh = traveltime2D(velmdh,grd,coordsrc1,coordrec,
    #                                       returntt=true,extraparams=extraparams)

    #             ett = extrema(tt_pdh[:,:,1] .- tt_mdh[:,:,1])

                
    #             misf1 = 0.0
    #             misf2 = 0.0
    #             nsrc = size(coordsrc1,1)
    #             for s=1:nsrc
    #                 misf1 += sum( (tp1[s] ).^2 )
    #                 misf2 += sum( (tp2[s] ).^2 )
    #             end
    #             misf1 *= 0.5
    #             misf2 *= 0.5
    #             gradmi = (misf1-misf2)/2dh

    #             @show gradmi

    #             gradmi_ij[i,j] = gradmi

    #             if i==18 && j==12
    #                 global tt1_save = tt_pdh[:,:,1]
    #                 global tt2_save = tt_mdh[:,:,1]
    #             end

    #             xsrc = coordsrc1[1]
    #             ysrc = coordsrc1[2]
    #             dist = sqrt((grd.x[i]-xsrc)^2+(grd.y[j]-ysrc)^2)
    #             tt_analytic[i,j] = dist/velmod[i,j]

    #         end
    #     end

    #     @show extrema(tt_ref),extrema(tt_analytic)
    #     @show extrema(tt_ref[:,:,1] .- tt_analytic)


    #######################################################

    if true

        # standard deviation of error on observed data
        stdobs = [0.0001.*ones(size(ttpicks[1])) for i=1:nsrc]
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
        gradvel,∂χ∂xysrc = gradttime2D(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                       gradtype,extraparams=extraparams)

   
        if compare_adj_FD

            dh = 0.001
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

                    
                    misf_plus[i,j] = misf_pdh
                    misf_less[i,j] = misf_mdh

                end
            end
            println()
        end     

    end

    
    #######################################################
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

    #end


end # function
