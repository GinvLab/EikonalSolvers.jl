
using Revise
using EikonalSolvers
using GLMakie


function runexample()
    
    # create the Grid2D struct
    hgrid = 30.00
    grd = Grid3DCart(hgrid=hgrid,xinit=0.0,yinit=0.0,zinit=0.0,
                     nx=130,ny=141,nz=82) 
    # nsrc = 4
    # coordsrc = [hgrid.*LinRange(10.0,190.0,nsrc)   hgrid.*100.0.*ones(nsrc)] # coordinates of the sources (4 sources)

    #nsrc = 1
    nrecaxis = 5
    xrec = LinRange(grd.x[1],grd.x[end],nrecaxis)
    yrec = LinRange(grd.y[1],grd.y[end],nrecaxis)
    zrec = 1.5 * grd.z[1]

    coordrec = [zeros(nrecaxis*nrecaxis,3)]
    l=0
    for j=1:nrecaxis, i=1:nrecaxis
        l+=1
        coordrec[1][l,:] .= (xrec[i],yrec[j],zrec)
    end

    
    # coordrec = [[hgrid.*LinRange(3.73,grd.nx-4.34,nrec) hgrid.*LinRange(3.73,grd.ny-4.34,nrec)  hgrid.*3.7.*ones(nrec);
    #               hgrid.*LinRange(2.73,grd.nx-3.34,nrec) hgrid.*LinRange(2.73,grd.ny-3.34,nrec) hgrid.*(grd.nz-4)*ones(nrec)]]

    #coordrec = [[hgrid.*1.4  hgrid*1.75 hgrid.*1.35]]

    @show coordrec

    velmod = 2500.0 .* ones(grd.nx,grd.ny,grd.nz)                                # velocity model
    if true
        # ## increasing velocity with depth...
        for i=1:grd.nz
            velmod[:,:,i] = 126.34 * i .+ velmod[:,:,i]
        end
    else
        println("\n=== Constant velocity model ===\n")
    end


    parallelkind = :serial #:sharedmem
    compare_adj_FD = false

    dogradwrtsrcloc = false


    gradvel = nothing
    

    
    for refinesrc in [false,true]

        extraparams=ExtraParams(parallelkind=parallelkind,
                                refinearoundsrc=refinesrc,
                                radiussmoothgradsrc=0)

        println("\n######## Refinement of the source region?  $refinesrc  ###########")

        srcongrid = false
        if srcongrid
            println("  Source on a grid point")
            coordsrc1 = [hgrid*(grd.nx-3) hgrid*(grd.ny-3) hgrid*(grd.nz-3)]
        else
            println("  Source NOT on a grid point")
            #coordsrc1 = [hgrid*(grd.nx÷2)+0.99*grd.hgrid/2  hgrid*(grd.ny÷2)+0.99*grd.hgrid/2 hgrid*(grd.nz÷2)+0.99*grd.hgrid/2]
            coordsrc1 = [hgrid*(grd.nx-1.75)  hgrid*(grd.ny-2.4) hgrid*(grd.nz-2.3)]
        end

        println("\n-------------- forward  ----------------")
        # run the traveltime computation with default algorithm ("ttFMM_hiord")
        ttpicks = eiktraveltime(velmod,grd,coordsrc1,coordrec,
                                extraparams=extraparams)



        # standard deviation of error on observed data
        nsrc = size(coordsrc1,1)
        stdobs = [0.0005.*ones(size(ttpicks[1])) for i=1:nsrc]
        # generate a "noise" array to simulate real data
        #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
        # add the noise to the synthetic traveltime data
        dobs = ttpicks #.+ noise


        
        if dogradwrtsrcloc
            
            # # create a guess/"current" model
            vel0 = copy(velmod)
            @assert velmod==vel0
            # nsrc = 1
            coordsrc2 = coordsrc1 .+ 14.234
            @show coordsrc2.-coordsrc1
            @show coordsrc1
            #coordsrc2 = [hgrid*22.53 hgrid*(grd.ny-38.812)]


            println("\n-------------- gradient w.r.t source location ----------------")
            ## calculate the gradient of the misfit function
            gradvel,∂χ∂xysrc = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                           :gradsrcloc,extraparams=extraparams)



            println("\n-------------- grad FD w.r.t source location  ----------------")
            misf_ref = ttmisfitfunc(vel0,dobs,stdobs,coordsrc2,coordrec,grd;
                                    extraparams=extraparams)

            dh = 0.0001

            coordsrc_plusdx = coordsrc2 .+ [dh 0.0 0.0]
            misf_pdx = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_plusdx,coordrec,grd;
                                    extraparams=extraparams)

            coordsrc_minusdx = coordsrc2 .+ [-dh 0.0 0.0]
            misf_mdx = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_minusdx,coordrec,grd;
                                    extraparams=extraparams)
            
            coordsrc_plusdy = coordsrc2 .+ [0.0 dh 0.0]
            misf_pdy = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_plusdy,coordrec,grd;
                                    extraparams=extraparams)
            
            coordsrc_minusdy = coordsrc2 .+ [0.0 -dh 0.0]
            misf_mdy = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_minusdy,coordrec,grd;
                                    extraparams=extraparams)


            coordsrc_plusdz = coordsrc2 .+ [0.0 0.0 dh]
            misf_pdz = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_plusdz,coordrec,grd;
                                    extraparams=extraparams)
            
            coordsrc_minusdz = coordsrc2 .+ [0.0 0.0 -dh]
            misf_mdz = ttmisfitfunc(vel0,dobs,stdobs,coordsrc_minusdz,coordrec,grd;
                                    extraparams=extraparams)
            

            ∂χ∂x_src_FD = (misf_pdx-misf_mdx)/(2*dh)
            ∂χ∂y_src_FD = (misf_pdy-misf_mdy)/(2*dh)
            ∂χ∂z_src_FD = (misf_pdz-misf_mdz)/(2*dh)
            ∂χ∂xysrc_FD = [∂χ∂x_src_FD ∂χ∂y_src_FD ∂χ∂z_src_FD]


            if ∂χ∂xysrc!=nothing
                @show ∂χ∂xysrc
                @show ∂χ∂xysrc_FD
                @show ∂χ∂xysrc.-∂χ∂xysrc_FD

            else
                println("No ∂χ∂xysrc requested.")
            end

        end

        ##########################################

        # standard deviation of error on observed data
        # stdobs = [0.01.*ones(size(ttpicks[1])) for i=1:nsrc]
        # generate a "noise" array to simulate real data
        #noise = [stdobs[i].^2 .* randn(size(stdobs[i])) for i=1:nsrc]
        # add the noise to the synthetic traveltime data
        #dobs = ttpicks #.+ noise


        # # create a guess/"current" model
        #vel0 = copy(velmod)
        #vel0 = copy(velmod)
        vel0 = 2200.0 .* ones(grd.nx,grd.ny,grd.nz)
        ## increasing velocity with depth...
        for i=1:grd.nz
            vel0[:,:,i] = 92.5 * i .+ vel0[:,:,i]
        end
        # nsrc = 1
        coordsrc2 = copy(coordsrc1)
        @assert coordsrc2==coordsrc1

        println("\n-------------- gradient w.r.t. velocity  ----------------")
        ## calculate the gradient of the misfit function
        gradvel,∂χ∂xysrc = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                       :gradvel,extraparams=extraparams)

        
        if compare_adj_FD

            dh = 0.0001
            @show dh
            gradvel_FD = similar(gradvel)
            for k=1:grd.nz
                for j=1:grd.ny
                    for i=1:grd.nx            

                        if i%10 == 0
                            print("\ri: $i, j: $j      ")
                        end
                        
                        velpdh = copy(vel0)
                        velpdh[i,j,k] += dh
                        velmdh = copy(vel0)
                        velmdh[i,j,k] -= dh
                        
                        misf_pdh = ttmisfitfunc(velpdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                                extraparams=extraparams)

                        misf_mdh = ttmisfitfunc(velmdh,dobs,stdobs,coordsrc2,coordrec,grd;
                                                extraparams=extraparams)

                        gradvel_FD[i,j,k] = (misf_pdh-misf_mdh)/(2*dh)
                    end
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

            jslice = 9

            var1 = gradvel[:,jslice,:] #abs.(gradvel).>0.0
            ax1 = Axis(fig[1,1],title="Refinement $refinesrc, grad adj.")
            vmax = maximum(abs.(var1))
            colorrange = (-vmax,vmax)
            hm1 = heatmap!(ax1,grd.x,grd.z,var1,colormap=:seismic,
                           colorrange=colorrange)
            Colorbar(fig[1,2], hm1)
            ax1.yreversed = true
            scatter!(ax1,coordsrc1[1],coordsrc1[3],
                     marker=:cross,#strokecolor=:green,strokewidth=3,
                     color=:orange,markersize=20,label="source")
            scatter!(ax1,coordrec[1][:,3],coordrec[1][:,3],
                     marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                     color=:orange,markersize=20,label="receiver")
            axislegend(ax1)
    

            if compare_adj_FD
                var2 = gradvel_FD[:,jslice,:] #abs.(gradvel_FD).>0.0
                ax2 = Axis(fig[2,1],title="Refinement $refinesrc, grad FD")
                vmax = maximum(abs.(var2))
                colorrange = (-vmax,vmax)
                hm2 = heatmap!(ax2,grd.x,grd.z,var2,colormap=:seismic,
                               colorrange=colorrange)
                Colorbar(fig[2,2], hm2)
                ax2.yreversed = true
                scatter!(ax2,coordsrc1[1],coordsrc1[3],
                         marker=:cross,#strokecolor=:green,strokewidth=3,
                         color=:orange,markersize=20,label="source")
                scatter!(ax2,coordrec[1][:,1],coordrec[1][:,3],
                         marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                         color=:orange,markersize=20,label="receiver")
                axislegend(ax2)

                
                var3 = gradvel[:,jslice,:].-gradvel_FD[:,jslice,:]
                scalamp = 1e0
                vmax = scalamp*maximum(abs.(var3))
                ax3 = Axis(fig[3,1],title="Refinement $refinesrc, gradvel - grad_FD, clipped at $scalamp * max val")
                colorrange = (-vmax,vmax)
                hm3 = heatmap!(ax3,grd.x,grd.z,var3,colormap=:seismic,
                               colorrange=colorrange)
                Colorbar(fig[3,2], hm3)
                ax3.yreversed = true
                scatter!(ax3,coordsrc1[1],coordsrc1[3],
                         marker=:cross,#strokecolor=:green,strokewidth=3,
                         color=:orange,markersize=20,label="source")
                scatter!(ax3,coordrec[1][:,1],coordrec[1][:,3],
                         marker=:dtriangle,#strokecolor=:green,strokewidth=3,
                         color=:orange,markersize=20,label="receiver")
                axislegend(ax3)
             
                @show extrema(gradvel)
                @show extrema(gradvel_FD)
                @show extrema(gradvel.-gradvel_FD)
            end
            #save("testgrad.png",fig)
            #display(GLMakie.Screen(),fig)
            display(fig)
            #sleep(5)
        end
        

    end

    return gradvel,grd
end


    
