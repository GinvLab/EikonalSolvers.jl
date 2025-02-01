
using Revise
using EikonalSolvers
using GLMakie



function rungradsrcloc()
    # create the Grid2D struct
    hgrid = 10.0
    grd = Grid2DCart(hgrid=hgrid,cooinit=(0.0,0.0),grsize=(380,260))


    #nsrc = 3
    # nrec = 2
    # coordrec = [[hgrid.*LinRange(5.73,grd.grsize[1]-7.34,nrec)  hgrid.*5.7.*ones(nrec)] for i=1:nsrc] 
    #coordrec = [vcat(coordrec...,
    #                 [[hgrid.*LinRange(5.73,grd.grsize[1]-7.34,nrec)  hgrid.*(grd.grsize[2]-10)*ones(nrec)] for i=1:nsrc]... )]


    velmod = 2500.0 .* ones(grd.grsize...)  # velocity model
    if true
        # ## increasing velocity with depth...
        for i=1:grd.grsize[2]
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

    @show extrema(grd.x)
    @show extrema(grd.y)

    for refinesrc in [false,true]

        extraparams=ExtraParams(parallelkind=parallelkind,
                                refinearoundsrc=refinesrc,
                                radiussmoothgradsrc=0)

        println("\n######## Refinement of the source region?  $refinesrc  ###########")

        coordsrc1 = [#hgrid*127.8 hgrid*(grd.grsize[2]-28.4);
                     #hgrid*245.39123 hgrid*(grd.grsize[2]-193.1123);
                     hgrid*315.120 hgrid*(grd.grsize[2]-15.230)]

        nrec = 4
        coordrec = [[hgrid.*LinRange(5.73,grd.grsize[1]-7.34,nrec)  hgrid.*5.7.*ones(nrec)]
                    for i=1:size(coordsrc1,1)]
        #coordrec = [[hgrid.*grd.grsize[1]-300.34  hgrid.*5.7.*ones(1)] for i=1:size(coordsrc1,1)]

        println("\n-------------- forward  ----------------")
        # run the traveltime computation with default algorithm ("ttFMM_hiord")
        ttpicks = eiktraveltime(velmod,grd,coordsrc1,coordrec,
                                extraparams=extraparams)

        # standard deviation of error on observed data
        stdobs = [0.05.*ones(size(ttpicks[1])) for i=1:size(coordsrc1,1)]
        # generate a "noise" array to simulate real data
        #noise = [stdobs[i] .* randn(size(stdobs[i])) for i=1:nsrc]
        # add the noise to the synthetic traveltime data
        dobs = ttpicks #.+ noise


        # # create a guess/"current" model
        #vel0 = copy(velmod)
        vel0 = copy(velmod)
        @assert velmod==vel0
        # vel0 = 2.3 .* ones(grd.nx,grd.ny)
        # nsrc = 1
        coordsrc2 = [#hgrid*22.53 hgrid*(grd.grsize[2]-38.812);
                     #hgrid*260.4 hgrid*(grd.grsize[2]-123.2);
                     #hgrid*280.230 hgrid*(grd.grsize[2]-28.45);
                     hgrid*280.0 2320.23]
        @show coordsrc2
        
        println("\n-------------- gradient  ----------------")
        ## calculate the gradient of the misfit function
        gradvel,∂χ∂xysrc,misf = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                            gradtype,extraparams=extraparams)


        @show coordsrc1
        @show coordsrc2

        println("\n-------------- misfit  ----------------")

        dh = 0.0001
        @show dh

        ∂χ∂xysrc_FD = zeros(size(coordsrc2,1),2)
        for s=1:size(coordsrc2,1)

            @show s
            @show coordsrc2[s:s,:].-coordsrc1[s:s,:]
            @show dobs[s:s]
            @show stdobs[s:s]

            coordsrc_plusdx = coordsrc2[s:s,:] .+ [dh 0.0]
            misf_pdx = eikttimemisfit(vel0,grd,coordsrc_plusdx,coordrec[s:s],dobs[s:s],stdobs[s:s];
                                      extraparams=extraparams)

            coordsrc_minusdx = coordsrc2[s:s,:] .+ [-dh 0.0]
            misf_mdx = eikttimemisfit(vel0,grd,coordsrc_minusdx,coordrec[s:s],dobs[s:s],stdobs[s:s];
                                      extraparams=extraparams)
            
            coordsrc_plusdy = coordsrc2[s:s,:] .+ [0.0 dh]
            misf_pdy = eikttimemisfit(vel0,grd,coordsrc_plusdy,coordrec[s:s],dobs[s:s],stdobs[s:s];
                                      extraparams=extraparams)
            
            coordsrc_minusdy = coordsrc2[s:s,:] .+ [0.0 -dh]
            misf_mdy = eikttimemisfit(vel0,grd,coordsrc_minusdy,coordrec[s:s],dobs[s:s],stdobs[s:s];
                                      extraparams=extraparams)
            
            ∂χ∂x_src_FD = (misf_pdx-misf_mdx)/(2*dh)
            ∂χ∂y_src_FD = (misf_pdy-misf_mdy)/(2*dh)
            ∂χ∂xysrc_FD[s,:] .= (∂χ∂x_src_FD,∂χ∂y_src_FD)

            @show dh,grd.hgrid
            @show misf_pdx,misf_mdx
            @show misf_pdy,misf_mdy

        end
            
        # @show misf_ref
        # @show misf_pdx,misf_mdx
        # @show misf_pdy,misf_mdy


        println("∂χ∂xysrc:")
        display(∂χ∂xysrc)
        println("∂χ∂xysrc_FD:")
        display(∂χ∂xysrc_FD)
        println("∂χ∂xysrc.-∂χ∂xysrc_FD:")
        display(∂χ∂xysrc.-∂χ∂xysrc_FD)


        ##
        if true
            fig = Figure(size=(1000,800))
            ax1 = Axis(fig[1,1],title="Refinement $refinesrc")
            vmax = maximum(abs.(gradvel))
            colorrange = (-vmax,vmax)
            hm = heatmap!(ax1,grd.x,grd.y,gradvel,colormap=:balance,colorrange=colorrange)
            Colorbar(fig[1,2], hm)

            for i=1:size(coordsrc1,1)
                scatter!(ax1,coordsrc1[i,1],coordsrc1[i,2],color=:black)
                scatter!(ax1,coordsrc2[i,1],coordsrc2[i,2],color=:red)
            end

            #ax1.yreversed = true

            #display(GLMakie.Screen(),fig)
            display(fig)
            sleep(3)
        end

    end

    return 
end
