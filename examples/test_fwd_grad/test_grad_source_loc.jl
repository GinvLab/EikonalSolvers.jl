
using Revise
using EikonalSolvers
using GLMakie



# create the Grid2D struct
hgrid = 180.73
grd = Grid2DCart(hgrid=hgrid,xinit=0.0,yinit=0.0,nx=380,ny=260) 
# nsrc = 4
# coordsrc = [hgrid.*LinRange(10.0,190.0,nsrc)   hgrid.*100.0.*ones(nsrc)] # coordinates of the sources (4 sources)

nsrc = 1
nrec = 5
#coordrec = [[hgrid.*LinRange(5.73,grd.nx-7.34,nrec)  hgrid.*5.7.*ones(nrec)] for i=1:nsrc] # coordinates of the receivers (10 receivers)

nrec = 5
coordrec = [[hgrid.*LinRange(5.73,grd.nx-7.34,nrec)  hgrid.*5.7.*ones(nrec)] for i=1:nsrc] # coordinates of the receivers (10 receivers)
#@show coordrec
coordrec = [vcat(coordrec...,
                [[hgrid.*LinRange(5.73,grd.nx-7.34,nrec)  hgrid.*(grd.ny-10)*ones(nrec)] for i=1:nsrc]... )]


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


for refinesrc in [false,true]

    extraparams=ExtraParams(parallelkind=parallelkind,
                            refinearoundsrc=refinesrc,
                            radiussmoothgradsrc=0)

    println("\n######## Refinement of the source region?  $refinesrc  ###########")

    #coordsrc = [hgrid*15.0 hgrid*(grd.ny-15.0)]
    #coordsrc = [hgrid*5.4 hgrid*(grd.ny-3.2)]
    coordsrc1 = [hgrid*37.8 hgrid*(grd.ny-28.4)]
    #coordsrc = [hgrid*5.39123 hgrid*(grd.ny-3.1123)]

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
    # nsrc = 1
    coordsrc2 = [hgrid*22.53 hgrid*(grd.ny-38.812)]
    #coordsrc = [hgrid*5.4 hgrid*(grd.ny-3.2)]


    println("\n-------------- gradient  ----------------")
    ## calculate the gradient of the misfit function
    gradvel,∂χ∂xysrc = eikgradient(vel0,grd,coordsrc2,coordrec,dobs,stdobs,
                                   gradtype,extraparams=extraparams)



    println("\n-------------- misfit  ----------------")

    dh = 0.00001

    coordsrc_plusdx = coordsrc2 .+ [dh 0.0]
    misf_pdx = eikttimemisfit(vel0,dobs,stdobs,coordsrc_plusdx,coordrec,grd;
                            extraparams=extraparams)

    coordsrc_minusdx = coordsrc2 .+ [-dh 0.0]
    misf_mdx = eikttimemisfit(vel0,dobs,stdobs,coordsrc_minusdx,coordrec,grd;
                            extraparams=extraparams)
    
    coordsrc_plusdy = coordsrc2 .+ [0.0 dh]
    misf_pdy = eikttimemisfit(vel0,dobs,stdobs,coordsrc_plusdy,coordrec,grd;
                            extraparams=extraparams)
    
    coordsrc_minusdy = coordsrc2 .+ [0.0 -dh]
    misf_mdy = eikttimemisfit(vel0,dobs,stdobs,coordsrc_minusdy,coordrec,grd;
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

    # println()
    # @show extrema(gradvel)

    ##
    if true
        fig = Figure(size=(1000,800))
        ax1 = Axis(fig[1,1],title="Refinement $refinesrc")
        vmax = maximum(abs.(gradvel))
        colorrange = (-vmax,vmax)
        hm = heatmap!(ax1,gradvel,colormap=:balance,colorrange=colorrange)
        Colorbar(fig[1,2], hm)
        ax1.yreversed = true
        
        #display(GLMakie.Screen(),fig)
        display(fig)
        #sleep(5)
    end

end


