

########################################################

function interpolate_receiver(ttime::Array{Float64,3},grd::AbstractGridEik3D,xypt::Vector{Float64} )
    intval = trilinear_interp(ttime,grd,xypt)
    return intval                          
end

########################################################

"""
$(TYPEDSIGNATURES)

Trinilear interpolation.
"""
function trilinear_interp(ttime::AbstractArray{Float64,3},grd::AbstractGridEik3D,
                          xyzpt::AbstractVector{Float64}; return_coeffonly::Bool=false)
 
    xin,yin,zin = xyzpt[1],xyzpt[2],xyzpt[3]

    if typeof(grd)==Grid3DCart
        dx = grd.hgrid
        dy = grd.hgrid
        dz = grd.hgrid
        xinit = grd.xinit
        yinit = grd.yinit
        zinit = grd.zinit

    elseif typeof(grd)==Grid3DSphere
        dx = grd.Δr
        dy = grd.Δθ
        dz = grd.Δφ
        xinit = grd.rinit
        yinit = grd.θinit
        zinit = grd.φinit

    end
    
    xh = (xin-xinit)/dx
    yh = (yin-yinit)/dy
    zh = (zin-zinit)/dz
    i = floor(Int64,xh)
    j = floor(Int64,yh)
    k = floor(Int64,zh)
    x = xin-xinit  
    y = yin-yinit
    z = zin-zinit

    ## if at the edges of domain choose previous square...
    nx,ny,nz=size(ttime)

    if i==nx
        i=i-1
    end
    if j==ny
        j=j-1
    end
    if k==nz
        k=k-1
    end

    if ((i>xh) | (j>yh) | (k>zh)) 
        println("trilinear_interp(): Interpolation failed...")
        println("$i,$xh,$j,$yh,$k,$zh")
        return
    elseif ((i+1<xh)|(j+1<yh)|(k+1<zh))
        println("trilinear_interp(): Interpolation failed...")
        println("$i,$xh,$j,$yh,$k,$zh")
        return
    end 

    # @show i,j,k
    # @show x,y,z

    x0=i*dx
    y0=j*dy
    z0=k*dz
    x1=(i+1)*dx
    y1=(j+1)*dy
    z1=(k+1)*dz

    ## Fortran indices start from 1 while i,j,k from 0
    ii=i+1
    jj=j+1
    kk=k+1
    f000 = ttime[ii,jj,  kk] 
    f010 = ttime[ii,jj+1,kk]
    f001 = ttime[ii,jj,  kk+1]
    f011 = ttime[ii,jj+1,kk+1]

    f100 = ttime[ii+1,jj,  kk] 
    f110 = ttime[ii+1,jj+1,kk]
    f101 = ttime[ii+1,jj,  kk+1]
    f111 = ttime[ii+1,jj+1,kk+1]

    ## On a periodic and cubic lattice, let x_d, y_d, and z_d be the differences between each of x, y, z and the smaller coordinate related, that is:
    xd = (x - x0)/(x1 - x0)
    yd = (y - y0)/(y1 - y0)
    zd = (z - z0)/(z1 - z0)
    
    # ## where x_0 indicates the lattice point below x , and x_1 indicates the lattice point above x and similarly for y_0, y_1, z_0 and z_1.
    # ## First we interpolate along x (imagine we are pushing the front face of the cube to the back), giving:
    # c00 = f000 * (1 - xd) + f100 * xd 
    # c10 = f010 * (1 - xd) + f110 * xd 
    # c01 = f001 * (1 - xd) + f101 * xd 
    # c11 = f011 * (1 - xd) + f111 * xd 

    # ## Where V[x_0,y_0, z_0] means the function value of (x_0,y_0,z_0). Then we interpolate these values (along y, as we were pushing the top edge to the bottom), giving:
    # c0 = c00 * (1 - yd) + c10 *yd
    # c1 = c01 * (1 - yd) + c11 *yd

    # ## Finally we interpolate these values along z(walking through a line):
    # interpval = c0 * (1 - zd) + c1 * zd


    # ## Finally we interpolate these values along z(walking through a line):
    # interpval = c0 * (1 - zd) + c1 * zd 

    ##########################################################33

    if return_coeffonly

        coeff = @SVector [(1-xd)*(1-yd)*(1-zd) ,
                          xd*(1-yd)*(1-zd) ,
                          (1-xd)*yd*(1-zd) ,
                          (1-xd)*(1-yd)*zd ,
                          xd*(1-yd)*(1-zd) ,
                          (1-xd)*yd*zd ,
                          xd*yd*(1-zd) ,
                          xd*yd*zd ]
        
        ijs = @SMatrix [ii   jj    kk;
                        ii  jj+1   kk;
                        ii   jj    kk+1;
                        ii  jj+1   kk+1;
                        ii+1  jj    kk;
                        ii+1  jj+1  kk;
                        ii+1  jj    kk+1;
                        ii+1  jj+1  kk+1 ]

        return coeff,ijs

    else

        interpval = f000 * (1-xd)*(1-yd)*(1-zd) +
            f100 * xd*(1-yd)*(1-zd) +
            f010 * (1-xd)*yd*(1-zd) +
            f001 * (1-xd)*(1-yd)*zd +
            f101 * xd*(1-yd)*(1-zd) +
            f011 * (1-xd)*yd*zd +
            f110 * xd*yd*(1-zd) +
            f111 * xd*yd*zd

        return interpval
    end
    # @show ii,jj,kk
    # @show ttime[ii,jj,kk],interpval

    return interpval
end

#################################################

function createfinegrid(grd::AbstractGridEik3D,xyzsrc::AbstractVector{Float64},
                        vel::Array{Float64,3},
                        grdrefpars::GridRefinementPars)

    # if typeof(grd)==Grid2DCart
    #     simtype=:cartesian
    # elseif typeof(grd)==Grid2DSphere
    #     simtype=:spherical
    # end
    # # downscalefactor::Int = 0
    # # noderadius::Int = 0
    
    # downscalefactor = grdrefpars.downscalefactor # 3 #5
    # ## if noderadius is not the same for forward and adjoint,
    # ##   then troubles with comparisons with brute-force fin diff
    # ##   will occur...
    # noderadius = grdrefpars.noderadius #2 #2

    # ## find indices of closest node to source in the "big" array
    # ## ix, iy will become the center of the refined grid
    # if simtype==:cartesian
    #     n1_coarse,n2_coarse = grd.nx,grd.ny
    #     ixsrcglob,iysrcglob = findclosestnode(xysrc[1],xysrc[2],grd.xinit,grd.yinit,grd.hgrid)

    #     rx = xysrc[1]-grd.x[ixsrcglob]
    #     ry = xysrc[2]-grd.y[iysrcglob]

    #     ## four points
    #     ijsrcpts = @MMatrix zeros(Int64,4,2)
    #     halfg = 0.0
    #     if rx>=halfg
    #         srci = (ixsrcglob,ixsrcglob+1)
    #     else
    #         srci = (ixsrcglob-1,ixsrcglob)
    #     end
    #     if ry>=halfg
    #         srcj = (iysrcglob,iysrcglob+1)
    #     else
    #         srcj = (iysrcglob-1,iysrcglob)
    #     end
        
    #     l=1
    #     for j=1:2, i=1:2
    #         ijsrcpts[l,:] .= (srci[i],srcj[j])
    #         l+=1
    #     end
        
    # elseif simtype==:spherical
    #     n1_coarse,n2_coarse = grd.nr,grd.nθ
    #     ixsrcglob,iysrcglob = findclosestnode_sph(xysrc[1],xysrc[2],grd.rinit,grd.θinit,grd.Δr,grd.Δθ)
    #     error(" ttaroundsrc!(): spherical coordinates still work in progress...")
    # end
    
    # ##
    # ## Define chunck of coarse grid
    # ##
    # # i1coarsevirtual = ixsrcglob - noderadius
    # # i2coarsevirtual = ixsrcglob + noderadius
    # # j1coarsevirtual = iysrcglob - noderadius
    # # j2coarsevirtual = iysrcglob + noderadius
    # i1coarsevirtual = minimum(ijsrcpts[:,1]) - noderadius
    # i2coarsevirtual = maximum(ijsrcpts[:,1]) + noderadius
    # j1coarsevirtual = minimum(ijsrcpts[:,2]) - noderadius
    # j2coarsevirtual = maximum(ijsrcpts[:,2]) + noderadius
    # # if hitting borders
    # outxmin = i1coarsevirtual<1
    # outxmax = i2coarsevirtual>n1_coarse
    # outymin = j1coarsevirtual<1 
    # outymax = j2coarsevirtual>n2_coarse
    # outxmin ? i1coarse=1         : i1coarse=i1coarsevirtual
    # outxmax ? i2coarse=n1_coarse : i2coarse=i2coarsevirtual
    # outymin ? j1coarse=1         : j1coarse=j1coarsevirtual
    # outymax ? j2coarse=n2_coarse : j2coarse=j2coarsevirtual
    
    # ##
    # ## Refined grid parameters
    # ##
    # # fine grid size
    # n1_fine = (i2coarse-i1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
    # n2_fine = (j2coarse-j1coarse)*downscalefactor+1     #downscalefactor * (2*noderadius) + 1 # odd number
   
    # ##
    # ## Get the vel around the source on the coarse grid
    # ##
    # velcoarsegrd = view(vel,i1coarse:i2coarse,j1coarse:j2coarse)    

    # ##
    # ## Nearest neighbor interpolation for velocity on finer grid
    # ##
    # n1_window_coarse = i2coarse-i1coarse+1
    # n2_window_coarse = j2coarse-j1coarse+1

    # # @show ijsrcpts
    # # @show (i1coarse,i2coarse),(j1coarse,j2coarse)
    # # @show n1_fine,n2_fine
    # # @show 
    # # @show n1_window_coarse,n2_window_coarse
    # # @show size(velcoarsegrd)

    # nearneigh_oper = spzeros(n1_fine*n2_fine, n1_window_coarse*n2_window_coarse)
    # #vel_fine = Array{Float64}(undef,n1_fine,n2_fine)
    # nearneigh_idxcoarse = zeros(Int64,size(nearneigh_oper,1))

    # for j=1:n2_fine
    #     for i=1:n1_fine
    #         di=div(i-1,downscalefactor)
    #         ri=i-di*downscalefactor
    #         ii = ri>=downscalefactor/2+1 ? di+2 : di+1

    #         dj=div(j-1,downscalefactor)
    #         rj=j-dj*downscalefactor
    #         jj = rj>=downscalefactor/2+1 ? dj+2 : dj+1

    #         # vel_fine[i,j] = velcoarsegrd[ii,jj]

    #         # compute the matrix acting as nearest-neighbor operator (for gradient calculations)
    #         f = i  + (j-1)*n1_fine
    #         c = ii + (jj-1)*n1_window_coarse
    #         nearneigh_oper[f,c] = 1.0

    #         ##
    #         #@show i,j,f,c
    #         nearneigh_idxcoarse[f] = c
    #     end
    # end

    # ##--------------------------------------------------
    # ## get the interpolated velocity for the fine grid
    # tmp_vel_fine = nearneigh_oper * vec(velcoarsegrd)
    # vel_fine = reshape(tmp_vel_fine,n1_fine,n2_fine)
    # # @show size(nearneigh_oper)
    # # @show size(vel_fine)


    # if simtype==:cartesian
    #     # set origin of the fine grid
    #     xinit = grd.x[i1coarse]
    #     yinit = grd.y[j1coarse]
    #     dh = grd.hgrid/downscalefactor
    #     # fine grid
    #     grdfine = Grid2D(hgrid=dh,xinit=xinit,yinit=yinit,nx=n1_fine,ny=n2_fine)

    # elseif simtype==:spherical
    #     # set origin of the fine grid
    #     rinit = grd.r[i1coarse]
    #     θinit = grd.θ[j1coarse]
    #     dr = grd.Δr/downscalefactor
    #     dθ = grd.Δθ/downscalefactor
    #     # fine grid
    #     grdfine = Grid2DSphere(Δr=dr,Δθ=dθ,nr=n1_fine,nθ=n2_fine,rinit=rinit,θinit=θinit)
    # end

    # srcrefvars = SrcRefinVars2D(downscalefactor,
    #                             (i1coarse,j1coarse,i2coarse,j2coarse),
    #                             (n1_window_coarse,n2_window_coarse),
    #                             (outxmin,outxmax,outymin,outymax),
    #                             nearneigh_oper,nearneigh_idxcoarse,vel_fine)

    return grdfine,srcrefvars
end

###########################################################################

"""
$(TYPEDSIGNATURES)

 Define the "box" of nodes around/including the source.
"""
function sourceboxloctt!(fmmvars::FMMVars3D,vel::Array{Float64,3},srcpos::AbstractVector,
                         grd::Grid3DCart )

    # ## source location, etc.      
    # mindistsrc = 10.0*eps()
    # xsrc,ysrc=srcpos[1],srcpos[2]

    # # get the position and velocity of corners around source
    # _,velcorn,ijsrc = bilinear_interp(vel,grd,srcpos,outputcoeff=true)
    
    # ## Set srcboxpar
    # Ncorn = size(ijsrc,1)
    # fmmvars.srcboxpar.ijsrc .= ijsrc
    # fmmvars.srcboxpar.xysrc .= srcpos
    # fmmvars.srcboxpar.velcorn .= velcorn

    # ## set ttime around source ONLY FOUR points!!!
    # for l=1:Ncorn
    #     i,j = ijsrc[l,:]

    #     ## set status = accepted == 2
    #     fmmvars.status[i,j] = 2

    #     ## corner position 
    #     xp = grd.x[i] 
    #     yp = grd.y[j]

    #     # set the distance from corner to origin
    #     distcorn = sqrt((xsrc-xp)^2+(ysrc-yp)^2)
    #     fmmvars.srcboxpar.distcorn[l] = distcorn

    #     # set the traveltime to corner
    #     fmmvars.ttime[i,j] = distcorn / vel[i,j] #velsrc

    # end
    return
end 

#################################################################################

function createsparsederivativematrices!(grd::AbstractGridEik3D,
                                         adjvars::AdjointVars3D,
                                         status::Array{Integer,3})

    # if typeof(grd)==Grid2DCart
    #     simtype = :cartesian
    # elseif typeof(grd)==Grid2DSphere
    #     simtype = :spherical
    # end

    # ptij = MVector(0,0)

    # # pre-determine derivative coefficients for positive codes (0,+1,+2)
    # if simtype==:cartesian
    #     n1,n2 = grd.nx,grd.ny
    #     hgrid = grd.hgrid
    #     #allcoeff = [[-1.0/hgrid, 1.0/hgrid], [-3.0/(2.0*hgrid), 4.0/(2.0*hgrid), -1.0/(2.0*hgrid)]]
    #     allcoeffx = CoeffDerivCartesian( MVector(-1.0/hgrid,
    #                                              1.0/hgrid), 
    #                                      MVector(-3.0/(2.0*hgrid),
    #                                              4.0/(2.0*hgrid),
    #                                              -1.0/(2.0*hgrid)) )
    #     # coefficients along X are the same than along Y
    #     allcoeffy = allcoeffx

    # elseif simtype==:spherical
    #     n1,n2 = grd.nr,grd.nθ
    #     Δr = grd.Δr
    #     ## DEG to RAD !!!!
    #     Δarc = [grd.r[i] * deg2rad(grd.Δθ) for i=1:grd.nr]
    #     # coefficients
    #     coe_r_1st = [-1.0/Δr  1.0/Δr]
    #     coe_r_2nd = [-3.0/(2.0*Δr)  4.0/(2.0*Δr) -1.0/(2.0*Δr) ]
    #     coe_θ_1st = [-1.0 ./ Δarc  1.0 ./ Δarc ]
    #     coe_θ_2nd = [-3.0./(2.0*Δarc)  4.0./(2.0*Δarc)  -1.0./(2.0*Δarc) ]

    #     allcoeffx = CoeffDerivSpherical2D( coe_r_1st, coe_r_2nd )
    #     allcoeffy = CoeffDerivSpherical2D( coe_θ_1st, coe_θ_2nd )

    # end

    # ##--------------------------------------------------------
    # ## pre-compute the mapping between fmm and original order
    # n12 = n1*n2
    # nptsonsrc = count(adjvars.fmmord.onsrccols)
    # #nptsonh   = count(adjvars.fmmord.onhpoints)

    # for i=1:n12
    #     # ifm = index from fast marching ordering
    #     ifm = adjvars.idxconv.lfmm2grid[i]
    #     if ifm==0
    #         # ifm is zero = end of indices for fmm ordering [lfmm2grid=zeros(Int64,nxyz)]
    #         Nnnzsteps = i-1
    #         # remove rows corresponding to points in the source region
    #         adjvars.fmmord.vecDx.Nsize[1] = Nnnzsteps - nptsonsrc
    #         adjvars.fmmord.vecDx.Nsize[2] = Nnnzsteps
    #         adjvars.fmmord.vecDy.Nsize[1] = Nnnzsteps - nptsonsrc
    #         adjvars.fmmord.vecDy.Nsize[2] = Nnnzsteps
    #         break
    #     end
    #     adjvars.idxconv.lgrid2fmm[ifm] = i
    # end

    # ##
    # ## Set the derivative operators in FMM order, skipping source points onwards
    # ## 
    # ##   From adjvars.fmmord.vecDx.lastrowupdated[]+1 onwards because some rows might have already
    # ##   been updated above and the CSR format used here requires to add rows in sequence to
    # ##   avoid expensive re-allocations
    # #startloop = adjvars.fmmord.vecDx.lastrowupdated[]+1
    # #startloop = count(adjvars.fmmord.onsrccols)+1
    # nptsfixedtt = count(adjvars.fmmord.onsrccols )

    # colinds = MVector(0,0,0)
    # colvals = MVector(0.0,0.0,0.0)
    # idxperm = MVector(0,0,0)

    # #for irow=startloop:n12
    # for irow=1:adjvars.fmmord.vecDx.Nsize[1]
        
    #     # compute the coefficients for X  derivatives
    #     setcoeffderiv2D!(adjvars.fmmord.vecDx,status,irow,adjvars.idxconv,adjvars.codeDxy,allcoeffx,ptij,
    #                      colinds,colvals,idxperm,nptsfixedtt, axis=:X,simtype=simtype)

        
    #     # compute the coefficients for Y derivatives
    #     setcoeffderiv2D!(adjvars.fmmord.vecDy,status,irow,adjvars.idxconv,adjvars.codeDxy,allcoeffy,ptij,
    #                      colinds,colvals,idxperm,nptsfixedtt, axis=:Y,simtype=simtype)
        
    # end

    return
end


###################################################################

"""
$(TYPEDSIGNATURES)

 Test if point is on borders of domain.
"""
function isonbord(ib::Int64,jb::Int64,kb::Int64,nx::Int64,ny::Int64,nz::Int64)
    isonb1 = false
    isonb2 = false
    ## check if the point is outside ranges for 1st order
    if ib<1 || ib>nx || jb<1 || jb>ny || kb<1 || kb>nz
        isonb1 =  true
    end
    ## check if the point is outside ranges for 2nd order
    if ib<2 || ib>nx-1 || jb<2 || jb>ny-1 || kb<2 || kb>nz-1 
        isonb2 =  true
    end
    return isonb1,isonb2
end

##########################################################################

"""
$(TYPEDSIGNATURES)

   Compute the traveltime at a given node using 2nd order stencil 
    where possible, otherwise revert to 1st order. 
   Three-dimensional Cartesian or spherical grid.
"""
function calcttpt_2ndord!(fmmvars::FMMVars3D,vel::Array{Float64,3},
                          grd::AbstractGridEik3D,ij::MVector{3,Int64},
                          codeD::MVector{3,<:Integer})
    
    # #######################################################
    # ##  Local solver Sethian et al., Rawlison et al.  ???##
    # #######################################################

    # # The solution from the quadratic eq. to pick is the larger, see 
    # #  Sethian, 1996, A fast marching level set method for monotonically
    # #  advancing fronts, PNAS

    # if typeof(grd)==Grid2D
    #     simtype=:cartesian
    # elseif typeof(grd)==Grid2DSphere
    #     simtype=:spherical
    # end

    # # sizes, etc.
    # if simtype==:cartesian
    #     n1 = grd.nx
    #     n2 = grd.ny
    #     Δh = MVector(grd.hgrid,grd.hgrid)
    # elseif simtype==:spherical
    #     n1 = grd.nr
    #     n2 = grd.nθ
    #     Δh = MVector(grd.Δr, grd.r[i]*deg2rad(grd.Δθ))
    # end
    # # slowness
    # slowcurpt = 1.0/vel[i,j]

    # ## Finite differences:
    # ##
    # ##  Dx_fwd = (tX[i+1] - tcur[i])/dx 
    # ##  Dx_bwd = (tcur[i] - tX[i-1])/dx 
    # ##

    # ##################################################
    # ### Solve the quadratic equation
    # #  "A second-order fast marching eikonal solver"
    # #    James Rickett and Sergey Fomel, 2000
    # ##################################################
    
    # alpha = 0.0
    # beta  = 0.0
    # gamma = - slowcurpt^2 ## !!!!
    # HUGE = typemax(eltype(vel)) #1.0e30
    # codeD[:] .= 0 # integers

    # ## 2 axis, x and y
    # for axis=1:2
        
    #     use1stord = false
    #     use2ndord = false
    #     chosenval1 = HUGE
    #     chosenval2 = HUGE
        
    #     ## 2 directions: forward or backward
    #     for l=1:2

    #         ## map the 4 cases to an integer as in linear indexing...
    #         lax = l + 2*(axis-1)
    #         if lax==1 # axis==1
    #             ish = 1 #1
    #             jsh = 0 #0
    #         elseif lax==2 # axis==1
    #             ish = -1  #-1
    #             jsh = 0  #0
    #         elseif lax==3 # axis==2
    #             ish = 0 #0
    #             jsh = 1 #1
    #         elseif lax==4 # axis==2
    #             ish = 0 # 0
    #             jsh = -1 #-1
    #         end

    #         ## check if on boundaries
    #         isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                                    
    #         ##==== 1st order ================
    #         if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
    #             ## first test value
    #             testval1 = fmmvars.ttime[i+ish,j+jsh]

    #             ## pick the lowest value of the two
    #             if testval1<chosenval1 ## < only!!!
    #                 chosenval1 = testval1
    #                 use1stord = true

    #                 # save derivative choices
    #                 axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
                    
    #                 ##==== 2nd order ================
    #                 ish2::Int64 = 2*ish
    #                 jsh2::Int64 = 2*jsh
    #                 if !isonb2nd && fmmvars.status[i+ish2,j+jsh2]==2 ## 2==accepted
    #                     # second test value
    #                     testval2 = fmmvars.ttime[i+ish2,j+jsh2]
    #                     ## pick the lowest value of the two
    #                     ##    compare to chosenval 1, *not* 2!!
    #                     ## This because the direction has already been chosen
    #                     ##  at the line "testval1<chosenval1"
    #                     if testval2<chosenval1 ## < only!!!
    #                         chosenval2 = testval2
    #                         use2ndord = true 
    #                         # save derivative choices
    #                         axis==1 ? (codeD[axis]=2*ish) : (codeD[axis]=2*jsh)
    #                     else
    #                         chosenval2=HUGE
    #                         # below in case first direction gets 2nd ord
    #                         #   but second direction, with a smaller testval1,
    #                         #   does *not* get a second order
    #                         use2ndord=false # this is needed!
    #                         # save derivative choices
    #                         axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)
    #                     end

    #                 end ##==== END 2nd order ================
                    
    #             end 
    #         end ##==== END 1st order ================


    #     end # end two sides

    #     ## spacing
    #     deltah = Δh[axis]

        
    #     if use2ndord && use1stord # second order
    #         tmpa2 = 1.0/3.0 * (4.0*chosenval1-chosenval2)
    #         ## curalpha: make sure you multiply only times the
    #         ##   current alpha for beta and gamma...
    #         curalpha = 9.0/(4.0 * deltah^2)
    #         alpha += curalpha
    #         beta  += ( -2.0*curalpha * tmpa2 )
    #         gamma += curalpha * tmpa2^2 ## see init of gamma : - slowcurpt^2

    #     elseif use1stord # first order
    #         ## curalpha: make sure you multiply only times the
    #         ##   current alpha for beta and gamma...
    #         curalpha = 1.0/deltah^2 
    #         alpha += curalpha
    #         beta  += ( -2.0*curalpha * chosenval1 )
    #         gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
    #     end

    # end ## for axis=1:2

    # ## compute discriminant 
    # sqarg = beta^2-4.0*alpha*gamma

    # ## To get a non-negative discriminant, need to fulfil:
    # ##    (tx-ty)^2 - 2*s^2/curalpha <= 0
    # ##    where tx,ty can be
    # ##     t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order
    # ##     t? = chosenval1  if 1st order 
    # ##

    # ##=========================================================================================
    # ## If discriminant is negative (probably because of sharp contrasts in
    # ##  velocity) revert to 1st order for both x and y
    # # if sqarg<0.0

    # #     begin    
    # #         codeD[:] .= 0 # integers
    # #         alpha = 0.0
    # #         beta  = 0.0
    # #         gamma = - slowcurpt^2 ## !!!!

    # #         ## 2 directions
    # #         for axis=1:2
                
    # #             use1stord = false
    # #             chosenval1 = HUGE
                
    # #             ## two sides for each direction
    # #             for l=1:2

    # #                 ## map the 4 cases to an integer as in linear indexing...
    # #                 lax = l + 2*(axis-1)
    # #                 if lax==1 # axis==1
    # #                     ish = 1
    # #                     jsh = 0
    # #                 elseif lax==2 # axis==1
    # #                     ish = -1
    # #                     jsh = 0
    # #                 elseif lax==3 # axis==2
    # #                     ish = 0
    # #                     jsh = 1
    # #                 elseif lax==4 # axis==2
    # #                     ish = 0
    # #                     jsh = -1
    # #                 end

    # #                 ## check if on boundaries
    # #                 isonb1st,isonb2nd = isonbord(i+ish,j+jsh,n1,n2)
                    
    # #                 ## 1st order
    # #                 if !isonb1st && fmmvars.status[i+ish,j+jsh]==2 ## 2==accepted
    # #                     testval1 = fmmvars.ttime[i+ish,j+jsh]
    # #                     ## pick the lowest value of the two
    # #                     if testval1<chosenval1 ## < only
    # #                         chosenval1 = testval1
    # #                         use1stord = true

    # #                         # save derivative choices
    # #                         axis==1 ? (codeD[axis]=ish) : (codeD[axis]=jsh)

    # #                     end
    # #                 end
    # #             end # end two sides

    # #             ## spacing
    # #             deltah = Δh[axis]
                
    # #             if use1stord # first order
    # #                 ## curalpha: make sure you multiply only times the
    # #                 ##   current alpha for beta and gamma...
    # #                 curalpha = 1.0/deltah^2 
    # #                 alpha += curalpha
    # #                 beta  += ( -2.0*curalpha * chosenval1 )
    # #                 gamma += curalpha * chosenval1^2 ## see init of gamma : - slowcurpt^2
    # #             end
    # #         end
            
    # #         ## recompute sqarg
    # #         sqarg = beta^2-4.0*alpha*gamma

    # #     end ## begin...

    #     if sqarg<0.0

    #         if fmmvars.allowfixsqarg==true
            
    #             gamma = beta^2/(4.0*alpha)
    #             sqarg = beta^2-4.0*alpha*gamma
    #             println("calcttpt_2ndord(): ### Brute force fixing problems with 'sqarg', results may be quite inaccurate. ###")
                
    #         else
    #             println("\n To get a non-negative discriminant, need to fulfil: ")
    #             println(" (tx-ty)^2 - 2*s^2/curalpha <= 0")
    #             println(" where tx,ty can be")
    #             println(" t? = 1.0/3.0 * (4.0*chosenval1-chosenval2)  if 2nd order")
    #             println(" t? = chosenval1  if 1st order ")
                
    #             error("calcttpt_2ndord(): sqarg<0.0, negative discriminant (at i=$i, j=$j)")
    #         end
    #      end
    # # end ## if sqarg<0.0
    # ##=========================================================================================

    # ### roots of the quadratic equation
    # tmpsq = sqrt(sqarg)
    # soughtt1 =  (-beta + tmpsq)/(2.0*alpha)
    # soughtt2 =  (-beta - tmpsq)/(2.0*alpha)

    # ## choose the largest solution
    # soughtt = max(soughtt1,soughtt2)

    return soughtt
end

################################################################################

"""
$(TYPEDSIGNATURES)

Find closest node on a grid to a given point.
"""
function findclosestnode(x::Float64,y::Float64,z::Float64,xinit::Float64,yinit::Float64,zinit::Float64,h::Float64) 
    # xini ???
    # yini ???
    ix = floor((x-xinit)/h)
    iy = floor((y-yinit)/h)
    iz = floor((z-zinit)/h)
    rx = x-(ix*h+xinit)
    ry = y-(iy*h+yinit)
    rz = z-(iz*h+zinit)
    middle = h/2.0

    if rx>=middle 
        ix = ix+1
    end
    if ry>=middle 
        iy = iy+1
    end
    if rz>=middle 
        iz = iz+1
    end
    return Int(ix+1),Int(iy+1),Int(iz+1) # julia
end

#############################################################
