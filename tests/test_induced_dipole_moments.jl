using Plots
using LinearAlgebra
using DiffEqOperators
using Elliptic
using Roots
using Rotations
using StaticArrays
using HDF5
using DifferentialEquations

include("../functions/physics_functions.jl")
include("../functions/finite_difference_functions.jl")
include("../functions/filament_geometry_functions.jl")

# Cm = 100 #pi^2+20.1 # f_mag / f_elast
# omega = -300 #15 * Cm #734/50 # 2pi / (T/tau) dimensionlessrotation frequency of the field

Cm = 1

global lambda = -(2 - 1)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag
global T = 2*pi / abs(omega)
global n = 35#50#80 # number of points - each corresponds to an length element of a rod.
global h = 1/n # distance between 2 pts # takes into account the half a point extention on each side


println("*****************************")
println("*****************************")
println("*****************************")
println("Cm = ", Cm)
println("omega = ", omega)
println("ζratio = ", -lambda+1 )
println("n = ", n)

## initialize the shaped
global rvecs = zeros(n,3)
phi = range(0.025*pi,pi*1.975,length=n)
global rvecs[:,1] = cos.(phi)#range(0.9,1,length=n) .* cos.(phi)#range(-0.5,0.5,length=n)
rvecs[:,2] = sin.(phi)#range(1,0.9,length=n) .* sin.(phi)#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
#rvecs[:,3] = 0.01 * sin.( pi*rvecs[:,1] ) #d*ones(size(rvecs[:,3]))
renormalize_length!(rvecs)
# rvecs = zeros(n,3)
# rvecs[:,2] = range(-0.5,0.5,length=n)

hvec= [0,1,0]
χ=0.66

mvecs = make_paramagnetic_moments(rvecs,hvec,h,χ,χ/(1+0.5*χ))
fdip_vecs = make_dipole_force(rvecs,mvecs,6e-5*Cm,h)

r_arr = reshape(rvecs',3*n,1)
fparamag_arr = make_Mmat_Fparamag(h,n,hvec, Cm) * r_arr
fparamag_vecs = reshape(fparamag_arr, 3, n)'

display(Plots.plot(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal))
for i = 1:n
    scale = h
    fscale = 10^-3
    marrow = [[rvecs[i,1], (rvecs[i,1]+ mvecs[i,1]*scale)],[rvecs[i,2],  (rvecs[i,2]+mvecs[i,2]*scale)]]
    display(Plots.plot!(marrow[1],marrow[2],arrow=true,color=:black,linewidth=2,label=""))
    farrow = [[rvecs[i,1], (rvecs[i,1]+ fdip_vecs[i,1]*fscale)],[rvecs[i,2],  (rvecs[i,2]+fdip_vecs[i,2]*fscale)]]
    if i != 1 && i != n
        display(Plots.plot!(farrow[1],farrow[2],arrow=true,color=:red,linewidth=2,label=""))
    else
        display(Plots.plot!(farrow[1],farrow[2],arrow=true,color=:blue,linewidth=2,label=""))
    end

    fparamagarrow = [[rvecs[i,1], (rvecs[i,1]+ fparamag_vecs[i,1]*fscale)],[rvecs[i,2],  (rvecs[i,2]+fparamag_vecs[i,2]*fscale)]]
    display(Plots.plot!(fparamagarrow[1],fparamagarrow[2],arrow=true,color=:green,linewidth=2,label=""))
end


plot(fdip_vecs)
#plot!(fparamag_vecs)