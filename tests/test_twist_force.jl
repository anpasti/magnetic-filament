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

global n = 20#50#80 # number of points - each corresponds to an length element of a rod.
global h = 1/n # distance between 2 pts # takes into account the half a point extention on each side

# initialize the shape
global rvecs = zeros(n,3)
global rvecs[:,1] = range(-0.5,0.5,length=n)
x = rvecs[:,1]
rvecs[:,2] = 0.01*exp.( -(rvecs[:,1]).^2 / 0.2^2 ) #0.01 * cos.( pi*rvecs[:,1] )#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
yll = exp.(-25*rvecs[:,1].^2) .* (25*rvecs[:,1].^2 .- 0.5)
ylll = exp.(-25*rvecs[:,1].^2) .* (75*rvecs[:,1] - 1250*rvecs[:,1].^3)
rvecs[:,3] = -0.01*exp.( -(rvecs[:,1].+0.05).^2 / 0.2^2 ) + 0.01*exp.( -(rvecs[:,1].-0.05).^2 / 0.2^2 )
zll = 1/16 * exp.(-1/16 * (20*x .+ 1).^2) .* (  -400*x.^2 + exp.(5*x).*(400*x.^2 - 40*x .-7) -40*x .+ 7  )
zlll = -5/32 * exp.(-1/16 * (20*x .+ 1).^2) .* (  -8000*x.^3 -1200*x.^2 + exp.(5*x) .* (  8000*x.^3 - 1200*x.^2 - 400*x .+ 23  )  +420*x .+23  )
#rvecs[:,3] = 0.001 * sin.( pi*rvecs[:,1] ) #d*ones(size(rvecs[:,3]))
renormalize_length!(rvecs)

om_twist = 0.1 * (range(-0.5,0.5,length=n)).*2  # -twist_omega*zeta_rot / C * (rvecs[:,1] .- 0.5)
om_twistl = 0.2

D1 = make_D1(rvecs)/h
D2_BC = make_D2_BC(rvecs)/h^2 # with boundary condition 0 at ends
D2 = make_D2(rvecs)/h^2
D3 = make_D3(rvecs)/h^3

ftwist_arr = make_ftwist_arr(h, om_twist, rvecs, D1, D2) # C/A=1 
ftwist_vec = reshape(ftwist_arr, 3, n)'

ftwist_arr2 = make_ftwist_arr_manual(h, om_twist, rvecs) # C/A=1
ftwist_vec2 = reshape(ftwist_arr2, 3, n)'

xs = rvecs[:,1]
ys = rvecs[:,2]
zs = rvecs[:,3]

ftwist_x_teor  = om_twist  * 0
ftwist_y_teor  = -om_twistl  .* zll  -  om_twist  .* zlll
ftwist_z_teor  = om_twistl  .* yll  + om_twist  .* ylll

plot(ftwist_vec, color = "blue") # seems better
plot!(ftwist_vec2, color="red") # seems slightly worse
plot!(ftwist_x_teor, color="green")
plot!(ftwist_y_teor, color="green")
plot!(ftwist_z_teor, color="green",legend = false)
