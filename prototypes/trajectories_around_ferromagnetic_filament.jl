using Plots
using LinearAlgebra
using DiffEqOperators
using Elliptic
using Roots
using Rotations
using StaticArrays
using HDF5
using DifferentialEquations
using Profile
using SparseArrays
using Glob

include("../functions/physics_functions.jl")
include("../functions/finite_difference_functions.jl")
include("../functions/filament_geometry_functions.jl")


filedir = "/home/andris/Documents/magnetic-filament/simulation_results/ferromagnetic_rotating"
writedir = "/home/andris/Documents/magnetic-filament/simulation_results/ferromagnetic_rotating/tracer_trajectories"

filename = glob("*.h5",filedir)[1] # choose first filament


fid = h5open(filename, "r")
n = read(fid, "n")
h = 1/n
Cm = read(fid, "Cm")
omega = read(fid, "omega")
rvecs = read(fid, "rvecs")
vvecs = read(fid, "vvecs")
tvecs = read(fid, "tvecs")
hvec = read(fid, "hvec")
t = read(fid, "t")
t0 = read(fid, "t0")
d = read(fid, "d")
ϵ = read(fid, "rho")
lambda = read(fid, "lambda")
ζratio = 1- read(fid, "lambda")


# ## for rigid rod
# rvecs = zeros(n,3)
# rvecs[:,1] = range(-0.5,0.5,length=n)

# ## for rigid rod
# tvecs = make_tvecs(rvecs)
# vvecs = zeros(n,3)
# vvecs[:,2] = omega*range(-0.5,0.5,length=n)

# for all
rvecs[:,3] = d*ones(size(rvecs[:,3])) # set correct height
rvecs[:,2] .*= 3 # 
renormalize_length!(rvecs) # scale

tvecs = make_tvecs(rvecs)

# rotating velocity field of the filament. should coincide with vvecs
vvecsrot = zeros(n,3)
vvecsrot[:,1] = -omega*rvecs[:,2]
vvecsrot[:,2] = omega*rvecs[:,1]
vvecs = vvecsrot


vvecs_par = make_dot_product(vvecs,tvecs) .* tvecs
vvecs_perp = vvecs - vvecs_par
fvecs = 1/ζratio * vvecs_par + vvecs_perp # force denisity
fvecs = fvecs * h # differential force


tmp3x1Float1 = [0.,0.,0.]
tmp3x1Float2 = [0.,0.,0.]
tmp3x1Float3 = [0.,0.,0.]
tmp3x1Float4 = [0.,0.,0.]
vvec = [0.,0.,0.]

plot(rvecs[:,1],rvecs[:,2],rvecs[:,3])


params = [omega, rvecs, fvecs, ϵ, d, h, tmp3x1Float1, tmp3x1Float2,tmp3x1Float3,tmp3x1Float4]
function velocity_surrounding_rotframe!(du_arr,u_arr,params,t)
    omega = params[1]
    rvecs = params[2]
    fvecs = params[3]
    ϵ = params[4]
    d = params[5]
    h = params[6]
    tmp3x1Float1 = params[7]
    tmp3x1Float2 = params[8]
    tmp3x1Float3 = params[9]
    tmp3x1Float4 = params[10]

    make_velocity_field_wall_fast!(du_arr,u_arr, rvecs,
                                 fvecs, ϵ, d, h, tmp3x1Float1, tmp3x1Float2,tmp3x1Float3,tmp3x1Float4)

    #add the rotating flow: {0,0,-ω} cross {x,y,z}
    du_arr[1] += u_arr[2]*omega
    du_arr[2] += -u_arr[1]*omega
end


periods = 10
tend = 2*pi/omega*periods
tspan = [0., tend]
u_arr = [0.4/sqrt(2),0.4/sqrt(2),d+0.04]
prob = ODEProblem(velocity_surrounding_rotframe!,u_arr,tspan,params)
@time sol = solve(prob, saveat=tend/100/periods,reltol=1e-8, abstol=1e-8)

xs = zeros(size(sol.u))
ys = zeros(size(sol.u))
zs = zeros(size(sol.u))
for (i, u) = enumerate(sol.u)
    xs[i] = u[1]
    ys[i] = u[2]
    zs[i] = u[3]
end

plot!(xs,ys,zs)


# plot(xs,ys)
# plot!(rvecs[:,1],rvecs[:,2])
# rvec_eval=[1.,2.,d]

# vvec0, uS, uD = make_velocity_field_wall(rvec_eval, rvecs, fvecs, ϵ, d, h)

# make_velocity_field_wall_fast!(vvec,rvec_eval, rvecs,
#                                  fvecs, ϵ, d, h, tmp3x1Float1, tmp3x1Float2,tmp3x1Float3,tmp3x1Float4)


# make_stokeslet_velocity_wall_fast!(tmp3x1Float3,fvecs[1,:],rvec_eval,rvecs[1,:],tmp3x1Float1,tmp3x1Float2)
# make_doublet_velocity_field_wall_fast!(tmp3x1Float4,fvecs[1,:],rvec_eval,rvecs[1,:],tmp3x1Float1,tmp3x1Float2)

# ss=make_stokeslet_velocity_wall(fvecs[1,:],rvec_eval,rvecs[1,:])
# dd = make_doublet_velocity_field_wall(fvecs[1,:],rvec_eval,rvecs[1,:])