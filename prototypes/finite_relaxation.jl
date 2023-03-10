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


# precession frequency multiplied by relaxation time
ωτ = 0.5

# infinitely fast precessing field
θmagic = 0.5*acos(-(1+2*ωτ^2)/(3+2*ωτ^2)) #magic angle
θ = θmagic+0.08 #+0.2 # precession angle

# torque density due to finite relaxation
Cmrel0 = 30
Cmrel = Cmrel0 * ωτ/(1+ωτ^2) * sin(θ)

# Magnetic force
Cm0 = 400
Cm = Cm0 * 1/(1+ωτ^2) * ( 1+2*ωτ^2 + (3+2*ωτ^2) * cos(2*θ) )

# twist elasticity over bending elasticity constant
C = 1

# anisotropy of drag
lambda = -(2 - 1)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag

# number of discretized elements
n = 51#50#80 # number of points - each corresponds to an length element of a rod.
h = 1/n # distance between 2 pts # takes into account the half a point extention on each side

# # initialize the shape
rvecs = zeros(n,3)
rvecs[:,1] = 0.01*range(-0.5,0.5,length=n)
#rvecs[:,2] = range(-0.5,0.5,length=n)
rvecs[:,3] = range(-0.5,0.5,length=n)

# # initialize the shape
# rvecs = zeros(n,3)
# rvecs[:,1] = cosh.(range(-0.5,0.5,length=n)).^4
# #rvecs[:,2] = range(-0.5,0.5,length=n)
# rvecs[:,3] = range(-0.5,0.5,length=n)

# initialize the shape
# global rvecs = zeros(n,3)
# phi = range(0,2*pi*1.9,length=n)
# global rvecs[:,1] = range(0.9,1,length=n) .* cos.(phi)#range(-0.5,0.5,length=n)
# rvecs[:,2] = range(1,0.9,length=n) .* sin.(phi)#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
# rvecs[:,3] = range(-0.5,0.5,length=n) #0.001 * sin.( pi*range(-0.5,0.5,length=n) ) #d*ones(size(rvecs[:,3]))


renormalize_length!(rvecs)
rvecs = rvecs .- make_center_of_mass(rvecs)'


# differentiation matrices
D1 = make_D1(rvecs)/h
D2_BC = make_D2_BC(rvecs)/h^2 # with boundary condition 0 at ends
D2 = make_D2(rvecs)/h^2

# differentiation matrix for elastic force
Mmat = make_Mmat_ceb(h,n)


params = [lambda, h, n, Cm, Cmrel, C, Mmat, D1, D2]

function velocity!(du_arr,u_arr,params,t)
    lambda = params[1]
    h = params[2]
    n = params[3]
    Cm = params[4]
    Cmrel = params[5]
    C = params[6]
    Mmat = params[7]
    D1 = params[8]
    D2 = params[9]
    
    rvecs = reshape(u_arr[:], 3, n)'
    r_arr = reshape(rvecs',3*n,1)
    

    # projection operator on inextensible motion
    P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    P = μ*P

    # magnetic force
    hvec = [0,0,1] # fast precessing field is effectively pointing in the ez direction
    Mmat_Fparamag = make_Mmat_Fparamag(h,n,hvec, Cm)

    # finite magentic relaxation force

    frelax_arr = make_frelax_arr(h,rvecs, D1)

    # twist
    c1 = rvecs[1,3]-rvecs[n,3]
    c2 = -(rvecs[1,3]+rvecs[n,3])/2
    l = range(-0.5,0.5,length=n)
    om_twist = -Cmrel/C * (  rvecs[:,3] + c1*l .+ c2  )
    ftwist_arr = make_ftwist_arr(h, om_twist, rvecs, D1, D2)
    
    total_force_arr = (Mmat + Mmat_Fparamag)*r_arr + C*ftwist_arr + Cmrel*frelax_arr
    du_arr[:] = P*(total_force_arr)
    # vvecs = reshape(du_arr[1:3*n], 3, n)'
    # du_arr[3*n+1:end] = 0*C / zeta_rot * D2om_twist + 0*make_dot_product(make_cross_product(D1*rvecs,D2_BC*rvecs), D1*vvecs)
end




# solve

# plot first frame
display(Plots.plot(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal))

tend = 0.01
tspan = (0,tend)
show_iters = 20
r_arr0 = reshape(rvecs',3*n,1)
prob = ODEProblem(velocity!,r_arr0,tspan,params)
@time global sol = solve(prob,alg_hints=[:stiff],saveat=tend/show_iters,reltol=1e-8, abstol=1e-8)

maxdevs = zeros(show_iters+1)
for (i, u) = enumerate(sol.u)
    local rvecs = reshape(u, 3, n)'
    local t = sol.t[i]

    display(Plots.plot!(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal,label="", color = :red))
    
    rvecs = rvecs .- make_center_of_mass(rvecs)'
    maxdevs[i] = sqrt(maximum(rvecs[:,1])^2 + maximum(rvecs[:,2])^2)
    println("max deviation = ", maxdevs[i])
end

for (i, u) = enumerate(sol.u)
    global rvecs_end = reshape(u, 3, n)'
end

c1 = rvecs_end[1,3]-rvecs_end[n,3]
c2 = -(rvecs_end[1,3]+rvecs_end[n,3])/2
l = range(-0.5,0.5,length=n)
om_twist_end = -Cmrel/C * (  rvecs_end[:,3] + c1*l .+ c2  )

#plot last frame
display(Plots.plot(rvecs_end[:,1],rvecs_end[:,2],rvecs_end[:,3],aspect_ratio=:equal))

println("Cm = ", Cm)
println("end elonagetion = ", rvecs_end[n,3] - rvecs_end[1,3])