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

Cm = 200
omega = 10* Cm

global lambda = -(2 - 1)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag
global T = 2*pi / abs(omega)
global n = 80#50#80 # number of points - each corresponds to an length element of a rod.
global h = 1/n # distance between 2 pts # takes into account the half a point extention on each side


println("*****************************")
println("*****************************")
println("*****************************")
println("Cm = ", Cm)
println("omega = ", omega)
println("ζratio = ", -lambda+1 )
println("n = ", n)

# initialize the shape
global rvecs = zeros(n,3)
global rvecs[:,1] = range(-0.5,0.5,length=n)
rvecs[:,2] = 0.01*exp.( -(rvecs[:,1]).^2 / 0.2^2 ) #0.01 * cos.( pi*rvecs[:,1] )#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
rvecs[:,3] = -0.01*exp.( -(rvecs[:,1].+0.1).^2 / 0.2^2 ) + 0.01*exp.( -(rvecs[:,1].-0.1).^2 / 0.2^2 )
#rvecs[:,3] = 0.001 * sin.( pi*rvecs[:,1] ) #d*ones(size(rvecs[:,3]))
renormalize_length!(rvecs)

# initialize twist
zeta_rot = 0.003 # D in my notes
C = 1. # twist elasticity constant
twist_omega = 0.1/zeta_rot # one end rotation velocity
#om_twist = -rvecs[:,1] * twist_omega*zeta_rot / C#-(rvecs[:,1] .- 0.5) * twist_omega*zeta_rot / C # initial linear distribution
om_twist = -0.4 * (range(-0.5,0.5,length=n))*2#-twist_omega*zeta_rot / C * (rvecs[:,1] .- 0.5)

println("max_om_twist = ", maximum(om_twist))
starttwist = maximum(om_twist)

# differentiation matrix for elastic force
global Mmat = make_Mmat_ceb(h,n)
# differentiation matrices
D1 = make_D1(rvecs)/h
D2_BC = make_D2_BC(rvecs)/h^2 # with boundary condition 0 at ends
D2 = make_D2(rvecs)/h^2

params = [lambda, h, n, omega, Cm, zeta_rot, C, Mmat, D1, D2, twist_omega, om_twist]

function velocity!(du_arr,u_arr,params,t)
    lambda = params[1]
    h = params[2]
    n = params[3]
    omega = params[4]
    Cm = params[5]
    zeta_rot = params[6]
    C = params[7]
    Mmat = params[8]
    D1 = params[9]
    D2 = params[10]
    twist_omega = params[11]
    om_twist = params[12]
    
    rvecs = reshape(u_arr[:], 3, n)'
    r_arr = reshape(rvecs',3*n,1)
    
    #om_twist[1] = 0
    #om_twist[end] = 0 

    # D1om_twist = D1*om_twist
    # #D1om_twist[1] = -twist_omega * zeta_rot / C # set boundary condition
    # #D1om_twist[end] = -twist_omega * zeta_rot / C # set boundary condition
    # D2om_twist = D1*D1om_twist

    P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    P = μ*P
    #hvec = -normalize([cos(omega*t),sin(omega*t),1/sqrt(2)]) # magic angle
    
    #Mmat_Fparamag = make_Mmat_Fparamag(h,n,hvec, Cm)

    ftwist_arr = make_ftwist_arr(h, om_twist, rvecs, D1, D2)

    total_force_arr = Mmat*r_arr + C*ftwist_arr
    total_force_vec = reshape(total_force_arr[:], 3, n)'
    total_force_vec[1,2:3] = [0,0] # start and end clamped
    total_force_vec[end,2:3] = [0,0] # start and end clamped
    total_force_arr = reshape(total_force_vec',3*n,1)
    du_arr[:] = P*(total_force_arr)
    # vvecs = reshape(du_arr[1:3*n], 3, n)'
    # du_arr[3*n+1:end] = 0*C / zeta_rot * D2om_twist + 0*make_dot_product(make_cross_product(D1*rvecs,D2_BC*rvecs), D1*vvecs)
end

# plot first frame
display(Plots.plot(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal))

global rvecs
tend = 0.03/100
tspan = (0,tend)
r_arr0 = reshape(rvecs',3*n,1)
prob = ODEProblem(velocity!,r_arr0,tspan,params)
@time global sol = solve(prob,alg_hints=[:stiff],saveat=tend/20,reltol=1e-8, abstol=1e-8)

for (i, u) = enumerate(sol.u)
    local rvecs = reshape(u, 3, n)'
    local t = sol.t[i]
    #local hvec = -normalize([cos(omega*t),sin(omega*t),1/sqrt(2)])
    #println(t/T)
    display(Plots.plot!(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal,label="", color = :red))
    #display(Plots.plot!([0,hvec[1]/4],[0,hvec[2]/4],arrow=true,color=:black,linewidth=2,label=""))
    # Plots.plot!(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal,label="", color = :red)
    # display(Plots.plot!([0,hvec[1]/4],[0,hvec[2]/4], [0,hvec[3]/4],arrow=true,color=:black,linewidth=2,label=""))

    # tvecs = make_tvecs(rvecs)
    # if dot(tvecs[end,:],hvec) >= 0 # nez kāpēc vajag šādu zīmi
    #     println("backnforth motion")
    #     global backnforth = true
    #     global converged = false
    # end
end

global t = sol.t[end]
println("t/T = ", t/T)
#renormalize length
global rvecs = reshape(sol.u[end], 3, n)'
println(make_length_of_filament(rvecs))
renormalize_length!(rvecs)

println("end max_om_twist = ", maximum(om_twist))
println("decrease = ", (starttwist - maximum(om_twist))/starttwist)

rvecsend = reshape(sol.u[end][1:3*n], 3, n)'
Plots.plot(rvecsend[:,1],rvecsend[:,2],rvecsend[:,3],aspect_ratio=:equal,label="", color = :red)
Plots.plot(rvecsend[:,1],rvecsend[:,2],aspect_ratio=:equal,label="", color = :red)
Plots.plot(rvecsend[:,2],rvecsend[:,3],aspect_ratio=:equal,label="", color = :red)
Plots.plot(rvecsend[:,1],rvecsend[:,3],aspect_ratio=:equal,label="", color = :red)
# plot(rvecsend[:,2],rvecsend[:,3])
# display(Plots.plot!([0,hvec[1]/4],[0,hvec[2]/4], [0,hvec[3]/4],arrow=true,color=:black,linewidth=2,label=""))

# h5open("./phase_space_zeta_2/Cm_$(Cm)_w_$(omega).h5", "w") do file
#     write(file, "n", n) 
#     write(file, "dt", "solved with ODE solver") 
#     write(file, "t", t) 
#     write(file, "converged", converged) 
#     write(file, "backnforth", backnforth) 
#     write(file, "omega", omega) 
#     write(file, "Cm", Cm) 
#     write(file, "lambda", lambda)
#     write(file, "rvecs", rvecs_rotated) 
#     write(file, "hvec", hvec_rotated) 
