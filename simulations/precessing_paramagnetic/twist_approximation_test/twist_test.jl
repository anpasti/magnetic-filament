using Plots
using LinearAlgebra
using DiffEqOperators
using Elliptic
using Roots
using Rotations
using StaticArrays
using HDF5
using DifferentialEquations
using SparseArrays
using Random

global rng = MersenneTwister(1234) # intialize random number generator

include("../../../functions/physics_functions.jl")
include("../../../functions/finite_difference_functions.jl")
include("../../../functions/filament_geometry_functions.jl")


Cm=-100
Cmrel=150

# twist elasticity over bending elasticity constant
C = 1.
# anisotropy of drag
lambda = -(2. - 1.)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag
# rotational drag coefficient 
zeta_rot = 7e-5#1e-4#7e-5

# number of discretized elements
n = 100#50#80 # number of points - each corresponds to a length element of a rod.
h = 1/n # distance between 2 pts # takes into account the half a point extention on each side

# initialize the shape
rvecs = zeros(n,3)
rvecs[:,3] = range(-0.5,0.5,length=n)  # vertical
println("applying noise with ", rng)
rvecs += 0.01*randn(rng,(n,3))

renormalize_length!(rvecs) # scale
rvecs = rvecs .- make_center_of_mass(rvecs)' # move center of mass to 0

# initialize twist
om_twist = 0 * (range(-0.5,0.5,length=n))


# differentiation matrices
D1 = make_D1(rvecs)/h
D2_BC = make_D2_BC(rvecs)/h^2 # with boundary condition 0 at ends
D2 = make_D2(rvecs)/h^2

# differentiation matrix for elastic force
Mmat = sparse(make_Mmat_ceb(h,n))

# paramagnetic force matrix
hvec = [0,0,1] # fast precessing field is effectively pointing in the ez direction
Mmat_Fparamag = sparse(make_Mmat_Fparamag(h,n,hvec, Cm))

total_Mmat = Mmat+Mmat_Fparamag

# allocate some memory
J = initialize_J_memory(n)
μ = initialize_μ_memory(n)
tmp3x1Float = [1.,1.,1.]
tmp3x3Float = ones(3,3)

params_fast = [lambda, h, n, Cm, Cmrel, C, Mmat, Mmat_Fparamag, D1, D2, J, μ, tmp3x1Float,tmp3x3Float, total_Mmat, zeta_rot]


function velocity_fast!(du_arr,u_arr,params,t)
    lambda = params[1]
    h = params[2]
    n = params[3]
    Cm = params[4]
    Cmrel = params[5]
    C = params[6]
    #Mmat = params[7]
    #Mmat_Fparamag = params[8]
    D1 = params[9]
    D2 = params[10]
    J = params[11]
    μ = params[12]
    tmp3x1Float = params[13]
    tmp3x3Float = params[14]
    total_Mmat = params[15]
    zeta_rot = params[16]

    
    # r_arr = u_arr #@view u_arr[:]
    # rvecs = reshape(r_arr, 3, n)'
    rvecs = reshape(u_arr[1:3*n], 3, n)'
    r_arr = reshape(rvecs',3*n,1)
    om_twist = u_arr[3*n+1:end] # Omega3
    D2om_twist = D2*om_twist

    # # projection operator on inextensible motion
    # P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    # P = μ*P # dense matrix
    make_J!(J,rvecs)
    make_mobility_tensor!(μ, tmp3x1Float, tmp3x3Float , rvecs, lambda, h)

    # finite magentic relaxation force
    frelax_arr = make_frelax_arr(h,rvecs, D1)

    # twist force
    ftwist_arr = make_ftwist_arr(h, om_twist, rvecs, D1, D2)
    
    # froce without tension
    total_force_arr = total_Mmat*r_arr + C*ftwist_arr + Cmrel*frelax_arr
    
    Λ =  Symmetric(J*μ*transpose(J))  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    # centerline time derivatives
    du_arr[1:3*n] = μ*(total_force_arr + ftension_arr)
    vvecs = reshape(du_arr[1:3*n], 3, n)'
    # omega3 time derivatives
    du_arr[3*n+1:end] = 1/zeta_rot * ( C  * D2om_twist + Cmrel*D2*rvecs[:,3]  ) + make_dot_product(make_cross_product(D1*rvecs,D2*rvecs), D1*vvecs)
    du_arr[3*n+1] = 0 # boundary condition
    du_arr[end] = 0 # boundary condition

end


# solve

u_arr = vcat(reshape(rvecs',3*n,1), om_twist ) # last n elements in varaibles is Omega3
tend = 0.02 #min( 10*maximum([abs(1/Cm),1/Cmrel]), 1.)
tspan=[0., tend]
prob_fast = ODEProblem(velocity_fast!,u_arr,tspan,params_fast)

# solve ODE problem
println("calculating: Cm =", Cm, " Cmrel = ", Cmrel , " n = ", n)
@time sol_fast = solve(prob_fast,alg_hints=[:stiff],saveat=tend/50,reltol=1e-8, abstol=1e-8)

rvecs_end = reshape(sol_fast.u[end][1:3*n],3,n)'
om_twist_end = sol_fast.u[end][3*n+1:end]
u_arr_end = vcat(reshape(rvecs_end',3*n,1), om_twist_end )
du_arr_end = zeros(size(u_arr_end))
velocity_fast!(du_arr_end,u_arr_end,params_fast,0.)
vvecs_end = reshape(du_arr_end[1:3*n], 3, n)'
v_end = sqrt.(sum(vvecs_end.^2,dims=2))
vvecs_end_plane = vvecs_end[:,1:2]
v_end_plane =  sqrt.(sum(vvecs_end_plane.^2,dims=2))
# plot first frame
display(Plots.plot(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal))

#display(plot(om_twist))
for (i, u) = enumerate(sol_fast.u)
    local rvecs = reshape(u[1:3*n], 3, n)'
    local om_twist = u[3*n+1:end]
    local t = sol_fast.t[i]

    display(Plots.plot!(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal,label="", color = :red))
    #display(plot!(om_twist))
end

# plot last frame
display(Plots.plot(rvecs_end[:,1],rvecs_end[:,2],rvecs_end[:,3],aspect_ratio=:equal))

# # stationary twist
c1 = rvecs_end[1,3]-rvecs_end[n,3]
c2 = -(rvecs_end[1,3]+rvecs_end[n,3])/2
l = range(-0.5,0.5,length=n)
om_twist_stat = -Cmrel/C * (  rvecs_end[:,3] + c1*l .+ c2  )

# plot last Omega3
plot(l,om_twist_end)
plot!(l,om_twist_stat)

println("maximum error in Omega3 = ", maximum(abs.((om_twist_stat - om_twist_end)/om_twist_stat)))


r_arrs = zeros(3*n,size(sol_fast.t,1))
for i = 1:size(sol_fast.t,1)
    r_arrs[:,i] = sol_fast.u[i][1:3*n]
end

om_twist_arrs = zeros(n,size(sol_fast.t,1))
for i = 1:size(sol_fast.t,1)
    om_twist_arrs[:,i] = sol_fast.u[i][3*n+1:end]
end

h5open("/home/andris/Documents/magnetic-filament/simulation_results/twist_test/n_$(n).h5", "w") do file
write(file, "n", n) 
write(file, "ts", sol_fast.t) 
write(file, "r_arrs", r_arrs) 
write(file, "om_twist_arrs", om_twist_arrs) 
write(file, "Cm", Cm) 
write(file, "Cmrel", Cmrel) 
write(file, "lambda", lambda)
write(file, "C", C)
write(file, "zeta_rot", zeta_rot)
end
