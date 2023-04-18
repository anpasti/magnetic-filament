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
using Random

global rng = MersenneTwister(1234) # intialize random number generator

include("../functions/physics_functions.jl")
include("../functions/finite_difference_functions.jl")
include("../functions/filament_geometry_functions.jl")


Cms = [-75, -250]*1.
for Cm = Cms
if Cm == 0
    Cmrels = [300, 200, 150, 125, 100, 75, 50, 25]*1.
else
    Cmrels = abs(Cm)*[0.9, 1., 1.25, 1.5, 1.75, 2., 3., 4.]
end
for Cmrel = Cmrels
for starting_shape = 1:6 # 3 horizontal, 3 vertical
# magnetic force
#Cm = -100.
# magnetic force from finite relaxation
#Cmrel = 120.
# twist elasticity over bending elasticity constant
C = 1.

# anisotropy of drag
lambda = -(2. - 1.)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag

# number of discretized elements
n = 40#50#80 # number of points - each corresponds to a length element of a rod.
h = 1/n # distance between 2 pts # takes into account the half a point extention on each side

# initialize the shape
rvecs = zeros(n,3)
if starting_shape < 4
    rvecs[:,1] = range(-0.5,0.5,length=n)  # horizontal
    println("shape ",starting_shape,": horizontal")
else 
    rvecs[:,3] = range(-0.5,0.5,length=n)  # vertical
    println("shape ",starting_shape,": vertical")
end
println("applying noise with ", rng)
rvecs += 0.001*randn(rng,(n,3))

renormalize_length!(rvecs) # scale
rvecs = rvecs .- make_center_of_mass(rvecs)' # move center of mass to 0

#plot(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal)

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

params_fast = [lambda, h, n, Cm, Cmrel, C, Mmat, Mmat_Fparamag, D1, D2, J, μ, tmp3x1Float,tmp3x3Float, total_Mmat]

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

    
    r_arr = u_arr #@view u_arr[:]
    rvecs = reshape(r_arr, 3, n)'
    

    # # projection operator on inextensible motion
    # P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    # P = μ*P # dense matrix
    make_J!(J,rvecs)
    make_mobility_tensor!(μ, tmp3x1Float, tmp3x3Float , rvecs, lambda, h)

    # finite magentic relaxation force

    frelax_arr = make_frelax_arr(h,rvecs, D1)

    # twist
    c1 = rvecs[1,3]-rvecs[n,3]
    c2 = -(rvecs[1,3]+rvecs[n,3])/2
    l = range(-0.5,0.5,length=n)
    om_twist = -Cmrel/C * (  rvecs[:,3] + c1*l .+ c2  )
    ftwist_arr = make_ftwist_arr(h, om_twist, rvecs, D1, D2)
    
    # froce without tension
    total_force_arr = total_Mmat*r_arr + C*ftwist_arr + Cmrel*frelax_arr
    
    Λ =  Symmetric(J*μ*transpose(J))  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    du_arr[:] = μ*(total_force_arr + ftension_arr)
    # vvecs = reshape(du_arr[1:3*n], 3, n)'
    # du_arr[3*n+1:end] = 0*C / zeta_rot * D2om_twist + 0*make_dot_product(make_cross_product(D1*rvecs,D2_BC*rvecs), D1*vvecs)
end

# set up ODE problem
u_arr = reshape(rvecs',3*n,1)
tend = min( 10*maximum([abs(1/Cm),1/Cmrel]), 1.)
tspan=[0., tend]
prob_fast = ODEProblem(velocity_fast!,u_arr,tspan,params_fast)

# solve ODE problem
println("calculating: Cm =", Cm, " Cmrel = ", Cmrel )
@time sol_fast = solve(prob_fast,alg_hints=[:stiff],saveat=tend/50,reltol=1e-8, abstol=1e-8)


r_arrs = zeros(3*n,size(sol_fast.t,1))
for i = 1:size(sol_fast.t,1)
    r_arrs[:,i] = sol_fast.u[i]
end

h5open("/home/andris/Documents/magnetic-filament/simulation_results/phase_space_precessing_rng/Cm_$(Cm)_Cmrel_$(Cmrel)_shape_$(starting_shape).h5", "w") do file
write(file, "n", n) 
write(file, "ts", sol_fast.t) 
write(file, "r_arrs", r_arrs) 
write(file, "Cm", Cm) 
write(file, "Cmrel", Cmrel) 
write(file, "lambda", lambda)
write(file, "C", C)
end


end
end
end
