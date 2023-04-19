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


include("../functions/physics_functions.jl")
include("../functions/finite_difference_functions.jl")
include("../functions/filament_geometry_functions.jl")

# magnetoelastic number
Cm = 100
# dimensionless angular velocity of field
omega = 10*Cm
# anisotropy of drag
lambda = -(2. - 1.) #-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag

# number of discretized elements
n = 80#50#80 # number of points - each corresponds to a length element of a rod.
h = 1/n # distance between 2 pts # takes into account the half a point extention on each side


# initialize the shape
rvecs = zeros(n,3)
#phi = range(0,pi*2.,length=n)
rvecs[:,1] = range(-0.5,0.5,length=n) # cos.(phi)#range(-0.5,0.5,length=n)
# rvecs[:,2] = sin.(phi)#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
rvecs[:,3] = 0.01*range(-0.5,0.5,length=n) #1000*range(-0.5,0.5,length=n) #0.001 * sin.( pi*range(-0.5,0.5,length=n) ) #d*ones(size(rvecs[:,3]))

renormalize_length!(rvecs) # scale
rvecs = rvecs .- make_center_of_mass(rvecs)' # move center of mass to 0
println("start length = ", make_length_of_filament(rvecs))

# differentiation matrix for elastic force
Mmat = sparse(make_Mmat_ceb(h,n))


# allocate some memory
J = initialize_J_memory(n)
μ = initialize_μ_memory(n)
tmp3x1Float = [1.,1.,1.]
tmp3x3Float = ones(3,3)

params_fast = [lambda, h, n, Cm, omega, J, μ, tmp3x1Float,tmp3x3Float, Mmat]

function velocity_fast!(du_arr,u_arr,params,t)
    lambda = params[1]
    h = params[2]
    n = params[3]
    Cm = params[4]
    omega = params[5]
    J = params[6]
    μ = params[7]
    tmp3x1Float = params[8]
    tmp3x3Float = params[9]
    Mmat = params[10]

    
    r_arr = u_arr #@view u_arr[:]
    rvecs = reshape(r_arr, 3, n)'
    

    # # projection operator on inextensible motion
    # P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    # P = μ*P # dense matrix
    make_J!(J,rvecs)
    make_mobility_tensor!(μ, tmp3x1Float, tmp3x3Float , rvecs, lambda, h)

    # magentic force
    hvec = -[cos(omega*t),sin(omega*t),0] # magnetic field direction unit vector
    fm_arr = make_fm_arr(n, hvec, Cm)

    
    # froce without tension
    total_force_arr = Mmat*r_arr + fm_arr
    
    Λ =  Symmetric(J*μ*transpose(J))  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    du_arr[:] = μ*(total_force_arr + ftension_arr)
    # vvecs = reshape(du_arr[1:3*n], 3, n)'
    # du_arr[3*n+1:end] = 0*C / zeta_rot * D2om_twist + 0*make_dot_product(make_cross_product(D1*rvecs,D2_BC*rvecs), D1*vvecs)
end


# set up ODE problem
u_arr = reshape(rvecs',3*n,1)
#tend = min( 5*abs(1/Cm), 1.)
tend = 2*pi/omega * 5
tspan=[0., tend]
prob_fast = ODEProblem(velocity_fast!,u_arr,tspan,params_fast)

# solve ODE problem
println("calculating: Cm = ", Cm, ", omega = ", omega, ", n = ", n)
@time sol_fast = solve(prob_fast,alg_hints=[:stiff],saveat=tend/50,reltol=1e-8, abstol=1e-8)


rvecs_end=reshape(sol_fast.u[end],3,n)'
plot(rvecs_end[:,1],rvecs_end[:,2],rvecs_end[:,3],aspect_ratio=:equal)
# plot results
for (i, u) = enumerate(sol_fast.u)
    local rvecs = reshape(u[1:3*n], 3, n)'
    local t = sol_fast.t[i]
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

println("end length = ", make_length_of_filament(rvecs_end))