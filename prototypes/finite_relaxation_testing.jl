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


# precession frequency multiplied by relaxation time
 ωτ = 0.5

# infinitely fast precessing field
 θmagic = 0.5*acos(-(1+2*ωτ^2)/(3+2*ωτ^2)) #magic angle
 θ = θmagic#+0.08 #+0.2 # precession angle

# torque density due to finite relaxation
 Cmrel0 = 20.
 Cmrel = Cmrel0 * ωτ/(1+ωτ^2) * sin(θ)

# Magnetic force
 Cm0 = 400.
 Cm = Cm0 * 1/(1+ωτ^2) * ( 1+2*ωτ^2 + (3+2*ωτ^2) * cos(2*θ) )

# twist elasticity over bending elasticity constant
 C = 1.

# anisotropy of drag
 lambda = -(2. - 1.)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag

# number of discretized elements
 n = 31#50#80 # number of points - each corresponds to an length element of a rod.
 h = 1/n # distance between 2 pts # takes into account the half a point extention on each side

# # initialize the shape
# rvecs = zeros(n,3)
# rvecs[:,1] = 0.01*range(-0.5,0.5,length=n)
# #rvecs[:,2] = range(-0.5,0.5,length=n)
# rvecs[:,3] = range(-0.5,0.5,length=n)

# # initialize the shape
l = range(-0.5,0.5,length=n)
rvecs = zeros(n,3)
rvecs[:,1] = 0.01*cosh.(range(-0.5,0.5,length=n)).^4 + 0.001*range(-0.5,0.5,length=n)
#rvecs[:,2] = range(-0.5,0.5,length=n)
rvecs[:,3] = range(-0.5,0.5,length=n)

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
  Mmat = sparse(make_Mmat_ceb(h,n))

# paramagnetic force matrix
  hvec = [0,0,1] # fast precessing field is effectively pointing in the ez direction
  Mmat_Fparamag = sparse(make_Mmat_Fparamag(h,n,hvec, Cm))

  total_Mmat = Mmat+Mmat_Fparamag

 params = [lambda, h, n, Cm, Cmrel, C, Mmat, Mmat_Fparamag, D1, D2]

function velocity!(du_arr,u_arr,params,t)
    lambda = params[1]
    h = params[2]
    n = params[3]
    Cm = params[4]
    Cmrel = params[5]
    C = params[6]
    Mmat = params[7]
    Mmat_Fparamag = params[8]
    D1 = params[9]
    D2 = params[10]
    
    rvecs = reshape(u_arr[:], 3, n)'
    r_arr = reshape(rvecs',3*n,1)
    

    # projection operator on inextensible motion
    P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    P = μ*P # dense matrix


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


 J = initialize_J_memory(n)
 μ = initialize_μ_memory(n)
 tmp3x1Float = [1.,1.,1.]
 tmp3x3Float = ones(3,3)
 tmp3Nx3NFloat = ones(3n,3n)
 tmp3Nx1Float = ones(3n)
 tmpNm1xNm1Float = ones(n-1,n-1)
 tmpNm1x1Float = ones(n-1)
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

    frelax_arr = make_frelax_arr(h,rvecs, D1)*h

    # twist
    c1 = rvecs[1,3]-rvecs[n,3]
    c2 = -(rvecs[1,3]+rvecs[n,3])/2
    l = range(-0.5,0.5,length=n)
    om_twist = -Cmrel/C * (  rvecs[:,3] + c1*l .+ c2  )
    ftwist_arr = make_ftwist_arr(h, om_twist, rvecs, D1, D2)*h
    
    total_force_arr = total_Mmat*r_arr + C*ftwist_arr + Cmrel*frelax_arr
    
    Λ =  Symmetric(J*μ*transpose(J))  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    du_arr[:] = μ*(total_force_arr + ftension_arr)*h^2
    # vvecs = reshape(du_arr[1:3*n], 3, n)'
    # du_arr[3*n+1:end] = 0*C / zeta_rot * D2om_twist + 0*make_dot_product(make_cross_product(D1*rvecs,D2_BC*rvecs), D1*vvecs)
end



u_arr = reshape(rvecs',3*n,1)
du_arr = zeros(size(u_arr))

# velocity!(du_arr,u_arr,params,0.);
# vvecs = reshape(du_arr,3,n)'
# plot(vvecs)

velocity_fast!(du_arr,u_arr,params_fast,0.);
vvecs = reshape(du_arr,3,n)'
# plot!(vvecs)
# plot(l,sum(vvecs.^2,dims=2).^(0.5))
plot!(l,sum(vvecs.^2,dims=2).^(0.5))

# t=0.

# @time velocity!(du_arr,u_arr,params,t)
# Profile.clear()
# @profile ( for _=1:10000; velocity!(du_arr,u_arr,params,t); end )
# @time ( for _=1:10000; velocity!(du_arr,u_arr,params,t); end )

#  tspan=[0., 0.01] # until 0.01 for testing
# prob = ODEProblem(velocity!,u_arr,tspan,params)
# prob_fast = ODEProblem(velocity_fast!,u_arr,tspan,params_fast)
#@time sol = solve(prob,alg_hints=[:stiff],saveat=0.01/10,reltol=1e-8, abstol=1e-8) # 82.578174 seconds # 79.163289 seconds if everything is declared as const # 122.920989 s, 266.82 M allocations: 218.478 GiB  on work station 
# @time sol_fast = solve(prob_fast,alg_hints=[:stiff],saveat=0.01/10,reltol=1e-8, abstol=1e-8)
# ;

