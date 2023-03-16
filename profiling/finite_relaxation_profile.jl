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
const ωτ = 0.1

# infinitely fast precessing field
const θmagic = 0.5*acos(-(1+2*ωτ^2)/(3+2*ωτ^2)) #magic angle
const θ = θmagic+0.025 #+0.2 # precession angle

# torque density due to finite relaxation
const Cmrel0 = 400.
const Cmrel = 150. #Cmrel0 * ωτ/(1+ωτ^2) * sin(θ)

# Magnetic force
const Cm0 = 400.
const Cm = -80. #Cm0 * 1/(1+ωτ^2) * ( 1+2*ωτ^2 + (3+2*ωτ^2) * cos(2*θ) )

# twist elasticity over bending elasticity constant
const C = 1.

# anisotropy of drag
const lambda = -(2. - 1.)#-(make_ζratio_wall(d,ϵ) - 1)  # -(zeta_perp / zeta_par - 1) : anisotropy of drag

# number of discretized elements
const n = 31#50#80 # number of points - each corresponds to an length element of a rod.
const h = 1/n # distance between 2 pts # takes into account the half a point extention on each side

# # initialize the shape
# rvecs = zeros(n,3)
# rvecs[:,1] = 0.01*range(-0.5,0.5,length=n)
# #rvecs[:,2] = range(-0.5,0.5,length=n)
# rvecs[:,3] = range(-0.5,0.5,length=n)

# # initialize the shape
# const rvecs = zeros(n,3)
# const rvecs[:,1] = 0.01*sin.(2*pi*range(-0.5,0.5,length=n)) #0.001*cosh.(range(-0.5,0.5,length=n)).^4 + 0.001*range(-0.5,0.5,length=n)
# #rvecs[:,2] = range(-0.5,0.5,length=n)
# const rvecs[:,3] = range(-0.5,0.5,length=n)

# initialize the shape
global rvecs = zeros(n,3)
phi = range(0,pi*1.9,length=n)
global rvecs[:,1] = range(0.9,1,length=n) .* cos.(phi)#range(-0.5,0.5,length=n)
rvecs[:,2] = range(1,0.9,length=n) .* sin.(phi)#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
rvecs[:,3] = range(-0.5,0.5,length=n) #0.001 * sin.( pi*range(-0.5,0.5,length=n) ) #d*ones(size(rvecs[:,3]))


renormalize_length!(rvecs)
const rvecs = rvecs .- make_center_of_mass(rvecs)'

plot(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal)


# differentiation matrices
const D1 = make_D1(rvecs)/h
const D2_BC = make_D2_BC(rvecs)/h^2 # with boundary condition 0 at ends
const D2 = make_D2(rvecs)/h^2

# differentiation matrix for elastic force
const Mmat = sparse(make_Mmat_ceb(h,n))

# paramagnetic force matrix
const hvec = [0,0,1] # fast precessing field is effectively pointing in the ez direction
const Mmat_Fparamag = sparse(make_Mmat_Fparamag(h,n,hvec, Cm))

const total_Mmat = Mmat+Mmat_Fparamag

const params = [lambda, h, n, Cm, Cmrel, C, Mmat, Mmat_Fparamag, D1, D2]

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


const J = initialize_J_memory(n)
const μ = initialize_μ_memory(n)
const tmp3x1Float = [1.,1.,1.]
const tmp3x3Float = ones(3,3)
const tmp3Nx3NFloat = ones(3n,3n)
const tmp3Nx1Float = ones(3n)
const tmpNm1xNm1Float = ones(n-1,n-1)
const tmpNm1x1Float = ones(n-1)
const params_fast = [lambda, h, n, Cm, Cmrel, C, Mmat, Mmat_Fparamag, D1, D2, J, μ, tmp3x1Float,tmp3x3Float, total_Mmat]

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
    
    total_force_arr = total_Mmat*r_arr + C*ftwist_arr + Cmrel*frelax_arr
    
    Λ =  Symmetric(J*μ*transpose(J))  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    du_arr[:] = μ*(total_force_arr + ftension_arr)
    # vvecs = reshape(du_arr[1:3*n], 3, n)'
    # du_arr[3*n+1:end] = 0*C / zeta_rot * D2om_twist + 0*make_dot_product(make_cross_product(D1*rvecs,D2_BC*rvecs), D1*vvecs)
end

function force_vel()
    total_force_arr = (Mmat + Mmat_Fparamag)*u_arr + C*u_arr + Cmrel*u_arr

    Λ =  J*μ*transpose(J)  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    du_arr = μ*(total_force_arr + ftension_arr)
end

function force_vel!(tmpNm1x1Float,tmpNm1xNm1Float)
    total_force_arr = tmp3Nx1Float
    total_force_arr[:] = total_Mmat*u_arr + C*u_arr + Cmrel*u_arr

    Λ =  J*μ*transpose(J)  \ (-J*μ*total_force_arr)  # lagrange multiplier
    ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    du_arr = μ*(total_force_arr + ftension_arr)
end

function lambdasol()
    Λ =  J*μ*transpose(J)  \ (-J*μ*u_arr)  # lagrange multiplier
    # ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    # du_arr = μ*(total_force_arr + ftension_arr)
end

function lambdasol!(tmpNm1x1Float,tmpNm1xNm1Float)
    JmJT = J*μ*transpose(J) 
    Jmf = (-J*μ*u_arr)
    

    Λ =  Symmetric(JmJT) \ Jmf  # lagrange multiplier
    # ftension_arr = transpose(J)*Λ # tension force for inextensible filament

    # du_arr = μ*(total_force_arr + ftension_arr)
end

const u_arr = reshape(rvecs',3*n,1)
# du_arr = zeros(size(u_arr))
# t=0.

# @time velocity!(du_arr,u_arr,params,t)
# Profile.clear()
# @profile ( for _=1:10000; velocity!(du_arr,u_arr,params,t); end )
# @time ( for _=1:10000; velocity!(du_arr,u_arr,params,t); end )

tend = 0.01
const tspan=[0., tend] # until 0.01 for testing
prob = ODEProblem(velocity!,u_arr,tspan,params)
prob_fast = ODEProblem(velocity_fast!,u_arr,tspan,params_fast)
#@time sol = solve(prob,alg_hints=[:stiff],saveat=0.01/10,reltol=1e-8, abstol=1e-8) # 82.578174 seconds # 79.163289 seconds if everything is declared as const # 122.920989 s, 266.82 M allocations: 218.478 GiB  on work station 
@time sol_fast = solve(prob_fast,alg_hints=[:stiff],saveat=tend/30,reltol=1e-8, abstol=1e-8)
;
# @time sol_fast = solve(prob_fast,QNDF(autodiff=false),saveat=0.01/10,reltol=1e-8, abstol=1e-8) # 82.578174 seconds # 79.163289 seconds if everything is declared as const # 122.920989 s on work station
# ;
#@time sol = solve(prob,alg_hints=[:stiff],saveat=0.01/10,reltol=1e-8, abstol=1e-8) # 82.578174 seconds # 79.163289 seconds if everything is declared as const
#@time sol = solve(prob,alg_hints=[:stiff],saveat=0.01/10,reltol=1e-6, abstol=1e-6) # very slight increase in speed  73.521214 seconds
#@time sol = solve(prob,alg_hints=[:stiff],reltol=1e-8, abstol=1e-8) # save at costs basically nothing   85.137595 seconds
#@time sol = solve(prob,alg_hints=[:stiff],saveat=0.01/10,reltol=1e-6, abstol=1e-3) # default tolerances 

# Profile.print(noisefloor=2,combine = true,sortedby=:count,format=:flat,mincount=50)

# using ProfileView

# @profview ( for _=1:100; velocity!(du_arr,u_arr,params,t); end )


# Profile.clear()
# @profile ( for _=1:10000; make_proj_operator_mobility_tensor(rvecs, lambda, h); end )
# Profile.print(noisefloor=2,combine = true,sortedby=:count,format=:flat,mincount=50)

#readline()


# https://docs.julialang.org/en/v1/manual/performance-tips/#Measure-performance-with-[@time](@ref)-and-pay-attention-to-memory-allocation


rvecs_end=reshape(sol_fast.u[end],3,n)'
#plot(rvecs_end[:,1],rvecs_end[:,2],rvecs_end[:,3],aspect_ratio=:equal)
cms = zeros(size(sol_fast.u,1),3)
for (i, u) = enumerate(sol_fast.u)
    local rvecs = reshape(u[1:3*n], 3, n)'
    local t = sol_fast.t[i]
    #local hvec = -normalize([cos(omega*t),sin(omega*t),1/sqrt(2)])
    #println(t/T)
    display(Plots.plot!(rvecs[:,1],rvecs[:,2],rvecs[:,3],aspect_ratio=:equal,label="", color = :red))
    cms[i,:] = make_center_of_mass(rvecs)
    
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
