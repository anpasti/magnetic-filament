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
global n = 50#50#80 # number of points - each corresponds to an length element of a rod.
global h = 1/n # distance between 2 pts # takes into account the half a point extention on each side


println("*****************************")
println("*****************************")
println("*****************************")
println("Cm = ", Cm)
println("omega = ", omega)
println("ζratio = ", -lambda+1 )
println("n = ", n)

# initialize the shaped
global rvecs = zeros(n,3)
global rvecs[:,1] = range(-0.5,0.5,length=n)
#rvecs[:,2] = 0.01 * cos.( pi*rvecs[:,1] )#exp.( -(rvecs[:,1]).^2 / 0.2^2 )
#rvecs[:,3] = 0.01 * sin.( pi*rvecs[:,1] ) #d*ones(size(rvecs[:,3]))
renormalize_length!(rvecs)

# differentiation matrix and magentic force
global Mmat = make_Mmat_ceb(h,n)

params = [lambda, h, n, omega, Cm, Mmat]

function velocity_prec!(dr_arr,r_arr,params,t)
    lambda = params[1]
    h = params[2]
    n = params[3]
    omega = params[4]
    Cm = params[5]
    Mmat = params[6]
    T = 2*pi / abs(omega)

    rvecs = reshape(r_arr, 3, n)'
    P, μ = make_proj_operator_mobility_tensor(rvecs, lambda, h)
    P = μ*P
    hvec = -normalize([cos(omega*t),sin(omega*t),1/sqrt(2)]) # magic angle
    
    Mmat_Fparamag = make_Mmat_Fparamag(h,n,hvec, Cm)

    dr_arr[:] = P*((Mmat + Mmat_Fparamag)*r_arr)
end

# plot first frame
display(Plots.plot(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal))
for period_n = 1:5 # run 5 periods, renormalize length every period
    global rvecs
    println("period = ", period_n)
    tspan = (T*(period_n-1),T*period_n)
    r_arr0 = reshape(rvecs',3*n,1)
    prob = ODEProblem(velocity_prec!,r_arr0,tspan,params)
    @time global sol = solve(prob,alg_hints=[:stiff],saveat=T/10,reltol=1e-12, abstol=1e-12)

    for (i, u) = enumerate(sol.u)
        local rvecs = reshape(u, 3, n)'
        local t = sol.t[i]
        local hvec = -normalize([cos(omega*t),sin(omega*t),1/sqrt(2)])
        #println(t/T)
        display(Plots.plot!(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal,label="", color = :red))
        display(Plots.plot!([0,hvec[1]/4],[0,hvec[2]/4],arrow=true,color=:black,linewidth=2,label=""))
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
    global hvec = -normalize([cos(omega*t),sin(omega*t),1/sqrt(2)])
    #renormalize length
    global rvecs = reshape(sol.u[end], 3, n)'
    println(make_length_of_filament(rvecs))
    renormalize_length!(rvecs)

end

rvecsend = reshape(sol.u[end], 3, n)'
Plots.plot(rvecsend[:,1],rvecsend[:,2],rvecsend[:,3],aspect_ratio=:equal,label="", color = :red)
display(Plots.plot!([0,hvec[1]/4],[0,hvec[2]/4], [0,hvec[3]/4],arrow=true,color=:black,linewidth=2,label=""))

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
