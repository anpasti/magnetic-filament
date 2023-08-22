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
using Test

include("../functions/physics_functions.jl")
include("../functions/finite_difference_functions.jl")
include("../functions/filament_geometry_functions.jl")


function test_relaxation_velocity()
    #
    n = 50
    h = 1/n
    lamdba = -(2-1)

    # initialize the shape
    A = 0.01
    rvecs = zeros(n,3)
    rvecs[:,1] = range(-0.5,0.5,length=n)
    rvecs[:,2] = A * cos.( pi*rvecs[:,1] )

    # expected velocity
    vexp = -pi^4 * A * cos.( pi*rvecs[:,1] )

    # calculate velocity
    #μ = make_mobility_tensor(rvecs,lamdba,h)
    P, μ = make_proj_operator_mobility_tensor(rvecs,lamdba,h)
    Mmat = make_Mmat_ceb(h,n)
    r_arr = reshape(rvecs',3*n,1)
    f_arr = Mmat*r_arr
    v_arr = μ*P*f_arr
    vvecs = reshape(v_arr, 3, n)'

    display(plot(vexp))
    display(plot!(vvecs))

    println()

    @test vvecs[:,2] ≈ vexp
end

test_relaxation_velocity()


