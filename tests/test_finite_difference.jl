using Plots
using LinearAlgebra
using DiffEqOperators
using Elliptic
using Roots
using Rotations
using StaticArrays
using HDF5
using DifferentialEquations
using Test

include("../functions/finite_difference_functions.jl")

function test_D1()
    n=10
    rvecs = zeros(n,3)
    phis = range(0, pi, length=n)
    rvecs[:,1] = 1/pi * cos.(phis)
    rvecs[:,2] = 1/pi * sin.(phis)
    h = 1/(n-1) # approximately

    D1 = make_D1(rvecs) / h

    Drvecs = zeros(n,3)
    Drvecs[:,1] = ones(n)
    Drvecs[:,2] = 2*rvecs[:,1]
    
    @test D1*rvecs ≈ Drvecs
end
test_D1()


function test_D2()
    n=10
    rvecs = zeros(n,3)
    rvecs[:,1] = range(-0.5,0.5,length=n)
    rvecs[:,2] = rvecs[:,1].^3
    h = 1/(n-1) # approximately

    D2 = make_D2(rvecs) / h^2

    D2rvecs = zeros(n,3)
    D2rvecs[:,2] = 2*3*rvecs[:,1]
    
    @test (D2*rvecs)[2:end-1,:] ≈ D2rvecs[2:end-1,:]
end
test_D2()