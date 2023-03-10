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

include("../functions/filament_geometry_functions.jl")



function test_cross_product()
    a = zeros(10,3)
    a[:,1] = ones(10)
    b = zeros(10,3)
    b[:,2] = ones(10)

    c = zeros(10,3)
    c[:,3] = ones(10)

   @test make_cross_product(a,b) == c
end
test_cross_product()

function test_dot_product()
    a = zeros(10,3)
    a[:,1] = ones(10)
    b = zeros(10,3)
    b[:,2] = ones(10)
    b[:,3] = ones(10)

    c = zeros(10)

   @test make_dot_product(a,b) == c
end
test_dot_product()