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
using Glob

include("../../../functions/physics_functions.jl")
include("../../../functions/finite_difference_functions.jl")
include("../../../functions/filament_geometry_functions.jl")

writedir = "/home/andris/Documents/magnetic-filament/simulation_results/ferromagnetic_rotating/velocity_fields"

