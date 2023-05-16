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

filedir = "/home/andris/Documents/magnetic-filament/simulation_results/ferromagnetic_rotating"
writedir = "/home/andris/Documents/magnetic-filament/simulation_results/ferromagnetic_rotating/velocity_fields"

for filename = glob("*.h5",filedir)
    println(filename)
    fid = h5open(filename, "r")

    n = read(fid, "n")
    h = 1/n
    Cm = read(fid, "Cm")
    omega = read(fid, "omega")
    rvecs = read(fid, "rvecs")
    vvecs = read(fid, "vvecs")
    tvecs = read(fid, "tvecs")
    hvec = read(fid, "hvec")
    t = read(fid, "t")
    t0 = read(fid, "t0")
    d = read(fid, "d")
    ϵ = read(fid, "rho")
    lambda = read(fid, "lambda")
    ζratio = 1- read(fid, "lambda")

    rvecs[:,3] = d*ones(size(rvecs[:,3]))

    # println(d)
    # println(rvecs)
    # fvecs = zeros(size(vvecs))
    # for i = 1:n

    # end
    
    vvecs_par = make_dot_product(vvecs,tvecs) .* tvecs
    vvecs_perp = vvecs - vvecs_par
    fvecs = 1/ζratio * vvecs_par + vvecs_perp # force denisity
    fvecs = fvecs * h # differential force

    function vfun_wall(x, y) 
        # only show velocities outside filament
    
        vvec = make_velocity_field_wall( [x, y, d], rvecs, fvecs, ϵ, d, h)
        dist, ind = find_distance_to_filament([x, y, d],rvecs)
        if  dist > ϵ
            return vvec[1], vvec[2]
        else
            return vvecs[ind,1], vvecs[ind,2]
        end
    end

    k = 37 # like in experiment
    if occursin("filament0", filename)
        maxrange = 0.9148064291632146 # like in experiment
    elseif occursin("filament1", filename)
        maxrange = 1.1108363782696178 # like in experiment
    elseif occursin("filament2", filename)
        maxrange = 1.4137917541613318 # like in experiment
    elseif occursin("filament3", filename)
        maxrange = 1.1108363782696178 # like in experiment
    end

    xs = Array(range(-maxrange,maxrange,length=k))
    ys = Array(range(-maxrange,maxrange,length=k))
    uswall = zeros( size(xs,1), size(ys,1) )
    vswall = zeros( size(xs,1), size(ys,1) )

    for i = 1:k
        for j = 1:k
            x = xs[i]
            y = ys[j]
    
            u,v = vfun_wall(x,y)
    
            uswall[i,j] = u
            vswall[i,j] = v
        end
    end

    h5open(writedir * filename[end-15:end], "w") do file
        write(file, "n", n) 
        write(file, "t", t) 
        write(file, "t0", t0) # in seconds
        write(file, "d", d) 
        write(file, "rho", ϵ) 
        write(file, "omega", omega) 
        write(file, "Cm", Cm) 
        write(file, "lambda", lambda)
        write(file, "rvecs", rvecs) 
        write(file, "vvecs", vvecs) 
        write(file, "tvecs", tvecs) 
        write(file, "hvec", hvec) 
        write(file, "xs", xs) 
        write(file, "ys", ys)
        write(file, "uswall", uswall) 
        write(file, "vswall", vswall) 
    end

end