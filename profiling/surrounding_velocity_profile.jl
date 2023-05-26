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

include("../functions/physics_functions.jl")
include("../functions/finite_difference_functions.jl")
include("../functions/filament_geometry_functions.jl")

# const Fvec = [1.,1.,0.] # z component must be 0
# const rvec = [1.,1.,0.3] # height iz the z component
# const rvec0 = [-1.,-1.,0.3]

# make_stokeslet_velocity_wall(Fvec,rvec,rvec0)

# make_doublet_velocity_field_wall(Fvec,rvec,rvec0)



function make_stokeslet_velocity_wall_fast!(vvec, Fvec,rvec,rvec0,tmp3x1Float1,tmp3x1Float2)
    # stokeslet velocity near a wall. see Blake and Chwang (1973).
    # evaluated at r due to force F at r0
    # F_z must be 0
    # wall is assumed to be at z=0

    Rvec = tmp3x1Float1
    @. Rvec = rvec - rvec0
    R = norm(Rvec)
    Rimvec = tmp3x1Float2
    @. Rimvec = rvec - rvec0 # image system radius vector
    h = rvec0[3] # height above wall
    Rimvec[3] += 2*h
    Rim = norm(Rimvec)#sqrt( R^2 + 4*h^2 )

    @. vvec = [0.,0.,0.]
    for i = 1:3
        for j = 1:2 # F_z=0
            vvec[i] += 
            Fvec[j]*(
             (Rvec[i]*Rvec[j]/R^3) 
            -(Rimvec[i]*Rimvec[j]/Rim^3)
            +2*h/Rim^3*
                (
                 -3*Rimvec[i]*Rimvec[j]/Rim^2 * (h-Rimvec[3])
                )
            )
            if i == j
                vvec[i] += Fvec[j]*(
                1/R - 1/Rim
                +2*h/Rim^3*
                    (
                     (h-Rimvec[3])
                    )
                )
            end
            if i==3
                vvec[i] += Fvec[j]*(
                    2*h/Rim^3*Rimvec[j]
                )
            end

        end
    end


    return  nothing
    # corresponds with Mathematica
end


function make_doublet_velocity_field_wall_fast!(vvec,Fvec,rvec,rvec0,tmp3x1Float1,tmp3x1Float2)
    # source doublet velocity near a wall. see Blake and Chwang (1973).
    # evaluated at r due to source F at r0
    # F_z must be 0
    # wall is assumed to be at z=0

    Rvec = tmp3x1Float1
    @. Rvec = rvec - rvec0
    R = norm(Rvec)
    z=rvec[3]
    Rimvec = tmp3x1Float2
    @. Rimvec = rvec - rvec0 # image system radius vector
    h = rvec0[3] # height above wall
    Rimvec[3] += 2*h
    Rim = norm(Rimvec) # sqrt( R^2 + 4*h^2 )

    @. vvec = [0.,0.,0.]
    for i = 1:3
        for j = 1:2 # F_z=0
            vvec[i] += 
            Fvec[j]*(
             ( - 3*Rvec[i]*Rvec[j]/R^5) 
            -( - 3*Rimvec[i]*Rimvec[j]/Rim^5)
            )
            if i == j
                vvec[i] += Fvec[j]*(
                    1/R^3 - 1/Rim^3
                )
            end
        end

        if i == 3
            for α=1:2
                vvec[i] -= 
                Fvec[α]*6*Rimvec[α]*Rimvec[3]/Rim^5
            end
        end

        for α=1:2
            for β=1:2
                if i == β
                    vvec[i] -= 
                    2*Fvec[α]*
                    (
                    15*Rimvec[3]*Rimvec[α]*Rimvec[β]*z/Rim^7
                    )
                    if α == β
                        vvec[i] -= 
                        2*Fvec[α]*
                        (
                        -3*Rimvec[3]*z/Rim^5 
                        )
                    end
                end
            end
        end

        if i == 3
            for α=1:2
                vvec[i] -= 
                2*Fvec[α]*
                (
                -3*Rimvec[α]*z/Rim^5
                +15*Rimvec[3]^2*z*Rimvec[α]/Rim^7
                )
            end
        end

    end

    return nothing
    # corresponds with Mathematica
end




function stokeslet_test(N)
    Fvec = [3.,2.,0.] # z component must be 0
    rvec = [1.,2.,3.] # height iz the z component
    rvec0 = [-1.,-1.,1]
    vvec = [0.,0.,0.]
    tmp1 = [0.,0.,0.]
    tmp2 = [0.,0.,0.]
    R = 0.
    Rim = 0.
    for _ = 1:N
        make_stokeslet_velocity_wall(Fvec,rvec,rvec0)
    end
    return make_stokeslet_velocity_wall(Fvec,rvec,rvec0)
end

function stokeslet_fast_test(N)
    Fvec = [3.,2.,0.] # z component must be 0
    rvec = [1.,2.,3.] # height iz the z component
    rvec0 = [-1.,-1.,1]
    vvec = [0.,0.,0.]
    tmp1 = [0.,0.,0.]
    tmp2 = [0.,0.,0.]
    R = 0.
    Rim = 0.
    for _ = 1:N
        make_stokeslet_velocity_wall_fast(Fvec,rvec,rvec0)
    end
    return make_stokeslet_velocity_wall_fast(Fvec,rvec,rvec0)
end

function stokeslet_very_fast_test(N)
    Fvec = [3.,2.,0.] # z component must be 0
    rvec = [1.,2.,3.] # height iz the z component
    rvec0 = [-1.,-1.,1]
    vvec = [0.,0.,0.]
    tmp1 = [0.,0.,0.]
    tmp2 = [0.,0.,0.]
    R = 0.
    Rim = 0.
    for _ = 1:N
        make_stokeslet_velocity_wall_fast!(vvec,Fvec,rvec,rvec0,tmp1,tmp2)
    end
    return vvec
end


@time out1 = stokeslet_test(10^6)
@time out2 = stokeslet_very_fast_test(10^6)

function doublet_test(N)
    Fvec = [1.,2.,0.] # z component must be 0
    rvec = [1.,2.,3.] # height is the z component
    rvec0 = [-1.,-1.,0.3]
    for _ = 1:N
        make_doublet_velocity_field_wall(Fvec,rvec,rvec0)
    end
    return make_doublet_velocity_field_wall(Fvec,rvec,rvec0)
end

function doublet_fast_test(N)
    Fvec = [1.,2.,0.] # z component must be 0
    rvec = [1.,2.,3.] # height iz the z component
    rvec0 = [-1.,-1.,0.3]
    vvec = [0.,0.,0.]
    tmp1 = [0.,0.,0.]
    tmp2 = [0.,0.,0.]
    R = 0.
    Rim = 0.
    for _ = 1:N
        make_doublet_velocity_field_wall_fast!(vvec,Fvec,rvec,rvec0,tmp1,tmp2)
    end
    return vvec
end


@time out3 = doublet_test(10^6);
@time out4 = doublet_fast_test(10^6);

println("good fast stokeslet: ", out1 ≈ out2 )
println("good fast doublet: ",out3 ≈ out4)