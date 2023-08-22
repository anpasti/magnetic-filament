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

# make the initial shape

n = 250
h = 1/n # distance between 2 pts 
ρ = h/sqrt(exp(1)) # radius of filament
d = 0.06 # height
println("ρ/L = ", ρ)
println("h/ρ = ", d/ρ)


# # initialize the shape
rvecs = zeros(n,3)
rvecs[:,1] = range(-0.5,0.5,length=n)
# rvecs[:,2] = range(-0.5,0.5,length=n).^3
# rvecs[:,3] = d*ones(size(rvecs[:,3]))
renormalize_length!(rvecs)
rvecs = rvecs .- make_center_of_mass(rvecs)'
rvecs[:,3] = d*ones(size(rvecs[:,3]))

display(Plots.scatter(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal))

tvecs = make_tvecs(rvecs)

larray = range(-0.5,0.5,length=n) # arclength

integralMat = zeros(3n,3n)
for y = 1:n
    rvec = rvecs[y,:]
    for x = 1:n
        if x != y
            rvec0 = rvecs[x,:]
            oseen_tensor = make_stokeslet_wall(rvec,rvec0)
            for j = 1:3
                for i = 1:3
                    integralMat[3*(y-1)+j, 3*(x-1)+i] = 1/(8*pi) * (
                        oseen_tensor[i,j]
                    )*h
                end #end i
            end #end j
        else
            for j = 1:3
                for i = 1:3
                    if i == j
                        integralMat[3*(y-1)+j, 3*(x-1)+i] = 1/(4*pi) * (
                            1 - tvecs[x,i]*tvecs[y,j]
                        )
                    else
                        integralMat[3*(y-1)+j, 3*(x-1)+i] = 1/(4*pi) * (
                            - tvecs[x,i]*tvecs[y,j]
                        )
                    end
                end #end i
            end #end j
        end
    end #end x
end #end y



ωvec = [0.,0.,1.]
vvecs = zeros(size(rvecs))

for i = 1:n # rotating velocity
    vvecs[i,:] = cross(ωvec,rvecs[i,:])
end


varr = reshape(vvecs',3*n,1)

farr = integralMat \ varr
fvecs = reshape(farr[1:3*n], 3, n)'


E10 = 2*asinh(1/(4*d))
E20 = 1/2/sqrt( 1+16*d^2 )
E30 = 1/2/sqrt( 1+16*d^2 )^3
α10 = log( (d+sqrt(d^2-ρ^2))/ρ )


E1s = asinh.((1 .- 2*larray)/(4*d)) + asinh.((1 .+ 2*larray)/(4*d))
E2s = (1 .- 2*larray) ./ ( 4*sqrt.( 16*d^2 .+ (1 .- 2*larray).^2 ) ) +
      (1 .+ 2*larray) ./ ( 4*sqrt.( 16*d^2 .+ (1 .+ 2*larray).^2 ) )
E3s = (1 .- 2*larray).^3 ./ ( 4*sqrt.( 16*d^2 .+ (1 .- 2*larray).^2 ).^3 ) + 
      (1 .+ 2*larray).^3 ./ ( 4*sqrt.( 16*d^2 .+ (1 .+ 2*larray).^2 ).^3 )
ρfun = ρ *sqrt.(1 .- 4*larray.^2) #shape of filament crossection radius
α1s = log.( (d .+ sqrt.(d^2 .- ρfun.^2)) ./ ρ )


ζperp0 = 8π / ( log(1/(4*d^2)) +1 -E10 - 2*E20 +2*α10   )
ζperp_inf = 4π / (log(1/ρ) + 1/2)
ζperps = 8π ./ ( log.( (1 .- 4*larray.^2)/(4*d^2) ) .+ 1 .- E1s .- 2*E2s .+ 2*α1s   )


display(Plots.plot(fvecs[:,2])) # only force in x direction
display(Plots.plot!(vvecs[:,2] .* ζperps)) # drag coefficient
display(Plots.plot!(vvecs[:,2]*ζperp0)) # drag coefficient constant
# display(Plots.plot!(vvecs[:,2]*ζperp_inf)) # drag coefficient constant, d->infinity 