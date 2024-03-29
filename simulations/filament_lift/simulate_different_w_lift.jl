using Plots
using LinearAlgebra
using DiffEqOperators
using Elliptic
using Roots
using Rotations
using StaticArrays
using HDF5
using DifferentialEquations
using DelimitedFiles

include("../../functions/physics_functions.jl")
include("../../functions/finite_difference_functions.jl")
include("../../functions/filament_geometry_functions.jl")

global n = 301#round(Int, 60*radius_scale)+1
global h = 1/(n-1) # distance between 2 pts 

wintegers = 2:2:Int64(1/h) # 4w/h should be integer # visually looks good

ws = h/2 * wintegers # from 0 to 0.5
ds = zeros(0)
Flifts = zeros(0)
Torques = zeros(0)
Torques_rod = zeros(0)
for w = ws

# make the initial shape
regλ = 0.1 #Tikhonov regularization parameter https://docs.juliahub.com/RegularizationTools/W7b5l/0.2.0/theory/theory/
global ρ = 2*h/sqrt(exp(1)) # radius of filament

d = 0.1 # height


#wint = 12*radius_scale # how many points in the middle section
# global w = 0.1#wint * h / 2 
r0 = exp(-pi/2)*w
θmax = log( exp(pi/2) * (1-2*w+2*sqrt(2)*w) / (2*sqrt(2)*w) )
armlength = 1/2 - w

println("n = ",n)
println("ρ/L = ", ρ)
println("h/ρ = ", d/ρ)
println("w = ", w)
println("height = ",d)



# # initialize the shape
rvecs = zeros(n,3)
# rvecs[:,1] = range(-0.5,0.5,length=n)
# rvecs[:,2] = range(-0.5,0.5,length=n).^3
# # rvecs[:,3] = d*ones(size(rvecs[:,3]))

# a convuluted function to make the shape
θ = θmax
for i = 1:n # first arm
    #println(θ)
    if θ < pi/2 + 0.001
        ycoord = w 
        for k = i:n # middle section
            #println(ycoord)
            rvecs[k,1] = 0.
            rvecs[k,2] = ycoord
            
            ycoord -= h

            if ycoord < -w*1.00001 

                for j = k:n # second arm
                    rvecs[j,1] = -rvecs[i-j+k,1]
                    rvecs[j,2] = -rvecs[i-j+k,2]
                end

                break
            end
        end

        break
    end

    rvecs[i,1] = r0 * exp(θ) * cos(θ)
    rvecs[i,2] = r0 * exp(θ) * sin(θ)

    θ = log( -h/sqrt(2)/r0 + exp(θ) )
end


# renormalize_length2!(rvecs)
# rvecs = rvecs .- make_center_of_mass(rvecs)'
rvecs[:,3] = d*ones(size(rvecs[:,3]))

# println(rvecs[1,:])
# println(rvecs[end,:])
# println(size(rvecs))
# println(make_length_of_filament2(rvecs))

if w == ws[1]
    display(Plots.scatter(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal))
else
    display(Plots.scatter!(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal))
end


tvecs = make_tvecs(rvecs)


hs = h*ones(n) # corresponding differential lengths in the integral
hs[1] = h/2
hs[end] = h/2 

# integralMat such that integralMat.farr is the integral
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
                    )*hs[x]
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

# rvec = [2.,-3.,2.]
# rvec0 = [-1.,6.,3.]
# oseen_tensor = make_stokeslet_wall(rvec,rvec0)

ωvec = [0.,0.,1.]
vvecs = zeros(size(rvecs))

for i = 1:n
    vvecs[i,:] = cross(ωvec,rvecs[i,:])
end

varr = reshape(vvecs',3*n,1)

# approximate solution
E10 = 2*asinh(1/(4*d))
E20 = 1/2/sqrt( 1+16*d^2 )
E30 = 1/2/sqrt( 1+16*d^2 )^3
α10 = log( (d+sqrt(d^2-ρ^2))/ρ )

ζperp0 = 8π / ( log(1/(4*d^2)) +1 -E10 - 2*E20 +2*α10   )
ζpar0 = 4π / ( log(1/(4*d^2)) -1 -E10 + E20 +2*α10   )

fvecs_local = zeros(size(vvecs))
for i = 1:n
    fvecs_local[i,:] = ζpar0*dot(vvecs[i,:], tvecs[i,:])*tvecs[i,:] +
                        ζperp0*( vvecs[i,:] - dot(vvecs[i,:], tvecs[i,:])*tvecs[i,:] )
end

farr0 = reshape(fvecs_local',3*n,1) # approximate solution

# farr = integralMat \ varr
farr = (  transpose(integralMat)*integralMat + regλ^2*I ) \ ( transpose(integralMat)*varr + regλ^2*I*farr0)

fvecs = reshape(farr[1:3*n], 3, n)'



# display(Plots.plot(fvecs_local[:,:]))
# display(Plots.plot!(fvecs[:,:]))
#display(Plots.plot(vvecs[:,:]))

# # in the normal direction the Fredholm integral equation of the 2nd type results in good results
# fvecs_local_n = fvecs_local - make_dot_product(fvecs_local,tvecs) .* tvecs
# fvecs_n = fvecs - make_dot_product(fvecs,tvecs) .* tvecs
# display(Plots.plot(fvecs_local_n[:,:]))
# display(Plots.plot!(fvecs_n[:,:]))

# # in the tangent direction the Fredholm integral equation of the 1st type results in oscillations
# fvecs_local_t = make_dot_product(fvecs_local,tvecs)
# fvecs_t = make_dot_product(fvecs,tvecs) 
# display(Plots.plot(fvecs_local_t))
# display(Plots.plot!(fvecs_t[:,:]))

Flift = sum(fvecs[:,3].*hs)
println("Flift = ",Flift)

Torque = sum( make_cross_product(rvecs,fvecs).*hs  )
println("Torque for rod = ",ζperp0/12)
println("Torque = ",Torque)
println("*********")

# append!(ds, d)
append!(Flifts, Flift)
append!(Torques, Torque)
append!(Torques_rod, ζperp0/12)

end # end ws


# open("./simulation_results/filament_lift/simulate_different_w_lift/results" * string(1/radius_scale) * ".txt", "w") do io
#     writedlm(io, ["w" w*ones(size(ds')); "radius" ρ*ones(size(ds')); "n" n*ones(size(ds')); "heights" ds'; "lift force" Flifts'; "torques" Torques'; "torques_rod" Torques_rod'])
# end



Plots.scatter(ws,Flifts)
#Plots.scatter(ds,Flifts)
# Plots.scatter(ds,Torques)
# Plots.scatter!(ds,Torques_rod)