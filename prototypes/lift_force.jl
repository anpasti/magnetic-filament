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

n = 17*2
h = 1/n # distance between 2 pts 
ρ = h/sqrt(exp(1)) # radius of filament
d = 9.9*ρ # height

wint = 11 # how many points in the middle section
w = wint * h / 2 # 2w/h should be integer
r0 = exp(-pi/2)*w
θmax = log( exp(pi/2) * (1-2*w+2*sqrt(2)*w) / (2*sqrt(2)*w) )
armlength = 1/2 - w



# # initialize the shape
rvecs = zeros(n,3)
# rvecs[:,1] = range(-0.5,0.5,length=n)
# rvecs[:,2] = range(-0.5,0.5,length=n).^3
# rvecs[:,3] = d*ones(size(rvecs[:,3]))

# a convuluted function to make the shape
θ = θmax
for i = 1:n # first arm
    global θ = log( -h/sqrt(2)/r0 + exp(θ) )
    #println(θ)
    if θ < pi/2
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
end


renormalize_length!(rvecs)
rvecs = rvecs .- make_center_of_mass(rvecs)'
rvecs[:,3] = d*ones(size(rvecs[:,3]))

display(Plots.scatter(rvecs[:,1],rvecs[:,2],aspect_ratio=:equal))

tvecs = make_tvecs(rvecs)


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

# rvec = [2.,-3.,2.]
# rvec0 = [-1.,6.,3.]
# oseen_tensor = make_stokeslet_wall(rvec,rvec0)

ωvec = [0.,0.,1.]
vvecs = zeros(size(rvecs))

for i = 1:n
    vvecs[i,:] = cross(ωvec,rvecs[i,:])
end

varr = reshape(vvecs',3*n,1)

farr = integralMat \ varr
fvecs = reshape(farr[1:3*n], 3, n)'


display(Plots.scatter(fvecs[:,3]))