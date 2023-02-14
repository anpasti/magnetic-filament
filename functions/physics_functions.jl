function make_fmatrix(n,D3_BC, D4_BC,h)
    # matrix such that F = fmatrix * rvecs, no magnetic force though yet #

    fmatrix = zeros((n,n))


    fmatrix[1,:] = -D3_BC[1,:] / h^3
    fmatrix[end,:] = D3_BC[1,:] / h^3
    fmatrix[2:end-1,:] = -D4_BC[2:end-1,:] / h^3

    return fmatrix
end

function make_fmagvecs(n,Cm, hvec)
    

    fmagvecs = zeros((n,3))


    fmagvecs[1,:] = Cm * hvec
    fmagvecs[end,:] = -Cm * hvec

    return fmagvecs
end

function make_fvecs(r4vecs, r3vecs, h, hvec, Cm)
    # these are dF vectors - with a dimension of force. 
    # Need to divide by h to get linear force density

    n = size(r4vecs,1)

    fvecs = zeros(size(r4vecs))

    fvecs[1,:] = Cm*hvec - r3vecs[1,:] 
    fvecs[end,:] = -Cm*hvec + r3vecs[end,:] 

    for i = 2:n-1
        fvecs[i,:] = -h * r4vecs[i,:]
    end

    return fvecs
end

function make_symbolic_J(rvecs)
    # J - the Jacobian matrix of the constarints #
    # https://arxiv.org/pdf/0903.5178.pdf #

    n = size(rvecs,1)

    J = zeros(Num,(n-1,3n)) # n-1 constraints

    for i = 1:n-1
        drvec = rvecs[i+1,:] - rvecs[i,:]
        
        J[i,  (i-1)*3 + 1 : (i-1)*3 + 3] = drvec
        J[i, (i-1)*3 + 1 + 3 : (i-1)*3 + 3 + 3] = -drvec
    end 

    return -2*J
end

function make_J(rvecs)
    # J - the Jacobian matrix of the constarints #
    # https://arxiv.org/pdf/0903.5178.pdf #

    n = size(rvecs,1)

    J = zeros(typeof(rvecs[1]),(n-1,3n)) # n-1 constraints

    for i = 1:n-1
        drvec = rvecs[i+1,:] - rvecs[i,:]
        J[i,  (i-1)*3 + 1 : (i-1)*3 + 3] = drvec
        J[i, (i-1)*3 + 1 + 3 : (i-1)*3 + 3 + 3] = -drvec
    end 

    return -2*J
end


function make_proj_to_inextensile(vvecs, rvecs)
    #projects velocities to not extend the chain#
    # https://arxiv.org/pdf/0903.5178.pdf #
    # v_p = P*v, but v is a 3n vector#

    n = size(rvecs,1)

    J = make_J(rvecs)

    P = I - transpose(J) * ( J * transpose(J) )^(-1) * J

    v = reshape(vvecs',3*n,1)
    v_p = P*v

    #println(J*v_p)

    return reshape(v_p, 3, n)'

end

function make_fm_arr(n, hvec, Cm)
    fm_arr = zeros(3*n,1)

    fm_arr[1:3] = Cm*hvec
    fm_arr[end-2:end] = -Cm*hvec

    return fm_arr
end

function make_Mmat(h,D2_BC,D2,D1)
    # Mmat 3n x 3n such that M*r_arr = F_nonmagnetic_arr
    n = size(D1,1)

    Mmat = zeros(3*n,3*n)
    D3_BC = D1*D2_BC
    D4_BC = D2*D2_BC

    for i = 1:n
        # first and last element - third derivative
        for k = 1:3
            for m = 1:3
                Mmat[(1-1)*3 + m,(i-1)*3+k] = D3_BC[1,i]

                Mmat[(n-1)*3 + m,(i-1)*3+k] = -D3_BC[n,i]
            end
        end
    end

    for i = 1:n
        for j = 2:n-1
            for k = 1:3
                for m = 1:3
                    # middle elements - fourth derivative
                    Mmat[(j-1)*3+k,(i-1)*3+m] = D4_BC[j,i]
                end
            end
        end
    end

    return -Mmat / h^3
end

function make_Mmat_ceb(h,n)
    # manuāli ar roku kā Cēbers
    # Mmat 3n x 3n such that M*r_arr = F_nonmagnetic_arr

    Mmat = zeros(3*n,3*n)


    # first and last element - third derivative
    for k = 1:3
        for m = 1:3
            if k==m
                Mmat[(1-1)*3 + m,(1-1)*3+k] = 1
                Mmat[(1-1)*3 + m,(2-1)*3+k] = -2
                Mmat[(1-1)*3 + m,(3-1)*3+k] = 1

                Mmat[(n-1)*3 + m,(n  -1)*3+k] = 1
                Mmat[(n-1)*3 + m,(n-1-1)*3+k] = -2
                Mmat[(n-1)*3 + m,(n-2-1)*3+k] = 1
            end
        end
    end


    # second and second to last element - fourth derivative with modified values
    for k = 1:3
        for m = 1:3
            if k == m
                Mmat[(2-1)*3 + m,(1-1)*3+k] = -2
                Mmat[(2-1)*3 + m,(2-1)*3+k] = 5
                Mmat[(2-1)*3 + m,(3-1)*3+k] = -4
                Mmat[(2-1)*3 + m,(4-1)*3+k] = 1

                Mmat[(n-1-1)*3 + m,(n  -1)*3+k] = -2
                Mmat[(n-1-1)*3 + m,(n-1-1)*3+k] = 5
                Mmat[(n-1-1)*3 + m,(n-2-1)*3+k] = -4
                Mmat[(n-1-1)*3 + m,(n-3-1)*3+k] = 1
            end
        end
    end

    # middle elements - fourth derivative
    for i = 1:n
        for j = 3:n-2
            for k = 1:3
                for m = 1:3
                    
                    if i==j
                        if k==m
                            Mmat[(j-1)*3+m,(i-2-1)*3+k] = 1
                            Mmat[(j-1)*3+m,(i-1-1)*3+k] = -4
                            Mmat[(j-1)*3+m,(i-0-1)*3+k] = 6
                            Mmat[(j-1)*3+m,(i+1-1)*3+k] = -4
                            Mmat[(j-1)*3+m,(i+2-1)*3+k] = 1
                        end
                    end
                end
            end
        end
    end

    return -Mmat / h^3
end

function make_Mmat_Fparamag(h,n,hvec, Cm)
    # manuāli ar roku kā Cēbers
    # Mmat 3n x 3n such that Mmat_Fparamag*rarr = F_paramag

    Mmat = zeros(3*n,3*n)


    # first and last element - first derivative
    for k = 1:3
        for m = 1:3
            if k==m
                Mmat[(1-1)*3 + m,(1-1)*3+k] = -1#-1
                Mmat[(1-1)*3 + m,(2-1)*3+k] = 1#1

                Mmat[(n-1)*3 + m,(n  -1)*3+k] = -1#-1
                Mmat[(n-1)*3 + m,(n-1-1)*3+k] = 1#1
            end
        end
    end

    # middle elements - second derivative
    for i = 1:n
        for j = 2:n-1
            for k = 1:3
                for m = 1:3
                    
                    if i==j
                        if k==m
                            Mmat[(j-1)*3+m,(i-1-1)*3+k] = 1
                            Mmat[(j-1)*3+m,(i-0-1)*3+k] = -2
                            Mmat[(j-1)*3+m,(i+1-1)*3+k] = 1
                        end
                    end

                end
            end
        end
    end

    Mmat = -Mmat / h

    # now project on hvec and multiply by Cm
    block = hvec * hvec'
    hprojMat = kron(I(n),Cm*block)

    return hprojMat*Mmat
end

# function make_hprojMat(n, hvec, Cm)
#     # 3n x 3n matrix that projects 3N vector on hvec and multipleis by Cm
#     block = hvec * hvec'
#     hprojMat = kron(I(n),Cm*block)
#     return hprojMat
# end


function make_Mmat_spont_curv(h,n)
    # manuāli ar roku kā Cēbers
    # Mmat 3n x 3n such that M*k0*n_arr = F_spont_curv

    Mmat = zeros(3*n,3*n)


    # first and last element - first derivative
    for k = 1:3
        for m = 1:3
            if k==m
                Mmat[(1-1)*3 + m,(1-1)*3+k] = -1#-1
                Mmat[(1-1)*3 + m,(2-1)*3+k] = 1#1

                Mmat[(n-1)*3 + m,(n  -1)*3+k] = -1#-1
                Mmat[(n-1)*3 + m,(n-1-1)*3+k] = 1#1
            end
        end
    end

    # middle elements - second derivative
    for i = 1:n
        for j = 2:n-1
            for k = 1:3
                for m = 1:3
                    
                    if i==j
                        if k==m
                            Mmat[(j-1)*3+m,(i-1-1)*3+k] = 1
                            Mmat[(j-1)*3+m,(i-0-1)*3+k] = -2
                            Mmat[(j-1)*3+m,(i+1-1)*3+k] = 1
                        end
                    end

                end
            end
        end
    end

    return -Mmat / h
end



function make_ftwist_arr(h, om_twist, rvecs, D1, D2)
    # force doe to twist
    # such that v_arr = P*(C*ftwist_arr)

    n = size(rvecs,1)

    rvecsl = D1*rvecs
    rvecsll = D2*rvecs
    rvecs_cross_term = make_cross_product(rvecsl,rvecsll)

    Force = om_twist .* rvecs_cross_term
    force_density = D1 * Force

    fvecs = zeros(size(rvecs))
    fvecs[2:end-1] = force_density[2:end-1] 
    fvecs[1] = Force[1] * h
    fvecs[end] = -Force[end] *h

    return reshape(fvecs',3*n,1)
end



function make_proj_operator(rvecs)
    #projects velocities to not extend the chain#
    # https://arxiv.org/pdf/0903.5178.pdf #

    n = size(rvecs,1)

    J = make_J(rvecs)

    P = I - transpose(J) * ( J * transpose(J) )^(-1) * J

    return P

end

function make_symbolic_proj_operator_mobility_tensor(rvecs,lambda,h)
    #projects velocities to not extend the chain#
    # https://arxiv.org/pdf/0903.5178.pdf #

    n = size(rvecs,1)

    J = make_symbolic_J(rvecs)
    μ = make_symbolic_mobility_tensor(rvecs, lambda,h)
    P = I - transpose(J) * ( J * μ * transpose(J) )^(-1) * J * μ

    return P, μ

end

function make_proj_operator_mobility_tensor(rvecs,lambda,h)
    #projects velocities to not extend the chain#
    # https://arxiv.org/pdf/0903.5178.pdf #

    n = size(rvecs,1)

    J = make_J(rvecs)
    μ = make_mobility_tensor(rvecs, lambda,h)
    P = I - transpose(J) * ( J * μ * transpose(J) )^(-1) * J * μ

    return P, μ

end

function make_symbolic_mobility_tensor(rvecs, lambda, h)
    # returns 3N x 3N matrix such that varr = mob_mat * farr 

    ζratio = 1 - lambda # zeta_perp / zeta_par

    n = size(rvecs,1)
    mob_mat = zeros(Num,(3n,3n))

    tvecs = make_tvecs(rvecs)
    for i = 1:n
        friction_mat_i = I + (1/ζratio - 1) * tvecs[i,:]*tvecs[i,:]'
        
        mob_mat[(i-1)*3+1 : (i-1)*3+3, (i-1)*3+1 : (i-1)*3+3] = friction_mat_i^(-1)
    end

    return mob_mat/h
end

function make_mobility_tensor(rvecs, lambda, h)
    # returns 3N x 3N matrix such that varr = mob_mat * farr 

    ζratio = 1 - lambda # zeta_perp / zeta_par

    n = size(rvecs,1)
    mob_mat = zeros(typeof(rvecs[1]),(3n,3n))

    tvecs = make_tvecs(rvecs)
    for i = 1:n
        friction_mat_i = I + (1/ζratio - 1) * tvecs[i,:]*tvecs[i,:]'
        
        mob_mat[(i-1)*3+1 : (i-1)*3+3, (i-1)*3+1 : (i-1)*3+3] = friction_mat_i^(-1)
    end

    return mob_mat/h
end


function make_stokeslet(rvec,rvec0)
    # evaluated at r due to r0
    Rvec = rvec - rvec0

    return  (I + Rvec*Rvec'/ norm(Rvec)^2 ) / norm(Rvec)
    
end

function make_stokeslet_velocity_wall(Fvec,rvec,rvec0)
    # stokeslet velocity near a wall. see Blake and Chwang (1973).
    # evaluated at r due to force F at r0
    # F_z must be 0
    # wall is assumed to be at z=0

    δ(i,j) = ==(i,j) # Kroneker delta

    Rvec = rvec - rvec0
    R = norm(Rvec)
    Rimvec = rvec - rvec0 # image system radius vector
    h = rvec0[3] # height above wall
    Rimvec[3] += 2*h
    Rim = norm(Rimvec)

    vvec = [0.,0.,0.]
    for i = 1:3
        for j = 1:2 # F_z=0
            vvec[i] += 
            Fvec[j]*(
             (δ(i,j)/R + Rvec[i]*Rvec[j]/R^3) 
            -(δ(i,j)/Rim + Rimvec[i]*Rimvec[j]/Rim^3)
            +2*h/Rim^3*
                (
                 δ(i,j)*(h-Rimvec[3])
                 -3*Rimvec[i]*Rimvec[j]/Rim^2 * (h-Rimvec[3])
                 +δ(i,3)*Rimvec[j]
                 -δ(j,3)*Rimvec[i]
                )
            )
        end
    end


    return  vvec
    # corresponds with Mathematica
end

function make_doublet(rvec,rvec0)
    # source and sink doublet
    # evaluated at r due to r0
    Rvec = rvec - rvec0

    return  (I - 3* Rvec*Rvec'/ norm(Rvec)^2 ) / norm(Rvec)^3
    
end

function make_doublet_velocity_field_wall(Fvec,rvec,rvec0)
    # source doublet velocity near a wall. see Blake and Chwang (1973).
    # evaluated at r due to source F at r0
    # F_z must be 0
    # wall is assumed to be at z=0

    δ(i,j) = ==(i,j) # Kroneker delta

    Rvec = rvec - rvec0
    R = norm(Rvec)
    z=rvec[3]
    Rimvec = rvec - rvec0 # image system radius vector
    h = rvec0[3] # height above wall
    Rimvec[3] += 2*h
    Rim = norm(Rimvec)

    vvec = [0.,0.,0.]
    for i = 1:3
        for j = 1:2 # F_z=0
            vvec[i] += 
            Fvec[j]*(
             (δ(i,j)/R^3 - 3*Rvec[i]*Rvec[j]/R^5) 
            -(δ(i,j)/Rim^3 - 3*Rimvec[i]*Rimvec[j]/Rim^5)
            )
        end

        for α=1:2
            vvec[i] -= 
            Fvec[α]*δ(i,3)*6*Rimvec[α]*Rimvec[3]/Rim^5
        end

        for α=1:2
            for β=1:2
                vvec[i] -= 
                2*Fvec[α]*δ(i,β)*
                (
                -3*Rimvec[3]*δ(α,β)*z/Rim^5 +
                15*Rimvec[3]*Rimvec[α]*Rimvec[β]*z/Rim^7
                )
            end
        end

        for α=1:2
            vvec[i] -= 
            2*Fvec[α]*δ(i,3)*
            (
            -3*Rimvec[α]*z/Rim^5
            +15*Rimvec[3]^2*z*Rimvec[α]/Rim^7
            )
        end

    end

    return vvec
    # corresponds with Mathematica
end


function make_velocity_field(rvec_eval, rvecs, fvecs, ϵ, h)
    # Like in Laurel Ohm thesis
    # flow at rvec_eval
    # ϵ = radius / length of filament
    # u_bkg - background flow
    # rvecs - filament element radius vectors
    # fvecs - dF array, with a dimension of force
    # h - distance between elements

    α = 1/(2*log(1/ϵ)+1)

    n = size(rvecs,1)

    u_disturbance = [0., 0., 0.]
    for i = 1:n
        S = make_stokeslet(rvec_eval,rvecs[i,:])
        D = make_doublet(rvec_eval,rvecs[i,:])

        u_disturbance += α*(S + ϵ^2 / 2 * D ) * fvecs[i,:]/h * h
    end

    return u_disturbance
end



function make_velocity_field_wall(rvec_eval, rvecs, fvecs, ϵ, d, h)
    # Like in Laurel Ohm thesis
    # flow at rvec_eval
    # ϵ = radius / length of filament
    # d = height above wall / length of filament
    # u_bkg - background flow
    # rvecs - filament element radius vectors
    # fvecs - dF array, with a dimension of force
    # h - distance between elements in units of length of filament

    α = 1/(log(1/(4*d^2)) + 1 - E1(d) - 2*E2(d) + 2α1(d,ϵ))

    n = size(rvecs,1)

    u_disturbance = [0., 0., 0.]
    for i = 1:n
        uS = make_stokeslet_velocity_wall(fvecs[i,:],rvec_eval,rvecs[i,:])
        uD = make_doublet_velocity_field_wall(fvecs[i,:],rvec_eval,rvecs[i,:])

        u_disturbance += α*(uS + ϵ^2 / 2 * uD )
    end

    return u_disturbance
end



# like in paper by Koen and Montenegro-Jonhson, PRF, 2021
E1(d) = 2 * asinh( 1/(4*d) )
E2(d) = 1 / ( 2* sqrt(1+16* d^2) )
E3(d) = 1 / ( 2 *sqrt(1+16* d^2)^3 )
α1(d,ϵ) = log( (d + sqrt(d^2-ϵ^2) )/ϵ )

function make_ζratio_wall(d,ϵ)
    # zeta_perp/zeta_par above a wall
    # like in paper by Koen and Montenegro-Jonhson, PRF, 2021
    # d - distance to wall, ϵ - filament radius. both in units of filament length.
    # tested with MAthematica - this function works correctly

    ζperp = 8*pi / ( log(1/(4*d^2)) + 1 - E1(d) - 2*E2(d) + 2* α1(d,ϵ) )
    
    ζpar = 4*pi / ( log(1/(4*d^2)) -1 - E1(d) + E2(d) + E3(d) + 2* α1(d,ϵ) )

    return ζperp/ζpar
end


