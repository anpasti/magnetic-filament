function calculate_weights(order::Int, x0::T, x::AbstractVector) where {T <: Real}
    #=
        order: The derivative order for which we need the coefficients
        x0   : The point in the array 'x' for which we need the coefficients
        x    : A dummy array with relative coordinates, e.g., central differences
               need coordinates centred at 0 while those at boundaries need
               coordinates starting from 0 to the end point
        The approximation order of the stencil is automatically determined from
        the number of requested stencil points.
    =#
    #############################################################
    # from thread https://discourse.julialang.org/t/generating-finite-difference-stencils/85876/17
    # https://github.com/SciML/DiffEqOperators.jl/blob/master/src/derivative_operators/fornberg.jl
    # Fornberg algorithm

    # This implements the Fornberg (1988) algorithm (https://doi.org/10.1090/S0025-5718-1988-0935077-0)
    # to obtain Finite Difference weights over arbitrary points to arbitrary order.

    N = length(x)
    @assert order<N "Not enough points for the requested order."
    M = order
    c1 = one(T)
    c4 = x[1] - x0
    C = zeros(T, N, M + 1)
    C[1, 1] = 1
    @inbounds for i in 1:(N - 1)
        i1 = i + 1
        mn = min(i, M)
        c2 = one(T)
        c5 = c4
        c4 = x[i1] - x0
        for j in 0:(i - 1)
            j1 = j + 1
            c3 = x[i1] - x[j1]
            c2 *= c3
            if j == i - 1
                for s in mn:-1:1
                    s1 = s + 1
                    C[i1, s1] = c1 * (s * C[i, s] - c5 * C[i, s1]) / c2
                end
                C[i1, 1] = -c1 * c5 * C[i, 1] / c2
            end
            for s in mn:-1:1
                s1 = s + 1
                C[j1, s1] = (c4 * C[j1, s1] - s * C[j1, s]) / c3
            end
            C[j1, 1] = c4 * C[j1, 1] / c3
        end
        c1 = c2
    end
    #=
        This is to fix the problem of numerical instability which occurs when the sum of the stencil_coefficients is not
        exactly 0.
        https://scicomp.stackexchange.com/questions/11249/numerical-derivative-and-finite-difference-coefficients-any-update-of-the-fornb
        Stack Overflow answer on this issue.
        http://epubs.siam.org/doi/pdf/10.1137/S0036144596322507 - Modified Fornberg Algorithm
    =#
    _C = C[:, end]
    if order != 0
        _C[div(N, 2) + 1] -= sum(_C)
    end
    return _C
end



function make_D1_1st_order(n)
    #= 
    calculate the 1st order finite differentiation matrix D1
    for an array of n points r a distance h appart
    dr/dl = D1 * r / h, where h is the distance between points r
    =#

    D1 = zeros((n,n))
    D1[1,1:2] = calculate_weights(1, 0., 0:1) # + O(h^2)
    D1[end,end-1:end] = calculate_weights(1, 0., -1:0) # + O(h^2)

    mid_weights = calculate_weights(1, 0., 0:1)
    for i = 2:n-1
        D1[i,i:i+1] = mid_weights
    end

    return D1
end



function make_padded_rvecs(rvecs)
    # pad rvecs with extrapolated ghost nodes to have d^2r / dl^2 = 0
    r0 = 2*rvecs[1,:] - rvecs[2,:] # linearly extrapolated
    rnplus1 = 2*rvecs[end,:] - rvecs[end-1,:] # linearly extrapolated

    return vcat(r0', rvecs, rnplus1')
end



function make_D1(rvecs)
    #= 
    calculate the 1st order finite differentiation matrix D1
    for an array of n points r a distance h appart
    dr/dl = D1 * r / h, where h is the distance between points r
    =#
    n = size(rvecs,1)

    D1 = zeros((n,n))
    D1[1,1:3] = calculate_weights(1, 0., 0:2) # + O(h^2)
    D1[end,end-2:end] = calculate_weights(1, 0., -2:0) # + O(h^2)

    mid_weights = calculate_weights(1, 0., -1:1)
    for i = 2:n-1
        D1[i,i-1:i+1] = mid_weights
    end

    return sparse(D1) # sparse gives 5x improvement in speed
end



function make_D2(rvecs)
    #= 
    calculate the 2nd order finite differentiation matrix D2
    for an array of n points r a distance h appart
    d^2r / dl^2 = D2 * r / h^2, where h is the distance between points r
    =#
    n = size(rvecs,1)

    D2 = zeros((n,n))
    D2[1,1:3] = calculate_weights(2, 0., 0:2) # + O(h)
    D2[end,end-2:end] = calculate_weights(2, 0., -2:0) # + O(h)

    mid_weights = calculate_weights(2, 0., -1:1)
    for i = 2:n-1
        D2[i,i-1:i+1] = mid_weights
    end

    return sparse(D2) # sparse gives 2.5x improvement in speed
end



function make_D2_BC(rvecs)
    #= 
    calculate the 2nd order finite differentiation matrix D2
    for an array of n points r a distance h appart
    d^2r / dl^2 = D2 * r / h^2, where h is the distance between points r

    boundary condition: 2nd derivative at the ends is 0
    =#
    n = size(rvecs,1)

    D2 = zeros((n,n))
    #D2[1,1:3] = calculate_weights(2, 0., 0:2) # + O(h)
    #D2[end,end-2:end] = calculate_weights(2, 0., -2:0) # + O(h)

    mid_weights = calculate_weights(2, 0., -1:1)
    for i = 2:n-1
        D2[i,i-1:i+1] = mid_weights
    end

    return sparse(D2)
end



function make_D3(rvecs)
    #= 
    calculate the 3rd order finite differentiation matrix D3
    for an array of n points r a distance h appart
    d^3r / d3^4 = D3 * r / h^3, where h is the distance between points r
    =#
    n = size(rvecs,1)
    D3 = zeros((n,n))

    D3[1,1:5] = calculate_weights(3, 0., 0:4) # + O(h^2)
    D3[end,end-4:end] = calculate_weights(3, 0., -4:0) # + O(h^2)

    # endpoints go nicely out of the loop.  
    D3[2,1:5] = calculate_weights(3, 0., -1:3) # + O(h^2)
    D3[end-1,end-4:end] = calculate_weights(3, 0., -3:1) # + O(h^2)
    
    # D4[1,1:3] = [-1, 2, -1] # + O(h)
    # D4[end,end-2:end] = [-1, 2, -1] # + O(h)
    # D4[2,1:4] = [2,-5,4,-1] # + O(h)
    # D4[end-1,end-3:end] = [-1,4,-5,2]# + O(h)

    mid_weights = calculate_weights(3, 0., -2:2)
    for i = 3:n-2
        D3[i,i-2:i+2] = mid_weights # + O(h^2)
    end

    return sparse(D3)

end



function make_D4(rvecs)
    #= 
    calculate the 4th order finite differentiation matrix D4 
    for an array of n points r a distance h appart
    d^4r / dl^4 = D4 * r / h^4, where h is the distance between points r
    =#
    n = size(rvecs,1)
    D4 = zeros((n,n))

    # last and first indices include d^2 r / dl^2 = 0 condition
    # calculated these by hand by comparing 5 indices of d^2 r / dl^2 matrix to d^4 matrix
    # see Rūdolfs Livanovičs thesis last appendix
    # D4[1,1:4] = 1/11 * [-24, 60, -48, 12] # + O(h)
    # D4[end,end-3:end] = 1/11 * [12, -48, 60, -24] # + O(h)
    # šis ir izrakstīts no Rūdolfa disertācijas:
    # D4[1,1:3] = 0.05*[1, -2, 1] # + O(h)
    # D4[end,end-2:end] = 0.05*[1, -2, 1] # + O(h)
    # šis ir default, ko dod Fornberga algoritms
    ########################
    D4[1,1:5] = calculate_weights(4, 0., 0:4) # + O(h)
    D4[end,end-4:end] = calculate_weights(4, 0., -4:0) # + O(h)

    # endpoints go nicely out of the loop. 
    # by coincidence, these coefficients are the same as for the mid points
    D4[2,1:5] = calculate_weights(4, 0., -1:3) # + O(h)
    D4[end-1,end-4:end] = calculate_weights(4, 0., -3:1) # + O(h)
    
    # D4[1,1:3] = [-1, 2, -1] # + O(h)
    # D4[end,end-2:end] = [-1, 2, -1] # + O(h)
    # D4[2,1:4] = [2,-5,4,-1] # + O(h)
    # D4[end-1,end-3:end] = [-1,4,-5,2]# + O(h)

    mid_weights = calculate_weights(4, 0., -2:2)
    for i = 3:n-2
        D4[i,i-2:i+2] = mid_weights # + O(h)
    end

    return sparse(D4)

end