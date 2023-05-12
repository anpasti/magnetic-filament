function make_tvecs(rvecs)
    # unit tangent vector for every point
    n = size(rvecs,1)
    tvecs = zeros(typeof(rvecs[1]),size(rvecs))

    tvecs[1,:] = normalize(rvecs[2,:] - rvecs[1,:])
    #tvecs[1,:] = (rvecs[2,:] - rvecs[1,:])/norm(rvecs[2,:] - rvecs[1,:])
    tvecs[end,:] = normalize(rvecs[end,:] - rvecs[end-1,:])
    #tvecs[end,:] = (rvecs[end,:] - rvecs[end-1,:])/norm(rvecs[end,:] - rvecs[end-1,:])
    for i = 2:n-1
        # dr_forw = rvecs[i,:] - rvecs[i-1,:]
        # dr_back = rvecs[i+1,:] - rvecs[i,:]
        tvecs[i,:] = normalize(rvecs[i+1,:] - rvecs[i-1,:])
        #tvecs[i,:] = (rvecs[i+1,:] - rvecs[i-1,:]) / norm(rvecs[i+1,:] - rvecs[i-1,:])
    end

    return tvecs
end

function make_tvecs_diff(rvecs, h)
    # unit tangent vector for every point
    n = size(rvecs,1)
    tvecs = zeros(typeof(rvecs[1]),size(rvecs))

    tvecs[1,:] = (rvecs[2,:] - rvecs[1,:])/h
    #tvecs[1,:] = (rvecs[2,:] - rvecs[1,:])/norm(rvecs[2,:] - rvecs[1,:])
    tvecs[end,:] = (rvecs[end,:] - rvecs[end-1,:])/h
    #tvecs[end,:] = (rvecs[end,:] - rvecs[end-1,:])/norm(rvecs[end,:] - rvecs[end-1,:])
    for i = 2:n-1
        # dr_forw = rvecs[i,:] - rvecs[i-1,:]
        # dr_back = rvecs[i+1,:] - rvecs[i,:]
        tvecs[i,:] = (rvecs[i+1,:] - rvecs[i-1,:])/(2h)
        #tvecs[i,:] = (rvecs[i+1,:] - rvecs[i-1,:]) / norm(rvecs[i+1,:] - rvecs[i-1,:])
    end

    return tvecs
end



function make_nvecs(rvecs, nvecs_prev)
    # unit normal vector for every point
    # nvecs_prev are normals from the previous time iteration - used to determine the direction
    n = size(rvecs,1)
    nvecs = zeros(size(rvecs))

    for i = 2:n-1
        # dr_forw = rvecs[i,:] - rvecs[i-1,:]
        # dr_back = rvecs[i+1,:] - rvecs[i,:]
        nvecs[i,:] = normalize( (rvecs[i,:] - rvecs[i-1,:])  - (rvecs[i+1,:] - rvecs[i-1,:]) / 2)
        if dot(nvecs[i,:], nvecs_prev[i,:]) < 0 # if wrong direction
            nvecs[i,:] = -nvecs[i,:]
        end
    end

    # different treatment for end and beginning normal vectors
    tvec2 = normalize(rvecs[3,:] - rvecs[1,:])
    tvec1 = normalize(rvecs[2,:] - rvecs[1,:])
    bvec2 = cross(tvec2,nvecs[2,:])
    nvecs[1,:] = cross(bvec2,tvec1)

    tvec_endm1 = normalize(rvecs[end,:] - rvecs[end-2,:])
    tvec_end = normalize(rvecs[end,:] - rvecs[end-1,:])
    bvec_endm1 = cross(tvec_endm1,nvecs[end-1,:])
    nvecs[end,:] = cross(bvec_endm1,tvec_end)

    return nvecs
end



function make_nvecs_plane(tvecs)
    # unit normal vector for every point
    # only works for filaments in xy plane
    n = size(tvecs,1)
    nvecs = zeros(size(tvecs))
    bvec = [0.,0.,1.]

    for i = 1:n
        nvecs[i,:] = cross(bvec,tvecs[i,:])
    end

    return nvecs
end



#function make_curvatures(tvecs,rvecs, h)
#    # curvature for every point
#    # it is assumed to be 0 at tips
#    n = size(rvecs,1)
#    ks = zeros(n,1)
#    D1 = make_D1(rvecs)
#
#    ks_times_nvecs = D1*tvecs / h
#    ks = sqrt.(sum(ks_times_nvecs.^2,dims=2))
#
#    return ks
#end


function make_curvatures(rvecs, h)
    # curvatures = |d tvec / dl|

    n = size(rvecs,1)
    tvecs = make_tvecs(rvecs)

    curvatures = zeros(n)
    for i = 2:n-1
        curvatures[i] = norm( (-0.5 * tvecs[i-1,:] + 0.5 * tvecs[i+1,:]) / h )
    end

    curvatures[1] = norm( (-1.5 * tvecs[1,:] + 2 * tvecs[2,:] - 0.5 * tvecs[3,:]) / h )
    curvatures[n] = norm( (0.5 * tvecs[n-2,:] - 2 * tvecs[n-1,:] + 1.5 * tvecs[n,:]) / h )

    return curvatures

end



function make_length_of_filament(rvecs)

    n = size(rvecs,1)
    L = 0
    for i = 1:n-1
        L += norm(rvecs[i,:] - rvecs[i+1,:])
    end

    return L*(n+1)/n # adjustment to include the half distances at each end
end



function renormalize_length!(rvecs)
    # rescale all the lengths between consecutive so that total filament length is 1
    n = size(rvecs,1)

    for i = 1:n-1 # for each segment
        rveci = rvecs[i+1,:] - rvecs[i,:]

        renorm = 1 - 1/n/norm(rveci)
        α1 = (n-i)*renorm / n
        α2 = i*renorm / n

        for j = 1:i
            rvecs[j,:] += α1*rveci
        end
        for j = i+1:n
            rvecs[j,:] -= α2*rveci
        end

    end
end



function make_hav(rvecs)
    # find the average differential line element corresponding to a vertex

    n = size(rvecs,1)

    hav = zeros(n)

    for i = 2:n-1
        hav[i] = 0.5*norm(rvecs[i+1,:] - rvecs[i,:]) + 0.5*norm(rvecs[i,:] - rvecs[i-1,:])
    end

    hav[1] = norm(rvecs[1,:] - rvecs[2,:])
    hav[n] = norm(rvecs[n-1,:] - rvecs[n,:])

    return hav
end



function make_center_of_mass(rvecs)
    
    n = size(rvecs,1)

    hav = make_hav(rvecs) # dl array
    L = sum(hav)
    rvec_cm = zeros(3)

    for i = 1:n
        rvec_cm += rvecs[i,:]*hav[i] 
    end

    rvec_cm /= L
    return rvec_cm 
end



function make_center_of_mass_velocity(vvecs)
    # assuming L = 1
    n = size(vvecs,1)

    vvec_cm = zeros(3)
    for i = 1:n
        vvec_cm += vvecs[i,:]
    end
    vvec_cm /= n
    return vvec_cm 
end



function rotate(vect,angle,axis)
    # rotate vect around axis by angle in radians
    # axis vector is automatically normalized
    Rmat = AngleAxis(angle, axis[1], axis[2], axis[3])

    return MVector(Rmat*vect) # annoyingly need to convert to mutable
end




function align_filament_horizontally(rvecs,hvec)
    tvecs = make_tvecs(rvecs)

    n = size(rvecs,1)
    if n%2 == 1 # if odd
        tvec_middle = tvecs[div(n+1,2),:]
    elseif n%2 == 0 # if even
        tvec_middle = normalize(tvecs[div(n,2),:] + tvecs[div(n,2)+1,:])
    end
    
    angle = -atan(tvec_middle[2],tvec_middle[1])

    rvecs_rotated = zeros(size(rvecs))
    for i = 1:n
        rvecs_rotated[i,:] = rotate(rvecs[i,:],angle,[0,0,1])
    end

    hvec_rotated = rotate(hvec,angle,[0,0,1])

    return rvecs_rotated, hvec_rotated
end


function align_filament_horizontally_adjust_velocity(rvecs,vvecs,hvec)
    tvecs = make_tvecs(rvecs)

    n = size(rvecs,1)
    if n%2 == 1 # if odd
        tvec_middle = tvecs[div(n+1,2),:]
    elseif n%2 == 0 # if even
        tvec_middle = normalize(tvecs[div(n,2),:] + tvecs[div(n,2)+1,:])
    end
    
    angle = -atan(tvec_middle[2],tvec_middle[1])

    rvecs_rotated = zeros(size(rvecs))
    for i = 1:n
        rvecs_rotated[i,:] = rotate(rvecs[i,:],angle,[0,0,1])
    end

    hvec_rotated = rotate(hvec,angle,[0,0,1])

    vvecs_rotated = zeros(size(vvecs))
    for i = 1:n
        vvecs_rotated[i,:] = rotate(vvecs[i,:],angle,[0,0,1])
    end

    return rvecs_rotated, hvec_rotated, vvecs_rotated
end




function find_distance_to_filament(vec,rvecs)

    n = size(rvecs,1)

    mindist = Inf
    mindistind = 0
    for i = 1:n
        dist = norm(vec - rvecs[i,:])
        if dist < mindist
            mindist = dist
            mindistind = i
        end
    end
    
    return mindist, mindistind
end


function make_cross_product(a,b)
    # cross product of two vector functions a,b, each represented as n vectors
    # returns an n vector
    
    n = size(a,1)
    c = Array{typeof(a[1])}(undef,size(a)) #zeros(size(a))

    for i = 1:n
        avec = a[i,:]
        bvec = b[i,:]

        c[i,:] = cross(avec,bvec)
    end
    
    return c
end



function make_cross_product3n(a,b)
    # cross product of two vector functions a,b, each represented as 3n vectors
    # returns a 3n vector
    
    n = div(size(a,1), 3)
    c = zeros(size(a))

    for i = 1:n
        ax = a[3*(i-1) + 1]
        ay = a[3*(i-1) + 2]
        az = a[3*(i-1) + 3]

        bx = b[3*(i-1) + 1]
        by = b[3*(i-1) + 2]
        bz = b[3*(i-1) + 3]

        c[3*(i-1) + 1] = ay*bz - az*by
        c[3*(i-1) + 2] = az*bx - ax*bz
        c[3*(i-1) + 3] = ax*by - ay*bx
    end
    
    return c
end

function make_dot_product(a,b)
    # dot product of two vector functions a,b, each represented as n vectors
    # returns a scalar function as an n vector
    
    n = size(a,1)
    c = zeros(n)

    for i = 1:n
        avec = a[i,:]
        bvec = b[i,:]

        c[i] = dot(avec,bvec)
    end
    
    return c
end