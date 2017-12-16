# Code for CWENO scheme from Kurganov and Levy
using DifferentialEquations


export recon!, semidiscrete

"""
Compute the third-order accurate centered WENO derivative at a particular location

See (2.8) and (2.9) in Kurganov and Levy  (2000)
"""
function cweno3_dx!(d,u)
    eps = 1e-6
    for  i=2:length(u)-1

        # derivatives
        l= u[i] - u[i-1]
        r= u[i+1] - u[i]
        c= (u[i+1] -  u[i-1])/2.0

        # smoothness indicator
        s1 = l*l
        s2 = 13.0/3.0 * (u[i+1] -2*u[i] + u[i-1])^2 + c^2
        s3 = r*r


        # weights
        a1 = 0.25/(eps + s1)^2
        a2 = 0.5/(eps + s2)^2
        a3 = 0.25/(eps + s3)^2

        s = a1 + a2+a3

        a1 /= s
        a2 /= s
        a3 /= s

        # right derivatives
        l = l
        r = r
        c = (u[i+1] - u[i-1])/2 + (u[i+1] -2*u[i] + u[i-1])


        # weighted product
        d[i] = a1*l +a2*r + a3*c
    end
    d
end

"""
recon!(l, r, q)

Compute the left-biased and right-biased WENO-3 reconstructions.
"""
function recon!(l, r, q)
    eps = 1e-6
    aj = bj = cj = 0
    for j in 2:size(q,1)-1

        # smoothness indicators
        s1 = (q[j]-q[j-1])^2
        s2 = 13.0/3.0*(q[j+1]-2*q[j]+q[j-1])^2 + (q[j+1]-q[j-1])^2/4.0
        s3 = (q[j+1]-q[j])^2

        # weights
        a1 = .25 /(eps + s1)^2
        a2 = .50 /(eps + s2)^2
        a3 = .25 /(eps + s3)^2

        s = a1 + a2 + a3
        a1 /= s
        a2 /= s
        a3 /= s

        # polynomial weights
        aj = q[j] -  a2/12.0*(q[j+1] - 2*q[j] + q[j-1])
        bj = (a1*(q[j+1]-q[j]) + a2 * (q[j+1]-q[j-1])/2.0 +
             a3 * (q[j]-q[j-1]))/2.0
        cj = a2 * (q[j+1]-2*q[j] +q[j-1])/4.0

        # left reconstruction
        l[j] = aj + bj + cj

        # right reconstruction uses aj,bj,cj from previous iterations
        r[j-1] = aj - bj + cj
    end
    l,r
end

function semidiscrete{T}(t, q::AbstractArray{T, 2}, h; flux=nothing)
    fq = copy(q)
    r = copy(q)
    l = copy(q)

    # periodic boundary conditions
    q[:,1:2] = q[:,end-3:end-2]
    q[:,end-1:end] = q[:,3:4]

    neq, nx = size(q)

    # perform reconstruction
    for v in 1:neq
        lv = view(l, v, :)
        rv = view(r, v, :)
        qv = view(q, v, :)

        recon!(lv, rv, qv)
    end

    # compute flux
    fl = flux(l)
    fr = flux(r)
    f = (fl + fr)/2.0 - 1.0 * (r -l)/2.0

    # take flux difference
    for v in 1:neq
        for j in 2:size(q,2)
            fq[v, j] = (f[v, j-1]-f[v,j])/h
        end
    end
    fq
end
