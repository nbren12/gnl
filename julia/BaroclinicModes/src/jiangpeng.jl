#
# Implement the scheme described in Jian and Peng (2000)
# 
using DifferentialEquations


export weno3_lrd!, TVD3

function weno3_ld(u)
    eps = 1e-6
    du = u[2:end] -u[1:end-1]
    d2u = du[2:end] -du[1:end-1]

    r = (eps + d2u[1]^2)/(eps+d2u[2]^2)
    w = 1.0 /(1.0+2.0*r^2)

    (du[2] + du[3])/2.0 - w/2.0 *(du[1] - 2 * du[2] + du[3])
end

function weno3_rd(u)
    eps = 1e-6
    du = u[2:end] -u[1:end-1]
    d2u = du[2:end] -du[1:end-1]

    r = (eps + d2u[2]^2)/(eps+d2u[1]^2)
    w = 1.0 /(1.0+2.0*r^2)

    (du[1] + du[2])/2.0 - w/2.0 *(du[2] - 2 * du[2] + du[1])
end

# weno3_rd(u) = weno3_ld(reverse(u))

typealias A1{T} AbstractArray{T, 1}
typealias A2{T} AbstractArray{T, 2}
typealias A3{T} AbstractArray{T, 3}

function weno3_lrd!(l::A1, r::A1, u::A1)
    for i=3:length(u)-2
        l[i] = weno3_ld(view(u, i-2:i+1))
        r[i] = weno3_rd(view(u, i-1:i+2))
    end
end

function weno3_lrd!(l::A2, r::A2, u::A2)
    nv, nx = size(u)
    for v=1:nv
        lv, rv, uv = map(x->view(x, v, :), [l, r, u])
        weno3_lrd!(lv, rv, uv)
    end
end

## TVD tableau
"""
Third order TVD from Jiang and Peng (2000) pp.8
"""
function constructTVD3(T::Type = Float64)
  A = [0 0 
       1 0  
       1//4 1//4 ]
  c = [0;1;1//2]
  α = [1//6; 1//6; 2//3]
  A = map(T,A)
  α = map(T,α)
  c = map(T,c)
  ExplicitRKTableau(A,c,α,3)
end

TVD3 = ExplicitRK(tableau=constructTVD3())