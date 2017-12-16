# Linear Shallow water solver using CWENO
#
#
# Kurganov and Levy (2000)
using BaroclinicModes
using DifferentialEquations
using Plots

"""
Linear shallow water flux
"""

# function flux!(fq, q)
    

#     # u0 u1 u2 t1 t2
#     # 1  2  3  4  5

#     fq[2,:]=  -q[4, :]
#     fq[3,:]=  -q[5, :]
#     fq[4,:] = -q[2, :]
#     fq[5,:] = -q[3, :]
#     fq
# end

bt = BasisTower(2)
fluxterms, linterms, vert = swexpressions(bt)


macro evalbody(lin)
    eval(lin)
end

function flux!(fq, q)
    L  = (size(fq,1)-1)/2 |> Int
    fu, ft = swview(fq, L=L)
    u, t = swview(q, L=L)

    for i=1:size(u,2)
        @evalbody linterms
    end

    fq
end

flux(q) = flux!(zero(q), q)



n = 100
h = inv(n)


x = (-2:n+1)*h

pmod(x, L=1.0) = (y = x%L; y < 0 ? y + L : y)

u(x,t) = exp(-(pmod.(x-t) - .5).^2./.1^2)
# initial condition for burgers equation
# u0(x) = sin(x*pi)/pi

tspan= (0.0,1.0)

u0 = u.(x,0.0)

q = zeros(5, n+4)
q[4,:] = u0

prob = ODEProblem((t,q)->semidiscrete(t, q, h, flux=flux), q, tspan)

sol = solve(prob, BaroclinicModes.TVD3, dt=h*.5, adaptive=false)


plotsol(sol; kw...) =
    plot(hcat([sol(t)[4,:] for t=[0,.25, .5, .75, 1.0]]...), layout=5; kw...)

plotsol(sol)


# anim = @animate for t=0.0:.01:1.0
#     plot(sol(t)[2,:], ylim=[0,1.0])
# end
#
# gif(anim, "out.gif", fps=10)
