# NonlinerLinear Shallow water solver using CWENO
#
#
# Kurganov and Levy (2000)
using BaroclinicModes
using DifferentialEquations
using Plots

bt = BasisTower(2)
fluxterms, linterms, vertterms = swexpressions(bt)


funs = quote
function flux!(fq, q)
    L  = (size(fq,1)-1)/2 |> Int
    fu, ft = swview(fq, L=L)
    u, t = swview(q, L=L)

    for i=1:size(u,2)
        $linterms
        $fluxterms
    end

    fq
end

flux(q) = flux!(zero(q), q)

function vert!(fq, q, h)
    L  = (size(fq,1)-1)/2 |> Int
    fu, ft = swview(fq, L=L)
    u, t = swview(q, L=L)

    nx = size(q, 2)
    w = zeros(L, nx)

    for v=1:L
        for i=2:size(fq, 2)-1
            w[v, i] = (u[v+1, i-1] - u[v+1, i+1])/2.0/h
        end
    end

    for i=1:size(fq,2)
        $vertterms
    end
    fq
end

function rhs(t, q, h)
    fq = semidiscrete(t, q, h, flux=flux)
    vert!(fq, q, h)

    for v=1:size(fq,1)
        for i=2:size(fq,2)-1
            fq[v,i] +=( q[v,i+1] - 2.0*q[v,i]+ q[v,i-1])
        end
    end
    fq[1,:] = 0.0
    fq
end
end

eval(funs)





n = 200
h = .01
L = n*h


x = (-2:n+1)*h

pmod(x, L=h*n) = (y = x%L; y < 0 ? y + L : y)

u(x,t) = sin(2*pi*x/L)
# initial condition for burgers equation
# u0(x) = sin(x*pi)/pi

tspan= (0.0,5.0)

u0 = u.(x,0.0)

q = zeros(5, n+4)
q[4,:] = u0*.1
q[2,:] = -q[4,:]

prob = ODEProblem((t,q)->rhs(t, q, h), q, tspan)

sol = solve(prob, BaroclinicModes.TVD3, dt=h*.5, adaptive=false)




# plotsol(sol; kw...) =
#     plot(hcat([sol(t)[5,:] for t=[0,.00, .25, .50]]...), layout=5; kw...)
plotsol(sol; kw...) =
    plot(hcat([sol(t)[3,:] for t=sol.t]...), st=:contourf; kw...)

plotsol(sol)


# anim = @animate for t=0.0:.01:1.0
#     plot(sol(t)[2,:], ylim=[0,1.0])
# end
#
# gif(anim, "out.gif", fps=10)
