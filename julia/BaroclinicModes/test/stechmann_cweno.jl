# NonlinerLinear Shallow water solver using CWENO
# and the stechmann, majda (2008) splitting
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
    fu0, fu, ft = swview(fq, L=L)
    u0, u, t = swview(q, L=L)


    fu[1,:] = -t[1,:] + 3.0/sqrt(2) .* u[1,:] .* u[2,:]
    fu[2,:] = -t[2,:]

    ft[1,:] = -u[1,:] + sqrt(2).*u[1,:].*t[2,:] - u[2,:] .* t[1,:]/sqrt(2)
    ft[2,:] = -u[2,:] / 4.0

    fq
end

flux(q) = flux!(zero(q), q)

function vert!(fq, q, h)
    L  = (size(fq,1)-1)/2 |> Int
    fu0, fu, ft = swview(fq, L=L)
    u0, u, t = swview(q, L=L)

    nx = size(q, 2)


    for i=2:size(fq, 2)-1
        ux1 = (u[1, i+1] - u[1, i-1])/2.0/h
        ux2 = (u[2, i+1] - u[2, i-1])/2.0/h
        tx1 = (t[1,i+1] - t[1,i-1])/2.0/h
        fu[2,i] += 3.0/2.0/sqrt(2) * u[1,i] * ux2
        ft[1,i] += -(2.0 *t[2,i] * ux1 + t[1,i]*ux2/2.0)/sqrt(2)
        ft[2,i] += -(u[1,i] *tx1 - t[1,i] * ux1)/sqrt(2)/2.0
    end
    fu[:,1:2] = 0.0
    fu[:,end-1:end] = 0.0

    fq
end

function rhs(t, q, h)
    fq = semidiscrete(t, q, h, flux=flux)
    vert!(fq, q, h)
    fq
end
end

eval(funs)





n = 200
h = .01
L = n*h


x = (-2:n+1)*h

pmod(x, L=h*n) = (y = x%L; y < 0 ? y + L : y)

u(x,t) = sin(4*2*pi*x/L)
# initial condition for burgers equation
# u0(x) = sin(x*pi)/pi

tspan= (0.0,5.0)

u0 = u.(x,0.0)

q = zeros(5, n+4)
q[4,:] = u0*.4
q[2,:] = -q[4,:]


u0, uu, t = swview(q, L=2)

rhs(0.0, q, .01)


prob = ODEProblem((t,q)->rhs(t, q, h), q, tspan)

sol = solve(prob, BaroclinicModes.TVD3, dt=h*.5, adaptive=false)

#
# plotsol(sol; kw...) =
#     plot(hcat([sol(t)[5,:] for t=[0,.00, .25, .50]]...), layout=5; kw...)
plotsol(sol, i; kw...) =
    plot(hcat([sol(t)[i,:] for t=sol.t]...), st=:contourf; kw...)

plotsol(sol,4)




# anim = @animate for t=0.0:.01:1.0
#     plot(sol(t)[2,:], ylim=[0,1.0])
# end
#
# gif(anim, "out.gif", fps=10)
