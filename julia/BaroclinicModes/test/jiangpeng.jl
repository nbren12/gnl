# Test of the jianpeng scheme
# 


using BaroclinicModes
using DifferentialEquations
using Plots


## test weno3_lrd
q = zeros(5, 100)
l = copy(q)
r = copy(q)

BaroclinicModes.weno3_lrd!(l, r, q)



## scalar advection problem
H(u) = u
H_llf(up, um) = H((up+um)/2.0) - 1.1*(up -um)/2.0

function f(t, u)
    n = inv(length(u) - 4)
    u[1:2] = u[end-3:end-2]
    u[end-1:end] = u[3:4]

    l = copy(u)
    r = copy(u)

    weno3_lrd!(l, r, u)

    l/=h
    r/=h

    -H_llf(r,l)
end



n = 100
h = inv(n)


x = (-2:n+1)*h

pmod(x, L=1.0) = (y = x%L; y < 0 ? y + L : y)

u(x,t) = exp(-(pmod.(x-t) - .5).^2./.1^2)
# initial condition for burgers equation
# u0(x) = sin(x*pi)/pi

tspan= (0.0,5.00)

u0 = u.(x,0.0)
prob = ODEProblem(f, u0, tspan)


sol = solve(prob, BaroclinicModes.TVD3, dt=h*.8, adaptive=false)

sol_default = solve(prob, dt=h*.8)
sol_tvd = solve(prob, TVD3, dt=h*.8, adaptive=false)

plotsol(sol; kw...) = plot(x, sol.t, hcat([sol(t) for t=sol.t]...)', st=:contourf; kw...)

plot(plotsol(sol_default, title="Default"), plotsol(sol_tvd, title="TVD Scheme"), size=(400, 800))