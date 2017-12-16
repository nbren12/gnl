# Test of the jianpeng scheme
# 
using BaroclinicModes
using DifferentialEquations
using Plots




## scalar advection problem
function f(q)
    fq = zero(q)
    fq[1,:]  = q[2,:]
    fq[2,:]  = q[1,:]

    fq
end
H_llf(up, um) = H((up+um)/2.0) - 2*(up -um)/2.0

function f(t, u)
    # u = reshape(u, 2, Int(length(u)/2))
    @show t
    h = inv(size(u,2) - 4)

    l = copy(u)
    r = copy(u)
    weno3_lrd!(l, r, u)

    l/=h
    r/=h
    fu = copy(u)

    a = 1
    fu[1,:] =  (l[2,:] + a * l[1,:])/2.0 + (r[2,:] - a* r[1,:])/2.0
    fu[2,:] =  (l[1,:] + a * l[2,:])/2.0 + (r[1,:] - a* r[2,:])/2.0

    -fu
end


## test weno3_lrd


n = 100
h = inv(n)


x = (-2:n+1)*h

pmod(x, L=1.0) = (y = x%L; y < 0 ? y + L : y)

u(x,t) = exp(-(pmod.(x-t) - .5).^2./.1^2)
# initial condition for burgers equation
# u0(x) = sin(x*pi)/pi

tspan= (0.0,.30)

u0 = u.(x,0.0)

q = zeros(2, n+4)
q[2,:] = u0

l = copy(q)
r = copy(q)
weno3_lrd!(l, r, q)

prob = ODEProblem(f, q, tspan)


sol = solve(prob, BaroclinicModes.TVD3, dt=h*.1, adaptive=false)


plotsol(sol; kw...) = plot(x, sol.t, hcat([sol(t)[1,:] for t=sol.t]...)', st=:contourf; kw...)

plotsol(sol)