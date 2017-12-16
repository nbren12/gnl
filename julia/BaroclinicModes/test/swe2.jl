using BaroclinicModes
using DifferentialEquations
using Plots




## Some basic tests of the interface
mkswop(BasisTower(2), :swop2!)

eval(mkswop(BasisTower(2), :swop2!))
eval(mkswop(BasisTower(4), :swop4!))

q = rand(9, 1000)
qx = copy(q)
fq = zero(q)


# @time swop2!(fq, q, qx)
# @time swop4!(fq, q, qx)

## SWE problem
H(q, qx) = (fq =zero(q); swop2!(fq, q, qx))
H_llf(q, up, um) = H(q, (up+um)/2.0) - 1.5(up -um)/2.0

function f(h,l,r,t,q)
    q[:,1:2] = q[:,end-3:end-2]
    q[:,end-1:end] = q[:, 3:4]


    weno3_lrd!(l, r, q)

    l/=h
    r/=h

    -H_llf(q,r,l)
end




# computational grid
n = 100
h = inv(n)


x = (-2:n+1)*h

# initial conditions

pmod(x, L=1.0) = (y = x%L; y < 0 ? y + L : y)


q = zeros(5,length(x))
u,th = swview(q, L=2)

t1 = -exp(-(x-.5).^2./.1^2) * .001
th[1,:] = t1


l = zero(q)
r = zero(q)

rhs(t, q) = f(h,l,r,t,q)

# setup solver object
tspan= (0.0,1.00)
prob = ODEProblem(rhs, q, tspan)


sol = solve(prob, TVD3, dt=h*.01, adaptive=false)

plot([sol.u[end][4,:]])
# plot(hcat([sol(t)[4,:] for t in sol.t]...), st=:contourf)