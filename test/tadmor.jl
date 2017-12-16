# Date: Feb 1 2017
# This module solves the burgers equation using the tadmor centered scheme
# The proposed scheme does not perform very well for Burger's equation
using BaroclinicModes
using Plots
## define

nx = 100
dx = 1.0/nx

p = BaroclinicModes.ScalarAdvection(nx+4)
# p = BaroclinicModes.Burgers(nx+4)


x = (-2:nx+1)*dx

pmod(x, L=1.0) = (y = x%L; y < 0 ? y + L : y)

q(t) = exp(-(pmod.(x-t) - .5).^2./.1^2) +1
# q(t) = cos((x-t)*pi)
p.q[:,1] = q(0)

dt = .1 * dx
lam  = dt/dx

nt = round(.1/dt)



for i=1:nt
    BaroclinicModes.onestep!(p, lam)
end

plot(x, p.q)
plot!(x, q(dt*nt))