from pylab import *


def JacobiIter(A,b,v,iter=1000):
    x  = v
    D = diag(A)
    LU = A-diagflat(D)
    for i in range(iter):
        x1 = b-dot(LU,x)
        x1 = x1/D
        x = x1
    return x

def VCycle(A,b,v):
    m = A.shape[0]
    if m == 4:
        v = solve(A,b)
    else:
        b2h = (b-dot(A,v))[0::2]
        v2h = zeros(m/2)
        A2h = (-2*diag(ones(m/2),0) + diag(ones(m/2-1),-1) +diag(ones(m/2-1),1))*(m/2)**2
        v2h = VCycle(A2h,b2h,v2h)
        v = v+ interp(range(m),range(0,m,2),v2h)
        v = JacobiIter(A,b,v)
    return v



m =2**12
thresh = .01/m**2

A = (-2*diag(ones(m),0) + diag(ones(m-1),-1) +diag(ones(m-1),1))*m**2
b = rand(m)*100
# Jacobi
x = JacobiIter(A,b,zeros(m))
v = VCycle(A,b,zeros(m))

y = solve(A,b)

