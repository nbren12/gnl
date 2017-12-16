abstract CentProblem

function minmod(a, b, c)
    # manual bubble sort
    if a > b a,b=b,a end
    if b > c b,c=c,b end
    if a > b a,b=b,a end

    z = zero(a)

    if a > z a
    elseif c < z c
    else z end
end

"""
Compute the Tadmor slopes with flux limiter in place

"""
function slopes!{T}(qx::Array{T, 2}, q::Array{T, 2}; tht::T=0.9)
    for v=1:size(qx,2)
        for i=2:size(qx,1)-1
            right = tht * (q[i+1,v]-q[i,v])
            cent = (q[i+1,v]-q[i-1,v])/2.0
            left = tht * (q[i,v]-q[i-1,v])
            qx[i,v] = minmod(left, cent, right)
        end
    end
end


function staggeravg!{T}(out::Array{T,2}, q::Array{T, 2}, qx::Array{T, 2}; d=:forward)

    if d == :forward f = 0 end
    if d == :backward f = -1 end

    for v=1:size(qx,2)
        for i=1-f:size(qx,1)-1 + f
            out[i,v] = (q[i+f,v] + q[i+1+f,v])/2.0 + 
                       (qx[i+f,v]-qx[i+1+f,v])/8.0
        end
    end
end

function onestep!{T <: CentProblem}(p::T, lam)

    # boundary conditions
    comm!(p, p.q)

    # staggered average
    slopes!(p.qx, p.q)
    staggeravg!(p.qp, p.q, p.qx)

    # predictor step
    p.q[:] -= lam/2.0 *(apply_op(p, p.q, p.qx) - rhs(p, p.q))

    # staggered average
    slopes!(p.qx, p.q)
    staggeravg!(p.qpn, p.q, p.qx)

    # corrector step
    w1 = apply_op(p, p.qpn, p.q)
    p.qp[1:end-1,:] -= lam *(w1[2:end,:]-w1[1:end-1,:])

    # final split in time step
    p.qp[:] += lam*rhs(p, p.q)

    # Backwards staggered average
    slopes!(p.qx,p.qp)
    staggeravg!(p.q, p.qp, p.qx, d=:backward)
end

# By default there is no forcing term
rhs(::CentProblem, a...) = 0.0

# periodic boundary conditions by default
# use 2 ghost cells
function comm!{T}(::CentProblem, q::Array{T, 2})
    q[1:2,:] = q[end-3:end-2,:]
    q[end-1:end,:] = q[3:4,:]
end


immutable ScalarAdvection  <: CentProblem
    q
    qx
    qp
    qpn
end


ScalarAdvection(nx::Int) = ScalarAdvection(zeros(nx,1), zeros(nx, 1), zeros(nx, 1), zeros(nx,1))
apply_op(::ScalarAdvection, u, v) = v


immutable Burgers  <: CentProblem
    q
    qx
    qp
    qpn
end

function comm!{T}(::Burgers, q::Array{T,2}) 
    q[1:2,:] = 1.0
    q[end-1:end,:] = -1.0
end

Burgers(nx::Int) = Burgers(zeros(nx,1), zeros(nx, 1), zeros(nx, 1), zeros(nx,1))
apply_op(::Burgers, u, v) = u .* v
