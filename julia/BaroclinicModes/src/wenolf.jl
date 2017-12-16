abstract WenoLFProblem 

function weno3_d(phi)
    phi[2:end] = 
end



# periodic boundary conditions by default
# use 2 ghost cells
function comm!{T}(::CentProblem, q::Array{T, 2})
    q[1:2,:] = q[end-3:end-2,:]
    q[end-1:end,:] = q[3:4,:]
end


immutable ScalarAdvection  <: WenoLFProblem
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
