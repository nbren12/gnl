
## Linear terms code

"""
Encodes the linear terms of the shallow water operator

# Note 

This logic is kind of tricky depending on whether the barotropic mode is present or not

# Example
```jldoctest
julia> bt = BaroclinicModes.BasisTower(2)
julia> BaroclinicModes.LinearTerms(bt).t_ugrad
Dict{Tuple{Int64,Int64},Float64} with 2 entries:
  (1,2) => 1.0
  (2,3) => 0.25

julia> BaroclinicModes.LinearTerms(bt).u_tgrad
Dict{Tuple{Int64,Int64},Float64} with 2 entries:
  (2,1) => 1.0
  (3,2) => 1.0
```
"""
type LinearTerms
    u_tgrad
    t_ugrad
    with_barotropic::Bool
end

# the convection is that the linear terms on the LHS
function LinearTerms(bt::BasisTower)

    # mass matrices
    tmat = inv(massmat(bt.t))
    umat = inv(massmat(bt.u))


    # divergence
    w = map(intfun, Base.product(bt.t.funs, bt.w.funs)) 

    nu  = length(bt.u)
    nt  = length(bt.t)

    # initialize matrices
    t_ugrad = zeros(eltype(umat), nt, nu)
    u_tgrad = zeros(eltype(umat), nu, nt)

    # If barotropic mode is included
    if nu  ==  nt +1
        u_tgrad[2:end,:] = umat[2:end, 2:end]
    elseif nu == nt
        u_tgrad = umat
    else
        throw(ArgumentError("Size of basis is different"))
    end
    t_ugrad = -tmat*w
    u_tgrad = sparsify(u_tgrad)
    t_ugrad = sparsify(t_ugrad)

    LinearTerms(u_tgrad, t_ugrad, length(bt.u) > length(bt.t))
end

