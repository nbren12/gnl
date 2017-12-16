#
# Code generators for nonlinear shallow water C-WENO scheme
#
# This section follows the splitting approach developed by Stechmann and Majda.
# The shallow water problem is split up in the following way:
#
#     q_t + (f(q))_x + A(q)q_x = 0
#
# The conservative terms are handled using the C-WENO scheme, while the non-conservative terms A(q) q_x, are computed using centered differences. The splitting above is arbitrary, but in practice we will let the nonconservative terms be those related to vertical advection.
#
# See Equations 15- 19 in the report from Dec 10, 2016 for more details about the formulation here.
#

export swexpressions

## Type for building expressions involving assigned products
type Prod{N}
    i::Pair{Symbol, Int} # Output variable
    j::NTuple{N, Pair{Symbol, Int}} # Input variables
    val::Float64
    iter
end


output(p::Prod) = Expr(:ref, p.i[1], p.i[2], p.iter...)
function exprprod(p::Prod)
    Expr(:call, :*, p.val, [Expr(:ref, j[1], j[2], p.iter...) for j in p.j]...)
end

"""
assignment_expr(p::Prod; op=:(+=))

# Examples

```jldoctest
julia> p  = Prod( :u=>1, (:u=>2, :u=>1), 1.0, (:i,))
Prod{2}(:u=>1,(:u=>2,:u=>1),1.0,(:i,))

julia> assignment_expr(p)
:(u[1,i] += 1.0 * u[2,i] * u[1,i])
```
"""
assignment_expr(p::Prod; op=:(+=)) = Expr(op, output(p), exprprod(p))



# compute the expressions for the flux and vertical terms
function swexpressions(bt::BasisTower)

     # advection of u by u
    Au = map(intfun, Base.product(bt.u.funs, bt.u.funs, bt.u.funs))

    # advection of u by w
    Bu = map(intfun, Base.product(bt.u.funs, bt.w.funs, adiff(bt.u).funs))

    # advection of t by u
    At = map(intfun, Base.product(bt.t.funs, bt.u.funs, bt.t.funs))

    # advection of t by w
    Bt = map(intfun, Base.product(bt.t.funs, bt.w.funs, adiff(bt.t).funs))


    # compute and apply mass matrix
    umat = inv(massmat(bt.u))
    tmat = inv(massmat(bt.t))


    Au /= 2.0
    divterms = copy(At)
    divterms[:,2:end]
    Bt += At[2,]

    @tensor begin
        # velocity equation
        fluxu[i,j,k] := umat[i,l] * Au[l,j,k]
        vertu[i,j,k] := umat[i,l] * Bu[l,j,k]

        # temperature equation
        fluxt[i,j,k] := tmat[i,l] * At[l,j,k]
        vertt[i,j,k] := tmat[i,l] * Bt[l,j,k]
    end

    itervars= [:i]
    flux_terms = quote end
    push!(flux_terms.args, mat2expr(fluxu, itervars, :fu, :u, :u)...)
    push!(flux_terms.args, mat2expr(fluxt, itervars, :ft, :u, :t)...)

    vert_terms = quote end
    push!(vert_terms.args, mat2expr(vertu, itervars, :fu, :w, :u)...)
    push!(vert_terms.args, mat2expr(vertt, itervars, :ft, :w, :t)...)


    ## Linear Terms
    lt = LinearTerms(bt)
    lin_terms = quote end

    # pressure gradient
    for ((l,k), val) in collect(lt.u_tgrad)
        push!(lin_terms.args, :(fu[$l,i] += $val * t[$k, i]))
    end

    # N^2 w term
    for ((l,k), val) in collect(lt.t_ugrad)
        k+=1
        push!(lin_terms.args, :(ft[$l,i] += $val * u[$k,i]))
    end

    flux_terms, lin_terms, vert_terms
end

function Prod(it, itervars, outvar, invars...)
    ijk = it.first; val = it.second;
    out = outvar=>ijk[1];
    ins = map((o,j)->o=>j, invars, ijk[2:end]);
    Prod(out, tuple(ins...), val, tuple(itervars...))
end

"""
mat2prods(mat, [:i, :j], :u, :ux, :w)
"""
function mat2prods(mat, args...) 
    smat = sparsify(mat)
    [Prod(it,args...) for it in smat]
end

mat2expr(mat, args...) = map(assignment_expr, mat2prods(mat, args...))