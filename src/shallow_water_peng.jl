type AdvectionTerms
    L
    uu
    wu
    ut
    wt
end



# TODO Combine the u grad(u) and div(u) u terms too make the code faster
function AdvectionTerms(bt::BasisTower)

     # advection of u by u
    auu = map(intfun, Base.product(bt.u.funs, bt.u.funs, bt.u.funs))

    # advection of u by w
    awu = map(intfun, Base.product(bt.u.funs, bt.w.funs, adiff(bt.u).funs))

    # advection of t by u
    aut = map(intfun, Base.product(bt.t.funs, bt.u.funs, bt.t.funs))

    # advection of t by w
    awt = map(intfun, Base.product(bt.t.funs, bt.w.funs, adiff(bt.t).funs)) 


    # compute and apply mass matrix
    umat = inv(massmat(bt.u))
    tmat = inv(massmat(bt.t))

    @tensor begin
        buu[i,j,k] := umat[i,l] * auu[l,j,k]
        bwu[i,j,k] := umat[i,l] * awu[l,j,k]
        but[i,j,k] := tmat[i,l] * aut[l,j,k]
        bwt[i,j,k] := tmat[i,l] * awt[l,j,k]
    end

    mats = map(sparsify, (buu, bwu, but, bwt))
    AdvectionTerms(length(bt.w),mats...)
end

AdvectionTerms(L::Integer) = AdvectionTerms(BasisTower(L))

## Code generation Total Shallow Water Operator


function swterms(bt::BasisTower) 
    at = AdvectionTerms(bt)
    lt = LinearTerms(bt)

    quote
        # $(terms(at))
        $(terms(lt))
    end
end


function terms(lt::LinearTerms)

    a = quote end
    # pressure gradient
    for ((l,k), val) in collect(lt.u_tgrad)
        p = :(tx[$k, i])
        push!(a.args, :(fu[$l,i] += $val * $p))
    end

    # N^2 w term
    for ((l,k), val) in collect(lt.t_ugrad)
        w = :((-ux[$k+1,i]))
        push!(a.args, :(ft[$l,i] += $val * $w))
    end
    a
end

function terms(at::AdvectionTerms)

    a = quote end

    # advection of u by u
    for ((l,k,m), val) in collect(at.uu)
        push!(a.args, :(fu[$l,i] += $val * u[$k,i] * ux[$m,i]))
    end

    # advection of u by w
    for ((l,k,m), val) in collect(at.wu)
        push!(a.args, :(w=-ux[$k+1,i]))
        push!(a.args, :(fu[$l,i] += $val * w * u[$m,i]))
    end

    # advection of t by u
    for ((l,k,m), val) in collect(at.ut)
        push!(a.args, :(ft[$l,i] += $val * u[$k,i] * t[$m,i]))
    end

    # advection of t by w
    for ((l,k,m), val) in collect(at.wt)
        push!(a.args, :(w=-ux[$k+1,i]))
        push!(a.args, :(ft[$l,i] += $val * w * t[$m,i]))
    end

    a
end

"""
mkswop(at, :swop)

Returns an expression containing a function `swop` which can be used to compute the action of the 
shallow water system operator.

    swop(fq, q, qx)

Operates inplace on the argument `fq`.

Macros are used to generate an expression with simple loops and inplace additions.
"""
function mkswop(bt::BasisTower, name::Symbol)
    # nonlinear advection terms

    # function body
    L = length(bt.t)
    quote
    function $(name)(fq, q, qx)
        u, t = BaroclinicModes.swview(q, L=$L)  # Names need to be qualified with the module name
        ux, tx = BaroclinicModes.swview(qx, L=$L)
        fu, ft = BaroclinicModes.swview(fq, L=$L)

        fill!(fq, 0.0)

        for i=1:size(u,2)
            $(swterms(bt))
        end    
        fq
    end
    end
end
