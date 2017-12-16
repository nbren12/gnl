#  This module defines details of the shallow water scheme
#
#
#
#
#
export BasisTower, mkswop, swview

import Base.length

## Vertical basis code
type VerticalBasis
    funs::Vector{Function}
end

length(x::VerticalBasis)= length(x.funs)

# Derivative of basis
adiff(vb::VerticalBasis) =
    VerticalBasis([z->ForwardDiff.derivative(fun, z) for fun in vb.funs])

type BasisTower
    w::VerticalBasis
    u::VerticalBasis
    t::VerticalBasis
end

type SIN end

BasisTower(L::Integer) = BasisTower(SIN(), L)
BasisTower(::SIN, L::Integer) = BasisTower(VerticalBasis(WB(), L),
                                           VerticalBasis(UB(), L),
                                           VerticalBasis(TB(), L))

type UB end
type TB end
type WB end

# dispatch on symbol input
VerticalBasis(sym::Symbol, args...) = VerticalBasis(Type{sym}, args...)
VerticalBasis(::WB, L) = VerticalBasis([z->sqrt(2)* sin(i*z)/i for i=1:L])
VerticalBasis(::TB, L) = VerticalBasis([z->-sqrt(2) *sin(i*z)*i for i=1:L])
VerticalBasis(::UB, L) =
    VerticalBasis([z->1, [z->sqrt(2) *cos(i*z) for i=1:L]...])

## Plotting recipes
@recipe function f(vb::VerticalBasis)
    vb.funs, 0, pi
end

@recipe function f(bt::BasisTower)
    layout := 3


    @series  begin
        subplot:=1
        bt.w
    end
    @series  begin
        subplot:=2
        bt.u
    end

    @series  begin
        subplot:=3
        bt.t
    end
end


"compute sparse structure"
sparsify(a::Array) = (mask = abs(a) .> 1e-10; zip(zip(findn(mask)...), a[mask]) |> Dict)

"""
Take vertical average of product of functions
"""
intfun(a) = quadgk(z->prod(map(a->a(z), a)), 0, pi, abstol=1e-10)[1]/pi

massmat(vb::VerticalBasis) = map(intfun, Base.product(vb.funs, vb.funs))
# return views of the data array q
swview{T}(q::AbstractArray{T, 2}; L=2) = view(q, 1,:), view(q, 2:L+1, :), view(q,L+2:2*L+1,:)


# linear terms
include("linear_terms.jl")

# advection terms which are compatible with the Peng scheme
include("shallow_water_peng.jl")

# CWENO advection terms
include("shallow_water_cweno.jl")
