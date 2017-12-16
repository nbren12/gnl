using Base.Test

wb = VerticalBasis(BaroclinicModes.WB(), 2)
ub = adiff(wb)
tb = adiff(ub)

bt = BasisTower(2)


## Nonlinear advection terms
adv = BaroclinicModes.AdvectionTerms(bt)


# Solve the linear shallow water equations using the CWENO scheme
# include("linshallow.jl")
