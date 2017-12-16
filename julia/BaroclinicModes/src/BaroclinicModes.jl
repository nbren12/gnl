module BaroclinicModes
using ForwardDiff
using Plots
using TensorOperations


## shallow water schemes
include("shallow_water.jl")

## Centered scheme
include("tadmor.jl")

## CWENO scheme
include("cweno.jl")

## Hamilton-Jacobi scheme
include("jiangpeng.jl")


end # module
