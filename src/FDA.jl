module FDA

using Requires

export  BSpline,
        FourierBasis,
        NonOrthogonalFourierBasis,
        gcv,
        ssr,
        evaluate,
        evaluate!

# package code goes here
include("b_splines.jl")
include("fourier.jl")
include("miscellaneous.jl")

@require Plots include("plotting.jl")

end # module
