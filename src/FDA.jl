__precompile__()

module FDA

using Compat, BandedMatrices, SIMD, Base.Cartesian

export  BSpline,
        FourierBasis,
        NonOrthogonalFourierBasis,
        gcv,
        ssr,
        evaluate,
        evaluate!

# package code goes here
include("knots.jl")
include("basis_splines/b_splines.jl")
include("basis_splines/penalty.jl")
include("basis_splines/evaluate.jl")
include("basis_splines/error_estimate.jl")
include("fourier_basis/fourier.jl")
include("miscellaneous.jl")



end # module
