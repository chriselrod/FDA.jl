__precompile__()

module FDA

using Requires, BandedMatrices

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
include("fourier_basis/fourier.jl")
include("miscellaneous.jl")

@require NullableArrays BSpline(x::NullableArrays.NullableArray, y::NullableArrays.NullableArray, a...) = BSpline(convert(Vector{Float64}, x), convert(Vector{Float64}, y), a...)
@require Plots include("plotting.jl")


end # module
