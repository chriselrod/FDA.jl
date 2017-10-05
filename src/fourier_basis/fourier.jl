abstract type AbstractFourierBasis{T} end

struct FourierBasis{T} <: AbstractFourierBasis{T}
    intercept::T
    cos_coefs::Vector{T}
    sin_coefs::Vector{T}
    y::Vector{T}
    period::T
    scale::T
    min::T
    max::T
    step::T
end
struct NonOrthogonalFourierBasis{T, K} <: AbstractFourierBasis{T}
    intercept::T
    cos_coefs::Vector{T}
    sin_coefs::Vector{T}
    y::Vector{T}
    period::T
    scale::T
    min::T
    max::T
    t::Vector{T}
end

function FourierBasis( t::AbstractVector, y::AbstractVector, args...)
    tdiff = diff(t)
    if all(isapprox.(tdiff, tdiff[1])) #Make sure things are evenly spaced.
        min_, max_ = extrema(t)
        FourierBasis(y, min_, max_, args...)
    else
        throw("Time is not evenly spaced. Coefficient matrix Φ is not orthogonal. Correct this, or call FourierBasis with Val{K} as the first argument, where K is the desired number of Fourier pairs.")
    end
end
function FourierBasis(y::AbstractVector, min_, max_)
    span = max_ - min_
    FourierBasis(y, min_, max_, span, span)
end
function FourierBasis(y::AbstractVector{T}, min_, max_, period, span = max_ - min_) where T
    n = length(y)
    target_length = convert(Int, cld(span, period)) * n
    if target_length > n
        sizehint!(y, target_length)
        for i ∈ n:target_length - 1
            push!(y, zero(T))
        end
    end
    ffty = fft(y) ./ n
    p = div(n, 2)
    cos_coefs = Vector{T}(p)
    sin_coefs = Vector{T}(p)
    for i ∈ 1:p
        cos_coefs[i] =  2*real(ffty[i+1])
        sin_coefs[i] = -2*imag(ffty[i+1])
    end
    scale = 2π/period
    FourierBasis{T}(real(ffty[1]), cos_coefs, sin_coefs, y, period, scale, min_, max_, scale * span / n)
end
@inline FourierBasis(vK::Val{K}, args...) where K = NonOrthogonalFourierBasis(vK, args...)
function NonOrthogonalFourierBasis(vK::Val{K}, t::AbstractVector, y::AbstractVector) where K
    min_, max_ = extrema(t)
    NonOrthogonalFourierBasis(vK, t, y, max_ - min_, min_, max_)
end
function NonOrthogonalFourierBasis(::Val{K}, t::AbstractVector, y::AbstractVector{T}, period, min_ = minimum(t), max_ = maximum(t)) where {T, K}
    n = length(y)
    scale = 2π / period
    t .= (t .- min_) .* scale
    Φt = ones(T, 2K + 1, n)
    for i ∈ eachindex(t)
        for j ∈ 1:K
            Φt[2j, i] = cos( j*t[i] )
            Φt[2j+1, i] = sin( j*t[i] )
        end
    end
    β, XtXⁱ = solve(Φt, y)
    cos_coefs = Vector{T}(K)
    sin_coefs = Vector{T}(K)
    for i ∈ 1:K
        cos_coefs[i] = β[2i]
        sin_coefs[i] = β[2i+1]
    end
    NonOrthogonalFourierBasis{T, K}( β[1], cos_coefs, sin_coefs, y, period, scale, min_, max_, t )
end


#Perhaps make it so you fill(x, intercept) when creating a new instead of similiar fill! later.

@inline evaluate(s::FourierBasis, x::AbstractArray, args...) = evaluate!(similar(x), s, x, args...)
@inline evaluate(s::NonOrthogonalFourierBasis{T,K}, x::AbstractArray, args...) where {T, K} = evaluate!(similar(x), s, x, K, args...)
#@inline evaluate(s::NonOrthogonalFourierBasis{T, K}, x) where {T, K} = evaluate(s, x, K)
function evaluate!(out, s::NonOrthogonalFourierBasis, x::AbstractArray, i::Int)
    evaluateFourier!(out, s, x, i)
end
function evaluate!(out, s::FourierBasis, x::AbstractArray, i::Int)
    evaluateFourier!(out, s, x, i)
end
function evaluate!(out, s::FourierBasis, x::AbstractArray, vd::Val{d}) where d
    add_derivative_coefs!(s, vd)
    evaluateFourier!(out, s.buffer, x, s.knots, s.coefficients[d+1], value_deriv(Val{p}(), Val{d}()))
end

evaluate(s::NonOrthogonalFourierBasis, x) = evaluate_scaled(s, (x - s.min) * s.scale)
evaluate(s::FourierBasis, x, num_pairs) = evaluate_scaled(s, (x - s.min) * s.scale, num_pairs)


function evaluate_scaled(s::NonOrthogonalFourierBasis{T, K}, x::T) where {T <: Real, K}
    out = s.intercept
    for j ∈ 1:K
        out += s.cos_coefs[j] * cos(j*x) + s.sin_coefs[j] * sin(j*x)
    end
    out
end
function evaluate_scaled(s::AbstractFourierBasis, x::T, num_pairs) where {T <: Real}
    out = s.intercept
    for j ∈ 1:num_pairs
        out += s.cos_coefs[j] * cos(j*x) + s.sin_coefs[j] * sin(j*x)
    end
    out
end

function evaluateFourier!(out, s, x, num_pairs)
    fill!(out, s.intercept)
    for i ∈ eachindex(x)
        xt = (x[i] - s.min) * s.scale
        for j ∈ 1:num_pairs
            out[i] += s.cos_coefs[j] * cos(j*xt) + s.sin_coefs[j] * sin(j*xt)
        end
    end
    out
end

function ssr(s::FourierBasis{T}, K::Int) where T
    out = zero(T)
    for i ∈ eachindex(s.y)
        out += abs2( s.y[i] - evaluate_scaled(s, (i-1)*s.step, K) )
    end
    out
end
function ssr(s::NonOrthogonalFourierBasis{T, K}) where {T, K}
    out = zero(T)
    for i ∈ eachindex(s.y)
        out += abs2( s.y[i] - evaluate_scaled(s, s.t[i]) )
    end
    out
end

function gcv(s::FourierBasis, K::Int)
    ssr(s, K) * length(s.y) / (length(s.y) - 2K - 1)^2
end
function gcv(s::NonOrthogonalFourierBasis{T, K}) where {T, K}
    ssr(s) * length(s.y) / (length(s.y) - 2K - 1)^2
end
