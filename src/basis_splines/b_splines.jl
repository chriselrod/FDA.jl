###why even have this? Why not untype?
Base.@pure value_deriv(::Val{p}, ::Val{d}) where {p,d} = Val{p-d}()
@generated demote_val(::Val{p}) where p = Val{p-1}()

#Every time a derivative is called, push on
struct BSpline{K, p, T}
  knots::K
  coefficients::Vector{Vector{T}}
  ΦᵗΦ⁻::Symmetric{T,Matrix{T}}
  buffer::Matrix{T}
  Φᵗ::Matrix{T}
  y::Vector{T}
  k::Int
end

"""
k is the number of basis.
"""
function BSpline(x::AbstractVector{T}, y, k::Int, extra_knots::Vector, vp::Val{p} = Val{3}()) where {T, p}
    issorted(extra_knots) || sort!(extra_knots)
    uek = unique(extra_knots)
    ekn = length(extra_knots)
    l = k - p + 1 - ekn

    per = l / (length(uek) + 1)

    println(per)
    min_, max_ = extrema(x)
    knot_vector = Vector{T}(k + p + 1)
    knot_vector[1:p] .= min_
    ind = p

    for j ∈ linspace(min_, uek[1], round(Int, per)+1)
        ind += 1
        knot_vector[ind] = j
    end
    for i ∈ 1:length(uek)-1
        for j ∈ 2:sum(extra_knots .== uek[i])
            ind += 1
            knot_vector[ind] = uek[i]
        end

        for j ∈ linspace(uek[i], uek[i+1], 1 + round(Int, (i+1)* per) - round(Int, (i)* per))
            ind += 1
            knot_vector[ind] = j
        end
    end
    i = length(uek)
    for j ∈ 2:sum(extra_knots .== uek[i])
        ind += 1
        knot_vector[ind] = uek[i]
    end
    for j ∈ linspace(uek[i], max_, 1 + round(Int, (i+1)* per) - round(Int, (i)* per))
        ind += 1
        knot_vector[ind] = j
    end

    knot_vector[end-p-1:end] .= max_
    println(knot_vector)

    BSpline(x, y, k, DiscreteKnots(vp, knot_vector))
#    return DiscreteKnots(vp, knot_vector)
end

function BSpline(x::Vector, y::Vector, knotv::Vector, vp::Val{p} = Val{3}()) where p
    knots = DiscreteKnots(knotv, vp)
    k = length(knots.v) - p - 1
    BSpline(x, y, k, knots)
end

function BSpline(x::Vector{T}, y::Vector, ::Val{p} = Val{3}(), k::Int = div(length(x),10)) where {T, p}
    simultaneous_sort!(x, y)
    BSpline(x, y, k, CardinalKnots(x, k, Val{p}()))
end
function BSpline(x::Vector{T}, y::Vector{T}, k::Int = div(length(x),10)) where {T, p}
    simultaneous_sort!(x, y)
    BSpline(x, y, k, CardinalKnots(x, k, Val{3}()))
end

function BSpline(x::Vector{T}, y::Vector{T}, k::Int = div(length(x),10), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    m = p+1
    simultaneous_sort!(x, y)
    x_member, min_k, max_k = k_structure!(x, y, knots, p)
    n = length(x)
    Φᵗ = zeros(promote_type(T, Float64), m, n)
    fillΦ_sorted!(Φᵗ, x, knots, x_member)
    Φ = Φᵗ'
    band = Array{Float64, 2}(uninitialized, m, k)
    fill_band!(band, Φ, min_k, max_k, Val{p}())
    β, ΦᵗΦ⁻ = solve(Φᵗ, y)
    BSpline(knots, [β], ΦᵗΦ⁻, Φ, Φᵗ, y, Val{p}(), k)
end
@inline function BSpline(knots::K, coef::Vector{Vector{T}}, S::Symmetric{T,Matrix{T}}, b::Matrix{T}, Φᵗ, y, ::Val{p}, k) where {p, K <: Knots{p}, T}
    BSpline{K, p, T}(knots, coef, S, b, Φᵗ, y, k)
end
