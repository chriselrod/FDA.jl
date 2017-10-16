###why even have this? Why not untype?
Base.@pure value_deriv(::Val{p}, ::Val{d}) where {p,d} = Val{p-d}()
@generated demote_val(::Val{p}) where p = Val{p-1}()

#Every time a derivative is called, push on
struct BSpline{K, p, T}
  knots::K
  coefficients::Vector{Vector{T}}
  ΦᵗΦ⁻::Symmetric{T,Matrix{T}}
  buffer::Vector{T}
  mat_buffer::Matrix{T}
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

function BSpline(x::Vector{T}, y::Vector, ::Val{p}, k::Int = div(length(x),5)) where {T, p}
    BSpline(x, y, k, CardinalKnots(x, k, Val{p}()))
end



function BSpline(x::Vector{T}, y::Vector, k::Int = div(length(x),10), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    n = length(x)
    Φᵗ = zeros(promote_type(T, Float64), k, n)
    vector_buffer = Vector{T}(p+1)
    matrix_buffer = Matrix{T}(p,p)
    fillΦ!(Φᵗ, matrix_buffer, x, knots)
#    Φt, y, knots, vector_buffer
#    println("P: $p, size of Φ: $(size(Φᵗ)), y: $(size(y))")
    β, ΦᵗΦ⁻ = solve(Φᵗ, y)
    BSpline(knots, [β], ΦᵗΦ⁻, vector_buffer,                    matrix_buffer, Φᵗ, y, Val{p}(), k)
end
function BSpline(knots::K, coef::Vector{Vector{T}}, S::Symmetric{T,Matrix{T}}, vb::Vector{T}, mb::Matrix{T}, Φᵗ, y, ::Val{p}, k) where {p, K <: Knots{p}, T}
    BSpline{K, p, T}(knots, coef, S, vb, mb, Φᵗ, y, k)
end


###Slow, despite aggressive inlining. The fact it isn't reusing the calculation of lower order splines is the problem.
@inline function recursive_B_spline(x, t::Knots, ::Val{1}, k = find_k(t, x), kₓ = find_k(t, x))
    if k == kₓ ## x on small side, so small half gets dropped => x "big"
        (t[k] - x)/range(t, k, k-1)
    else ##imples k - 1 == kₓ; x on big side; big gets dropped
        (x - t[k-2])/range(t, k-1, k-2)
    end
end
@inline function recursive_B_spline(x, t::Knots, vp::Val{p}, k = find_k(t, x), kₓ = find_k(t, x)) where p
    if k == kₓ ### big x
        (t[k] - x)/range(t, k, k-p) * recursive_b_lower(x, t, demote_val(vp), k, kₓ)
    elseif k == kₓ + p
        (x - t[k-p-1])/range(t, k-1, k-p-1) * recursive_b_upper(x, t, demote_val(vp), k-1, kₓ)
    else
        (x - t[k-p-1])/range(t, k-1, k-p-1) * recursive_B_spline(x, t, demote_val(vp), k-1, kₓ) + (t[k] - x)/range(t, k, k-p) * recursive_B_spline(x, t, demote_val(vp), k, kₓ)
    end
end
@inline function recursive_b_lower(x, t::Knots, vp::Val{p}, k = find_k(t, x), kₓ = find_k(t, x)) where p
    (t[k] - x)/range(t, k, k-p) * recursive_b_lower(x, t, demote_val(vp), k, kₓ)
end
@inline function recursive_b_upper(x, t::Knots, vp::Val{p}, k = find_k(t, x), kₓ = find_k(t, x)) where p
    (x - t[k-p-1])/range(t, k-1, k-p-1) * recursive_b_upper(x, t, demote_val(vp), k-1, kₓ)
end
@inline function recursive_b_lower(x, t::Knots, vp::Val{1}, k = find_k(t, x), kₓ = find_k(t, x))
    (t[k] - x)/range(t, k, k-1)
end
@inline function recursive_b_upper(x, t::Knots, vp::Val{1}, k = find_k(t, x), kₓ = find_k(t, x))
    (x - t[k-2])/range(t, k-1, k-2)
end
