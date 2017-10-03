abstract type Knots{p} end

Base.@pure Base.Val(T) = Val{T}()


###why even have this? Why not untype?
Base.@pure value_deriv(::Val{p}, ::Val{d}) where {p,d} = Val{p-d}()

"P is the degree of the polynomial fit; order is p+1."
struct CardinalKnots{p,T} <: Knots{p}
    min::T
    max::T
    v::T
    n::Int
end
struct DiscreteKnots{p,T} <: Knots{p}
    min::T
    max::T
    v::Vector{T}
    n::Int
end
function CardinalKnots(x::Vector{T}, k::Int, ::Val{p}) where {p,T}
    miv, mav = extrema(x)
    nm1 = k - p
    δ = (mav - miv)/ nm1
    δpoo5 = δ * 0.005
    CardinalKnots{p,T}( miv - δpoo5, mav + δpoo5, δ*(100nm1+1)/100nm1, nm1+1)
end
function CardinalKnots(minval::T, maxval::T, n::Int, ::Val{p}) where {p,T}
    S = promote_type(T, Float64)
    CardinalKnots{p,S}(convert(S, minval), convert(S, maxval), (maxval-minval)/(n-1), n)
end
function DiscreteKnots(::Val{p}, v::Vector{T}) where {p,T}
    n = length(v)
    DiscreteKnots{p,T}(v[1], v[n], v, n)
end
function DiscreteKnots(v::Vector{T}, ::Val{p}) where {p,T}
    issorted(v) || sort!(v)
    n = length(v)
    minval = v[1]
    maxval = v[n]
    n = length(v)
    if v[2] != minval
        sizehint!(v, n + 2p)
        for i ∈ 1:p
            unshift!(v, minval)
            push!(v, maxval)
        end
    end
    DiscreteKnots{p,T}(minval, maxval, v, n)
end
DiscreteKnots(v::StepRangeLen, ::Val{p}) where p = CardinalKnots{p,Float64}(v[1], v[end], convert(Float64, v.step), length(v))


@inline function Base.getindex(t::CardinalKnots{p,T}, i)::T where {p,T}
    impm1 = i - p - 1
    if impm1 <= 0
        return t.min
    elseif impm1 >= t.n - 1
        return t.max
    end
    t.min + impm1 * t.v
end
@inline Base.getindex(t::DiscreteKnots, i) = t.v[i]




#Every time a derivative is called, push on
struct BSpline{K, p, T}
  knots::K
  coefficients::Vector{Vector{T}}
  ΦᵗΦ⁻::Symmetric{T,Matrix{T}}
  buffer::Vector{T}
  mat_buffer::Matrix{T}
  Φᵗ::Matrix{T}
  y::Vector{T}
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

@require NullableArrays BSpline(x::NullableArrays.NullableArray, y::NullableArrays.NullableArray, a...) = BSpline(convert(Vector{Float64}, x), convert(Vector{Float64}, y), a...)

function BSpline(x::Vector{T}, y::Vector, k::Int = div(length(x),10), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    n = length(x)
    Φᵗ = zeros(promote_type(T, Float64), k, n)
    vector_buffer = Vector{T}(p+1)
    matrix_buffer = Matrix{T}(p,p)
    fillΦ!(Φᵗ, matrix_buffer, x, knots)
#    Φt, y, knots, vector_buffer
#    println("P: $p, size of Φ: $(size(Φᵗ)), y: $(size(y))")
    β, ΦᵗΦ⁻ = solve(Φᵗ, y)
    BSpline(knots, [β], ΦᵗΦ⁻, vector_buffer, matrix_buffer, Φᵗ, y, Val{p}())
end
function BSpline(knots::K, coef::Vector{Vector{T}}, S::Symmetric{T,Matrix{T}}, vb::Vector{T}, mb::Matrix{T}, Φᵗ, y, ::Val{p}) where {p, K <: Knots{p}, T}
    BSpline{K, p, T}(knots, coef, S, vb, mb, Φᵗ, y)
end

function ssr!(s::BSpline)
    sum(abs2, Base.LinAlg.BLAS.gemm!('T', 'N', 1.0, s.Φᵗ, s.coefficients[1], -1.0, s.y))
end
function ssr(s::BSpline{K, p, T}) where {K, p, T}
    out = zero(T)
    for i ∈ eachindex(s.y)
        δ = s.y[i]
        for j ∈ eachindex(s.coefficients[1])
            δ -= s.coefficients[1][j] * s.Φᵗ[j,i]
        end
        out += abs2(δ)
    end
    out
end
function confidence_width(s::BSpline{K, p, T}, x) where {K, p, T}
    k = find_k(s.knots, x)
    fillΦ3!(s.buffer, s.mat_buffer, x, s.knots, p, k)
    out = zero(T)
    for i ∈ 1:p+1
        out += s.ΦᵗΦ⁻.data[i, i] * s.buffer[i]^2
        for j ∈ 1:i-1
            out += 2s.ΦᵗΦ⁻.data[j+p+1, i+p+1] * s.buffer[i] * s.buffer[j]
        end
    end
    out
end

function scaleVar!(s::BSpline{K, p}) where {K, p}
    σ = ssr(s) / (length(s.y) - s.knots.n - p + 1)
    s.ΦᵗΦ⁻ ./= σ
end

function gcv(s::BSpline{K, p}) where {K, p}
    ssr(s) * length(s.y) / (length(s.y) - s.knots.n - p + 1)^2
end
function gcv!(s::BSpline{K, p}) where {K, p}
    nmdf = length(s.y) - s.knots.n - p + 1
    σ = ssr(s) / nmdf
    s.ΦᵗΦ⁻.data ./= σ
    σ * length(s.y) / (length(s.y) - s.knots.n - p + 1)
end

@inline evaluate(s::BSpline, x::AbstractArray, args...) = evaluate!(similar(x), s, x, args...)
function evaluate!(out, s::BSpline{K, p}, x::AbstractArray) where {K, p}
    deBoor!(out, s.buffer, x, s.knots, s.coefficients[1], Val{p}())
end

###First 5 derivates may be wrong?!?!?
#julia> d1_fd = ForwardDiff.derivative.(x -> deBoor(x, bs.knots, bs.coefficients[1], Val{3}(), find_k(bs.knots, ForwardDiff.value(x))), tt);

#julia> d2_fd = ForwardDiff.derivative.(y -> ForwardDiff.derivative(x -> deBoor(x, bs.knots, bs.coefficients[1], Val{3}(), find_k(bs.knots, ForwardDiff.value(ForwardDiff.value(x)))), y), tt);


###d1 = evaluate(bs, tt, Val{1}())
function evaluate!(out, s::BSpline{K, p}, x::AbstractArray, vd::Val{d}) where {K, p, d}
    add_derivative_coefs!(s, vd)
    deBoor!(out, s.buffer, x, s.knots, s.coefficients[d+1], value_deriv(Val{p}(), Val{d}()))
end
function evaluate(s::BSpline{K, p}, x::T, args...) where {T <: Real, K, p}
    deBoor!(s.buffer, x, s.knots, s.coefficients[1], Val{p}(), find_k(s.knots, x))
end

function add_derivative_coefs!(s::BSpline{K, p}, vd::Val{d}) where {K <: DiscreteKnots, p, d}
    for i ∈ length(s.coefficients):d #i is derivative being added
        coef = diff(s.coefficients[i])
        pmi = p - i + 1
        for j ∈ eachindex(coef)
            coef[j] *= pmi / (s.knots[j+1+pmi] - s.knots[j+1])
        end
        push!(s.coefficients, coef)
    end
end
function add_derivative_coefs!(s::BSpline{K, p}, vd::Val{d}) where {K <: CardinalKnots, p, d}
    for i ∈ length(s.coefficients):d #i is derivative being added
        coef = diff(s.coefficients[i]) ./ s.knots.v
        push!(s.coefficients, coef)
    end
end

find_k(t::DiscreteKnots, x) = searchsortedfirst(t.v, x, lt = <=)
@inline find_k(t::CardinalKnots{p}, x) where p = convert(Int, cld(x - t.min, t.v)) + p + 1


deBoor(x, t::Knots{p}, c, k = find_k(t, x)) where p = deBoor(x, t, c, Val{p}(), k)
deBoor(x::T, t, c, vp::Val{p}, k = find_k(t, x)) where {p,T} = deBoor!(Vector{promote_type(Float64,T)}(p+1), x, t, c, vp, k)

function deBoor!(d::Vector, x, t::Knots{q}, c, vp::Val{p}, k = find_k(t, x)) where {p, q}
    for j ∈ 1:p+1
        d[p+2-j] = c[j+k-q-2]
    end
    deBoorCore!(d, x, t, c, vp, k)
    d[1]
end

deBoorCore(d::Vector, x, t::Knots{p}, c, k = find_k(t, x)) where p = deBoorCore(d, x, t, c, Val{p}(), k)

#would @inline improve performance?
function deBoorCore!(d::Vector, x, t::DiscreteKnots, c, ::Val{p}, k = find_k(t, x)) where p
    for r ∈ 1:p, j ∈ 1:1+p-r
        α = (x - t[k-j]) / (t[p+1+k-r-j] - t[k-j])
        d[j] = α*d[j] + (1-α)*d[j+1]
    end
end
function deBoorCore!(d::Vector, x, t::CardinalKnots, c, ::Val{p}, k = find_k(t, x)) where p
    denom = t.v * p
    for r ∈ 1:p
        for j ∈ 1:1+p-r
            α = (x - t[k-j]) / denom
            d[j] = α*d[j] + (1-α)*d[j+1]
        end
        denom -= t.v
    end
end


function deBoor!(out::Vector, d::Vector, x::Vector, t::Knots{q}, c, Vp::Val{p}) where {q, p}
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        for j ∈ 1:p+1
            d[p+2-j] = c[j+k-q-2]
        end
        deBoorCore!(d, xᵢ, t, c, Vp, k)
        out[i] = d[1]
    end
    out
end
function sorteddeBoor!(out::Vector, d::Vector, x::Vector, t::Knots{q}, c, Vp::Val{p}) where {q, p}
    k = q+2
    t_next = t[k]
    for i ∈ eachindex(x)
        xᵢ = x[i]
        while xᵢ > t_next
            k += 1
            t_next = t[k]
        end
        for j ∈ 1:p+1
            d[p+2-j] = c[j+k-p-2]
        end
        deBoorCore!(d, xᵢ, t, c, Vp, k)
        out[i] = d[1]
    end
    out
end


function fillΦ2!(d, x, t, p, k = find_k(t, x))
  for r ∈ 1:p
    for j ∈ p:-1:r
      α = (x - t[j+k-p-1]) / (t[j+k-r] - t[j+k-p-1])
      for i ∈ 1+j-r:j+1
          d[i,j+1] = (1-α)*d[i,j] + α*d[i,j+1]
      end
    end
  end
  d[:,p+1]
end
@inline fillΦ(x, t, p, k = find_k(t, x)) = fillΦ2!(eye(p+1), x, t, p, k)




###Faster version? May as well set d...
function fillΦ!(d, x, t, p, k = find_k(t, x))
  fill!(d, 0.0)
  for i ∈ 1:p+1
      d[i,i] = 1.0
  end
  for r ∈ 1:p
    for j ∈ p:-1:r
      α = (x - t[j+k-p-1]) / (t[j+k-r] - t[j+k-p-1])
      for i ∈ 1+j-r:j+1 ###perhaps copy this for j = p and insert between r and j loops; will let you write results directly into target vector (design matrix)
          d[i,j+1] = (1-α)*d[i,j] + α*d[i,j+1]
      end
    end
  end
  view(d, :,p+1)
end

#d is a p x p buffer; last column truncated to write directly into Φt


function sorted_fillΦ!(Φt::Matrix{T}, d::Matrix{T}, x::Vector{T}, t::CardinalKnots{p}) where {T, p}
    k = p+2
    t_next = t[k]
    for i ∈ eachindex(x)
        xᵢ = x[i]
        while xᵢ > t_next
            k += 1
            t_next += t.v
        end
        fillΦ_core!(Φt, d, x, t, i, k, xᵢ)
    end
end


function fillΦ!(Φt::Matrix{T}, d::Matrix{T}, x::Vector{T}, t::Knots{p}) where {T, p}
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        fillΦ_core!(Φt, d, t, i, k, xᵢ)
    end
end
function fillΦ!(Φt::Matrix{T}, d::Matrix{T}, x::Vector{T}, t::CardinalKnots{p}) where {T, p}
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        k > 2p && p + k < t.n ? fillΦ_coreKGP!(Φt, d, t, i, k, xᵢ) : fillΦ_core!(Φt, d, t, i, k, xᵢ)
    end
end

function fillΦ_coreDEFUNCT!(Φt::Matrix{T}, d::Matrix{T}, t::CardinalKnots{p}, i::Int, k::Int, xᵢ::T) where {T, p}
    denom = t.v * p
    α = (xᵢ - t[k-1]) / denom
    Φt[k-2,i] = 1-α
    Φt[k-1,i] = α
    for j ∈ 2:p
        α = (xᵢ - t[k-j]) / denom
        d[p+1-j,j-1] = 1-α
        d[p+2-j,j-1] = α
    end
    denom -= t.v
    for r ∈ 2:p
        α = (xᵢ - t[k-1]) / denom
        omα = 1-α
        Φt[k-r-1,i] = omα*d[1+p-r]
        for l ∈ 2+p-r:p
            Φt[l+k-p-2,i] = α*Φt[l+k-p-2,i] + omα*d[l]
        end
        Φt[k-1,i] *= α

        for j ∈ 2:1+p-r
            α = (xᵢ - t[k-j]) / denom
            omα = 1-α
            d[p+2-j-r,j-1] = omα*d[p+2-j-r,j]
            for l ∈ p+3-j-r:p+1-j
                d[l,j-1] = α*d[l,j-1] + omα*d[l,j]
            end
            d[p+2-j,j-1] *= α
        end
        println(d)
        denom -= t.v
    end
end


function fillΦ_coreKGP!(out::Matrix{T}, d::Matrix{T}, t::CardinalKnots{p}, l::Int, k::Int, x::T) where {T, p}
    @inbounds begin
        denom = t.v * p
        α = (x - t[k-1]) / denom
        out[k-2,l] = 1-α
        out[k-1,l] = α
        for j ∈ 2:p
            α = (x - t[k-j]) / denom
            d[1+p-j,j-1] = (1-α)
            d[p-j+2,j-1] = α
        end
        denom -= t.v
        for r ∈ 2:p
            α = (x - t[k-1]) / denom
            out[k-r-1,l] = (1-α)*d[1+p-r]
            for i ∈ 2+p-r:p
                out[i+k-p-2,l] = (1-α)*d[i] + α*out[i+k-p-2,l]
            end
            out[k-1,l] *= α
            for j ∈ 2:1+p-r
                α = (x - t[k-j]) / denom
                d[2+p-j-r,j-1] = (1-α)*d[2+p-j-r,j] #+ α*d[2+p-j-r,j]
                for i ∈ 3+p-j-r:p-j+1
                    d[i,j-1] = α*d[i,j-1] + (1-α)*d[i,j]
                end
                d[p+2-j,j-1] *= α
            end
            denom -= t.v
        end
    end
end


function fillΦ_core!(out::Matrix{T}, d::Matrix{T}, t::Knots{p}, l::Int, k::Int, x::T) where {T, p}
   @inbounds begin
       α = (x - t[k-1]) / (t[p+k-1] - t[k-1])
       out[k-2,l] = 1-α
       out[k-1,l] = α
       for j ∈ 2:p
           α = (x - t[k-j]) / (t[p+k-j] - t[k-j])
           d[1+p-j,j-1] = (1-α)
           d[p-j+2,j-1] = α
       end
       for r ∈ 2:p
           α = (x - t[k-1]) / (t[p+k-r] - t[k-1])
           out[k-r-1,l] = (1-α)*d[1+p-r]
           for i ∈ 2+p-r:p
               out[i+k-p-2,l] = (1-α)*d[i] + α*out[i+k-p-2,l]
           end
           out[k-1,l] *= α
           for j ∈ 2:1+p-r
               α = (x - t[k-j]) / (t[1+p+k-r-j] - t[k-j])
               d[2+p-j-r,j-1] = (1-α)*d[2+p-j-r,j] #+ α*d[2+p-j-r,j]
               for i ∈ 3+p-j-r:p-j+1
                   d[i,j-1] = α*d[i,j-1] + (1-α)*d[i,j]
               end
               d[p+2-j,j-1] *= α
           end
       end
   end
end

function fillΦ3!(out, d, x, t, p, k = find_k(t, x))
    α = (x - t[k-1]) / (t[p+k-1] - t[k-1])
    out[p] = (1-α)
    out[p+1] = α
    for j ∈ 2:p
        α = (x - t[k-j]) / (t[p+k-j] - t[k-j])
        d[1+p-j,j-1] = (1-α)
        d[p-j+2,j-1] = α
    end
    for r ∈ 2:p
        α = (x - t[k-1]) / (t[p+k-r] - t[k-1])
        out[1+p-r] = (1-α)*d[1+p-r,1]
        for i ∈ 2+p-r:p
            out[i] = (1-α)*d[i,1] + α*out[i]
        end
        out[p+1] *= α
        for j ∈ 2:1+p-r
            α = (x - t[k-j]) / (t[1+p+k-r-j] - t[k-j])
            d[2+p-j-r,j-1] = (1-α)*d[2+p-j-r,j] #+ α*d[2+p-j-r,j]
            for i ∈ 3+p-j-r:p-j+1
                d[i,j-1] = α*d[i,j-1] + (1-α)*d[i,j]
            end
            d[p+2-j,j-1] *= α
        end
    end
    out
end

#Base.LinAlg.LAPACK.gels!('N', X, copy(y)) |> x -> x[2] #first K vals of Y

function fillBand!(B, k, x, t, c, p)
  d = [c[j+k-p] for j ∈ -1:p-1]
  for r ∈ 1:p
    for j ∈ p:-1:r
      α = (x - t[j+k-p-1]) / (t[j+k-r] - t[j+k-p-1])
      d[j+1] = (1-α)*d[j] + α*d[j+1]
    end
  end
  d[p+1]
end
