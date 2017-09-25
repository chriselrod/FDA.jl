abstract type Knots{p} end

#Base.@pure Base.Val(T) = Val{T}()


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
function CardinalKnots(x::Vector{T}, n::Int, ::Val{p}) where {p,T}
    miv = minimum(x)
    mav = maximum(x)
    nm1 = n - 1
    δ = (mav - miv)/ nm1
    δpoo5 = δ * 0.005
    CardinalKnots{p,S}( miv - δpoo5, mav + δpoo5, δ*(100nm1+1)/100nm1, n)
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


@inline Base.getindex(t::CardinalKnots{p}, i) where p = t.min + (i - p - 1) * t.v
@inline Base.getindex(t::DiscreteKnots, i) = t.v[i]




#Every time a derivative is called, push on
struct BSpline{K, p, T}
  knots::K
  coefficients::Vector{Vector{T}}
  ssr::T
  buffer::Vector{T}
end


"""
k is the number of basis.
"""
function BSpline(x::Vector, y::Vector, v::Vector, vp::Val = Val{3}())
    knots = DiscreteKnots(v, vp)
    k = length(knots.v) - p - 1
    BSpline(x, y, k, knots)
end

function BSpline(x::Vector{T}, y::Vector, k::Int = div(length(x),5), ::Val{p} = Val{3}()) where {T, p}
    BSpline(x, y, k, CardinalKnots(x, k, Val{p}()))
end

function BSpline(x::Vector{T}, y::Vector, k::Int = div(length(x),5), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    n = length(x)
    Φt = zeros(k, n)
    vector_buffer = Vector{T}(p+1)
    matrix_buffer = Matrix{T}(p,p)
    fillΦ!(Φt, matrix_buffer, x, knots)
    f, b, ssr = Base.LinAlg.LAPACK.gels!('T', Φt, y)
    BSpline(knots, [b], ssr, vector_buffer)
end


function eval_spline(s::BSpline{K, p}, x::Vector) where {K, p}
    add_derivative_coefs!(s, vd)
    sorteddeBoor!(s.buffer, x, s.knots, s.coefficients[1], Val{p}())
end
function eval_spline(s::BSpline{K, p}, x::Vector, ::Val{d}) where {K, p, d}
    add_derivative_coefs!(s, vd)
    sorteddeBoor!(s.buffer, x, s.knots, s.coefficients[d+1], value_deriv(Val{p}(), Val{d}()))
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

"""
p is the penalty degree, and k the number of coefficients in your model.
"""
function penalty_matrix!(out::Matrix{T}, buffer::Vector{T}, k::Int, p::Int) where T
    fill!(out, zero(T))
    buffer[1] = 1
    next = last =  one(T)
    for i ∈ 1:p
        for j ∈ 1:i-1
            last, next = next, buffer[j+1]
            buffer[j+1] -= last
        end
        last, next = next, one(T)
        buffer[i+1] -= last
    end
    dk2 = div(k,2)
    for i ∈ 1:dk2, j ∈ max(1,i-p):i, l ∈ 1:min(j,p+1+j-i)
        out[j,i] += buffer[l]*buffer[l+i-j]
    end
    for i ∈ dk2+1:k, j ∈ max(1,i-p):i
        out[j,i] = out[k+1-i,k+1-j]
    end
    out
end
penalty_matrix(k, p) = penalty_matrix!(Matrix{Float64}(k, k), zeros(p+1), k, p)

@inline find_k(t::DiscreteKnots, x) = searchsortedfirst(t.v, x)
@inline find_k(t::CardinalKnots{p}, x) where p = convert(Int, cld(x - t.min, t.v)) + p


deBoor(x, t::Knots{p}, c, k = find_k(t, x)) where p = deBoor(x, t, c, Val{p}(), k)
deBoor(x::T, t, c, vp::Val{p}, k = find_k(t, x)) where {p,T} = deBoor!(Vector{T}(p+1), x, t, c, vp, k)

function deBoor!(d::Vector, x, t::Knots{q}, c, vp::Val{p}, k = find_k(t, x)) where {p, q}
    for j ∈ 1:p+1
        d[p+2-j] = c[j+k-q-2]
    end
    deBoorCore!(d, x, t, c, k, vp)
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


function sorteddeBoor!(out::Vector, d::Vector, x::Vector, t::Knots{q}, c, Vp::Val{p}) where {q, p}
    k = q+2
    t_next = t.v[k]
    for i ∈ eachindex(x)
        xᵢ = x[i]
        while xᵢ > t_next
            k += 1
            t_next = t.v[k]
        end
        for j ∈ 1:p+1
            d[p+2-j] = c[j+k-p-2]
        end
        deBoorCore!(d, xᵢ, t, c, k, Vp)
        out[i] = d[1]
    end
    out
end


function fillΦ2!(d, x, t, p, k = searchsortedfirst(t, x))
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
@inline fillΦ(x, t, p, k = searchsortedfirst(t, x)) = fillΦ2!(eye(p+1), x, t, p, k)

###Faster version? May as well set d...
function fillΦ!(d, x, t, p, k = searchsortedfirst(t, x))
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
  d[:,p+1]
end

#d is a p x p buffer; last column truncated to write directly into Φt
function fillΦ!(Φt::Matrix{T}, d::Matrix{T}, x::Vector{T}, t::DiscreteKnots{p}) where {T, p}
    k = p+2
    t_next = t[k]
    for i ∈ eachindex(x)
        xᵢ = x[i]
        while xᵢ > t_next
            k += 1
            t_next = t[k]
        end
        fill!(d, zero(T))
        ind = 2
        for j ∈ 1:p
            ind += p - 1
            d[ind] = one(T)
        end
        for r ∈ 1:p
            α = (xᵢ - t[k-1]) / (t[p+k-r] - t[k-1])
            omα = 1-α
            for l ∈ 1+p-r:p
                Φt[l+k-p-2,i] = α*Φt[l+k-p-2,i] + omα*d[l]
            end
            Φt[k-1,i] *= α

            for j ∈ 2:1+p-r
                α = (xᵢ - t[k-j]) / (t[p+1+k-r-j] - t[k-j])
                omα = 1-α
                for l ∈ p+2-j-r:p+2-j
                    d[l,j-1] = α*d[l,j-1] + omα*d[l,j]
                end
            end
        end
    end
end


function fillΦ!(Φt::Matrix{T}, d::Matrix{T}, x::Vector{T}, t::CardinalKnots{p}) where {T, p}
    k = p+2
    t_next = t[k]
    for i ∈ eachindex(x)
        xᵢ = x[i]
        while xᵢ > t_next
            k += 1
            t_next += t.v
        end
        fill!(d, zero(T))
        ind = 2
        for j ∈ 1:p
            ind += p - 1
            d[ind] = one(T)
        end
        denom = t.v * p
        for r ∈ 1:p
            α = (xᵢ - t[k-1]) / denom
            omα = 1-α
            for l ∈ 1+p-r:p
                Φt[l+k-p-2,i] = α*Φt[l+k-p-2,i] + omα*d[l]
            end
            Φt[k-1,i] *= α

            for j ∈ 2:1+p-r
                α = (xᵢ - t[k-j]) / denom
                omα = 1-α
                for l ∈ p+2-j-r:p+2-j
                    d[l,j-1] = α*d[l,j-1] + omα*d[l,j]
                end
            end
            denom -= t.v
        end
    end
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
