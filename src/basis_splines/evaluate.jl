
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



@inline fillΦ!(Φt::Matrix, d::Matrix, x::Vector, t::Knots{p}) where p = fillΦ!(Φt, d, x, t, Val{p}())
function fillΦ!(Φt::Matrix, d::Matrix, x::Vector, t::Knots, Vp::Val)
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        fillΦ_core!(Φt, d, t, i, k, xᵢ, Vp)
    end
end
function fillΦ!(Φt::Matrix{T}, d::Matrix{T}, x::Vector{T}, t::CardinalKnots{d}, p) where {T, d}
  if Threads.nthreads() == 1
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        k > 2p && p + k < t.n ? fillΦ_coreKGP!(Φt, d, t, i, k, xᵢ, Vp) : fillΦ_core!(Φt, d, t, i, k, xᵢ, Vp)
    end
  else
    Threads.@threads for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        k > 2p && p + k < t.n ? fillΦ_coreKGP!(Φt, d, t, i, k, xᵢ, Vp) : fillΦ_core!(Φt, d, t, i, k, xᵢ, Vp)
    end
  end
end

macro threaded(expr) = Threads.nthreads() == 1 ? expr : Threads.threads(expr)


###Need to make decision about whether to deparametrize, or unroll the loops.
function fillΦ_coreKGP!(out::Matrix{T}, d::AbstractArray{T}, t::CardinalKnots{d}, l::Int, k::Int, x::T, p) where {T, d}
    begin
        denom = t.v * p
        α = (x - t[k-1]) / denom
        out[k-2,l] = 1-α
        out[k-1,l] = α
        for j ∈ 2:p
            α = (x - t[k-j]) / denom
            d[(p-1)*(j-1)] = (1-α)
            d[(p-1)*(j-1)+1] = α
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
                d[(p-1)*(j-1)+1-r] = (1-α)*d[(j-1)p+1-r] #+ α*d[2+p-j-r,j]
                for i ∈ 3+p-j-r:p-j+1
                    d[i+(j-2)p] = α*d[i+(j-2)p] + (1-α)*d[i+(j-1)p]
                end
                d[(p-1)*(j-1)+1] *= α
            end
            denom -= t.v
        end
    end
end

"""d is a buffer. It must be linearly indexable, muttable, and of length at least p^2. Eg, a simple Vector{Float64}(p^2) or Matrix{Float64}(p,p). It was initially required to be the latter, but all indices were made linear to accomodate more flexible choices in buffer, and making it easier to reuse the same chunks of memory."""
function fillΦ_core!(out::Matrix{T}, d::AbstractArray{T}, t::Knots{d}, l::Int, k::Int, x::T, p::Int) where {T, d}
   @inbounds begin
       α = (x - t[k-1]) / (t[p+k-1] - t[k-1])
       out[k-2,l] = 1-α
       out[k-1,l] = α
       for j ∈ 2:p
           α = (x - t[k-j]) / (t[p+k-j] - t[k-j])
           d[(p-1)*(j-1)] = (1-α)
           d[(p-1)*(j-1)+1] = α
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
               d[(p-1)*(j-1)+1-r] = (1-α)*d[(j-1)p+1-r] #+ α*d[2+p-j-r,j]
               for i ∈ 3+p-j-r:p-j+1
                   d[i+(j-2)p] = α*d[i+(j-2)p] + (1-α)*d[i+(j-1)p]
               end
               d[(p-1)*(j-1)+1] *= α
           end
       end
   end
end
