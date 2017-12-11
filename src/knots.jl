abstract type Knots{p,T} end
"P is the degree of the polynomial fit; order is p+1."
struct CardinalKnots{p,T} <: Knots{p,T}
    min::T
    max::T
    min_::T
    max_::T
    v::T
    n::Int
    done::T
    minind::Int
    maxind::Int
end
struct DiscreteKnots{p,T} <: Knots{p,T}
    min::T
    max::T
    v::Vector{T}
    n::Int
    Δ::Vector{T}
end
struct intervals{K,T}
    knots::K
    done::T
end
intervals(knots::CardinalKnots) = intervals(knots, knots.done - knots.v)
intervals(knots::DiscreteKnots{p}) where p = intervals(knots, knots.n + p - 1)

function CardinalKnots(x::Vector{T}, k::Int, ::Val{p}, ::Val{true}) where {p,T}
    miv = x[1]; mav = x[2];
    nm1 = k - p
    δ = (mav - miv)/ nm1
    δpoo5 = δ * 0.005
    minv_ = miv - δpoo5
    maxv_ = mav + δpoo5
    CardinalKnots{p,T}(minv_, maxv_, prevfloat(minv_), nextfloat(maxv_), δ*(100nm1+1)/100nm1, nm1+1, mav + 2δpoo5, p+1, p+nm1+1)
end
function CardinalKnots(x::Vector{T}, k::Int, ::Val{p}) where {p,T}
    miv, mav = extrema(x)
    nm1 = k - p
    δ = (mav - miv)/ nm1
    δpoo5 = δ * 0.005
    minv_ = miv - δpoo5
    maxv_ = mav + δpoo5
    CardinalKnots{p,T}(minv_, maxv_, prevfloat(minv_), nextfloat(maxv_), δ*(100nm1+1)/100nm1, nm1+1, mav + 2δpoo5, p+1, p+nm1+1)
end
function CardinalKnots(minval::T, maxval::T, n::Int, ::Val{p}) where {p,T}
    S = promote_type(T, Float64)
    v = convert(S, (maxval-minval)/(n-1))
    minv_ = convert(S, minval)
    maxv_ = convert(S, maxval)
    CardinalKnots{p,S}(minv_, maxv_, prevfloat(minv_), nextfloat(maxv_),  v, n, 0.5v + maxval, p+1, p+n)
end
function DiscreteKnots(::Val{p}, v::Vector{T}, n::Int = length(v), Δ::Vector{T} = T[]) where {p,T}
    DiscreteKnots{p,T}(v[1], v[n], v, n, Δ)
end
function DiscreteKnots(v::Vector{T}, ::Val{p}) where {p,T}
    issorted(v) || sort!(v)
    n = length(v)
    Δ = T[]
    minval = v[1]
    maxval = v[n]
    if v[2] != minval
        n = length(v)
        sizehint!(v, n + 2p)
        for i ∈ 1:p
            unshift!(v, minval)
            push!(v, maxval)
        end
    else
        n = length(v) - 2p
    end
    DiscreteKnots{p,T}(minval, maxval, v, n, Δ)
end
DiscreteKnots(v::StepRangeLen, ::Val{p}) where p = CardinalKnots{p,Float64}(v[1], v[end], convert(Float64, v.step), length(v))

function count_intervals!(t::DiscreteKnots{p,T}) where {p,T}
    length(t.Δ) == 0 || return length(t.Δ)
    last = t.v[p+1]
    for i ∈ p+2:length(t.v)
        next = t.v[i]
        δ = next  - last
        δ ≈ 0 || push!(t.Δ, δ)
        last = next
    end
    length(t.Δ)
end
count_intervals!(t::CardinalKnots) = t.n - 1

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

function k_structure!(x::AbstractVector, y::AbstractVector, t::Knots, p, β::AbstractVector)
    n = length(x)
    coef_count = length(β)
    x_member = sortperm(x)
    x .= @view(x[x_member])
    y .= @view(y[x_member])
    min_k = reinterpret(Int, β)
    max_k = Vector{Int}(coef_count)
    last_k = p+2
    min_k[1:p+1] .= 1
    for i ∈ 1:n
#    @inbounds for i ∈ 1:n
        new_k = find_k(t, x[i], last_k)
        new_k == last_k || begin
            for j ∈ last_k:new_k-1
                min_k[j] = i
                max_k[j-p-1] = i - 1
            end
        end
        x_member[i] = new_k
        last_k = new_k
    end
    max_k[coef_count-p:coef_count] .= n
    x_member, min_k, max_k
end
function k_structure!(x::AbstractVector, y::AbstractVector, t::Knots, p, coef_count::Int)
    n = length(x)
    x_member = sortperm(x)
    x .= @view(x[x_member])
    y .= @view(y[x_member])
    min_k = Vector{Int}(coef_count)
    max_k = Vector{Int}(coef_count)
    last_k = p+2
    min_k[1:p+1] .= 1
    for i ∈ 1:n
#    @inbounds for i ∈ 1:n
        new_k = find_k(t, x[i], last_k)
        new_k == last_k || begin
            for j ∈ last_k:new_k-1
                min_k[j] = i
                max_k[j-p-1] = i - 1
            end
        end
        x_member[i] = new_k
        last_k = new_k
    end
    max_k[coef_count-p:coef_count] .= n
    x_member, min_k, max_k
end

@inline find_k(t::CardinalKnots, x, start) = find_k(t, x)
@inline find_k(t::DiscreteKnots{p}, x::Real, start) where p = x >= t.max ? t.n + p : start - 1 + searchsortedfirst( @view(t.v[start:end]), x, Base.Order.Lt(<=) )
@inline find_k(t::DiscreteKnots{p}, x::Real) where p = x >= t.max ? t.n + p : searchsortedfirst(t.v, x, Base.Order.Lt(<=))
@inline function find_k(t::DiscreteKnots{p}, x) where p
    xr = real(x)
    xr >= t.max ? t.n + p : searchsortedfirst(t.v, xr, Base.Order.Lt(<=))
end
@inline find_k(t::CardinalKnots{p}, x) where p = convert(Int, fld(real(x) - t.min_, t.v)) + p + 2
@inline find_k(t::CardinalKnots{p}, x::Real) where p = convert(Int, fld(x - t.min_, t.v)) + p + 2

Base.unique(t::CardinalKnots) = t.n
Base.unique(t::DiscreteKnots) = length(t.Δ)+1

Base.size(x::Knots) = (x.n, )
Base.length(x::Knots) = x.n
Base.eltype(::Knots{p,T}) where {p,T} = T

struct CardinalKnotDiff{T} <: AbstractArray{T,1}
    v::T
    n::Int
end
Base.getindex(x::CardinalKnotDiff, i) = x.v
Base.diff(x::CardinalKnots{p,T}) where {p,T} = CardinalKnotDiff{T}(x.v, x.n-1)
Base.length(x::CardinalKnotDiff) = x.n
Base.size(x::CardinalKnotDiff) = (x.n, )
IndexStyle(::CardinalKnotDiff) = IndexLinear()
Base.diff(x::DiscreteKnots) = x.Δ

@generated Base.start(x::DiscreteKnots{p}) where p = p + 1
Base.start(x::CardinalKnots) = x.min
Base.next(x::DiscreteKnots, state) = x.v[state], state + 1
Base.next(x::CardinalKnots, state) = state, state + x.v
Base.done(x::DiscreteKnots, state) = state > x.n + p
Base.done(x::CardinalKnots, state) = state > x.done

Base.size(x::intervals) = (x.knots.n - 1, )
Base.length(x::intervals) = x.knots.n - 1
Base.eltype(::intervals{K}) where {p, T, K <: Knots{p, T}} = T

Base.range(t::DiscreteKnots, max::Int, min::Int) = t.v[max] - t.v[min]
Base.range(t::CardinalKnots, max_::Int, min_::Int) = t.v * (min(max_,t.maxind) - max(min_,t.minind))

@generated Base.start(::intervals{K}) where {p, K <: DiscreteKnots{p}} = p + 1
function Base.next(x::intervals{DiscreteKnots{p,T}}, state) where {p, T}
    next = state + 1
    lb = x[state]
    ub = x[next]
    (lb, ub, ub - lb), next
end
Base.done(x::intervals{DiscreteKnots{p,T}}, state) where {p, T} = state > x.done
Base.start(x::intervals{K}) where K <: CardinalKnots = x.min
function Base.next(x::intervals{K}, state) where K <: CardinalKnots
    next = state + x.v
    (state, next, x.v), next
end
Base.done(::intervals{CardinalKnots{p,T}}, state) where {p, T} = state > x.done
