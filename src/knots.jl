abstract type Knots{p,T} end
"P is the degree of the polynomial fit; order is p+1."
struct CardinalKnots{p,T} <: Knots{p,T}
    min::T
    max::T
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
    unique::Int
end
struct intervals{K,T}
    knots::K
    done::T
end
intervals(knots::CardinalKnots) = intervals(knots, knots.done - knots.v)
intervals(knots::DiscreteKnots{p}) where p = intervals(knots, knots.n + p - 1)

function CardinalKnots(x::Vector{T}, k::Int, ::Val{p}) where {p,T}
    miv, mav = extrema(x)
    nm1 = k - p
    δ = (mav - miv)/ nm1
    δpoo5 = δ * 0.005
    CardinalKnots{p,T}( miv - δpoo5, mav + δpoo5, δ*(100nm1+1)/100nm1, nm1+1, mav + 2δpoo5, p+1, p+nm1+1)
end
function CardinalKnots(minval::T, maxval::T, n::Int, ::Val{p}) where {p,T}
    S = promote_type(T, Float64)
    v = convert(S, (maxval-minval)/(n-1))
    CardinalKnots{p,S}(convert(S, minval), convert(S, maxval), v, n, 0.5v + maxval, p+1, p+n)
end
function DiscreteKnots(::Val{p}, v::Vector{T}, n::Int = length(v), unique::Int = count_unique_sorted(n)) where {p,T}
    DiscreteKnots{p,T}(v[1], v[n], v, n, unique)
end
function DiscreteKnots(v::Vector{T}, ::Val{p}) where {p,T}
    issorted(v) || sort!(v)
    n = length(v)
    unique = count_unique_sorted(n)
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
    DiscreteKnots{p,T}(minval, maxval, v, n, unique)
end
DiscreteKnots(v::StepRangeLen, ::Val{p}) where p = CardinalKnots{p,Float64}(v[1], v[end], convert(Float64, v.step), length(v))

function count_unique_sorted(v)
    unique = 0
    last = v[1]
    for i ∈ 2:length(v)
        next = v[i]
        if !(last ≈ next)
            unique += 1
        end
        last = next
    end
    unique
end

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

@inline find_k(t::DiscreteKnots{p}, x::Real) where p = x >= t.max ? t.n + p : searchsortedfirst(t.v, x, lt = <=)
@inline find_k(t::CardinalKnots{p}, x::Real) where p = convert(Int, cld(x - t.min, t.v)) + p + 1
@inline function find_k(t::DiscreteKnots{p}, x) where p
    xr = real(x)
    xr >= t.max ? t.n + p : searchsortedfirst(t.v, xr, lt = <=)
end
@inline find_k(t::CardinalKnots{p}, x) where p = convert(Int, cld(real(x) - t.min, t.v)) + p + 1

unique(t::CardinalKnots) = t.n
unique(t::DiscreteKnots) = t.unique

Base.size(x::Knots) = (x.n, )
Base.length(x::Knots) = x.n
Base.eltype(::Knots{p,T}) where {p,T} = T

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
    (lb, ub, xb - lb), next
end
Base.done(x::intervals{DiscreteKnots{p,T}}, state) where {p, T} = state > x.done
Base.start(x::intervals{K}) where K <: CardinalKnots = x.min
function Base.next(::intervals{K}, state) where K <: CardinalKnots
    next = state + x.v
    (state, next, x.v), next
end
Base.done(::intervals{CardinalKnots{p,T}}, state) where {p, T} = state > x.done
