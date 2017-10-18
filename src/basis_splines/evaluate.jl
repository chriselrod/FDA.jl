

macro threaded(expr)
   Threads.nthreads() == 1 ? expr : :(Threads.@threads($:(expr)))
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
    for i ∈ length(s.coefficients):d #i is derivative being added; final length of coefs is d+1
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

@inline fillΦ!(Φt::AbstractMatrix, d::AbstractArray, x::AbstractVector, t::Knots{p}) where p = fillΦ!(Φt, d, x, t, p)
function fillΦ_unthreaded!(Φt::AbstractMatrix, d::AbstractArray, x::AbstractVector, t::DiscreteKnots, p::Int)
    for i ∈ eachindex(x)
        xᵢ = x[i]
        fillΦ_core!(Φt, d, t, i, xᵢ, p, find_k(t, xᵢ))
    end
end
function fillΦ_threaded2!(Φt::AbstractMatrix, d::AbstractArray, x::AbstractVector, t::DiscreteKnots, p::Int)
    buffers = [d]
    for i ∈ 2:Threads.nthreads()
        push!(buffers, similar(d))
    end
 #   res = Vector{Tuple{Tuple{Int,Int},Vector{T}}}(length(x))
    @threads for i ∈ eachindex(x)
        xᵢ = x[i]
        fillΦ_core!(Φt, buffers[Threads.threadid()], t, i, xᵢ, p, find_k(t, xᵢ))
    #    fillΦ_core_threaded!(Φt, buffers[Threads.threadid()], t, i, xᵢ, p, find_k(t, xᵢ))
    #    res[i] = fillΦ_core_threaded!(buffers[Threads.threadid()], t, i, xᵢ, p, find_k(t, xᵢ))
    end
    #@inbounds for i ∈ eachindex(res)
    #    Φt[ res[i][1][1]:res[i][1][2], i] .= res[i][2]
    #end
end
@inline fillΦ!(args...) = Threads.nthreads() == 1 ? fillΦ_unthreaded!(args...) : fillΦ_unthreaded!(args...)

function fillΦ_unthreaded!(Φt::AbstractMatrix, d::AbstractArray, x::AbstractVector, t::CardinalKnots, p::Int)
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        k > 2p && p + k < t.n ? fillΦ_coreKGP!(Φt, d, t, i, xᵢ, p, k) : fillΦ_core!(Φt, d, t, i, xᵢ, p, k)
    end
end
function fillΦ_threaded!(Φt::AbstractMatrix, d::AbstractArray, x::AbstractVector, t::CardinalKnots, p::Int)
    buffers = [d]
    range::Base.OneTo{Int64}
    for i ∈ 2:Threads.nthreads()
        push!(buffers, similar(d))
    end
    Threads.@threads for i ∈ eachindex(x)
        xᵢ = x[i]
        k = find_k(t, xᵢ)
        k > 2p && p + k < t.n ? fillΦ_coreKGP!(Φt, buffers[Threads.threadid()], t, i, xᵢ, p, k) : fillΦ_core!(Φt, buffers[Threads.threadid()], t, i, xᵢ, p, k)
    end

end
function fillΦ_threaded!(Φt::AbstractMatrix{T}, d::AbstractArray, x::AbstractVector{T}, t::DiscreteKnots, p::Int) where T
    buffers = [d]
    for i ∈ 2:Threads.nthreads()
        push!(buffers, similar(d))
    end


    
    function threadsfor_fun(onethread=false)
        r = eachindex(x) # Load into local variable
        lenr = length(r)
        # divide loop iterations among threads
        if onethread
            tid = 1
            len, rem = lenr, 0
        else
            tid = Threads.threadid()
            len, rem = divrem(lenr, Threads.nthreads())
        end
        # not enough iterations for all the threads?
        if len == 0
            if tid > rem
                return
            end
            len, rem = 1, 0
        end
        # compute this thread's iterations
        f = 1 + ((tid-1) * len)
        l = f + len - 1
        # distribute remaining iterations evenly
        if rem > 0
            if tid <= rem
                f = f + (tid-1)
                l = l + tid
            else
                f = f + rem
                l = l + rem
            end
        end
        # run this thread's iterations
        for i = f:l
            local lidx = Base.unsafe_getindex(r,i)
            
            xᵢ = x[lidx]
            fillΦ_core_threaded!(Φt, buffers[Threads.threadid()], t, lidx, xᵢ, p, find_k(t, xᵢ))
        end
    end
    # Hack to make nested threaded loops kinda work

#    in_threaded_loop[] = true
    # the ccall is not expected to throw
    ccall(:jl_threading_run, Ref{Void}, (Any,), threadsfor_fun)
#    in_threaded_loop[] = false
    nothing


end


###Need to make decision about whether to deparametrize, or unroll the loops.
function fillΦ_coreKGP!(out::AbstractMatrix, d::AbstractArray, t::CardinalKnots, l::Int, x, p::Int, k::Int = find_k(t, x))
    @inbounds begin
    #begin
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
                d[(p-1)*(j-1)+1-r] = (1-α)*d[(p-1)j+2-r]
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
function fillΦ_core!(out::AbstractMatrix, d::AbstractArray, t::Knots, l::Int, x, p::Int, k::Int = find_k(t, x))
    @inbounds begin
    #begin
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
               d[(p-1)*(j-1)+1-r] = (1-α)*d[(p-1)j+2-r]
               for i ∈ 3+p-j-r:p-j+1
                   d[i+(j-2)p] = α*d[i+(j-2)p] + (1-α)*d[i+(j-1)p]
               end
               d[(p-1)*(j-1)+1] *= α
           end
       end
   end
end


function fillΦ_core_threaded!(out::AbstractMatrix, d::AbstractArray, t::Knots, l::Int, x, p::Int, k::Int = find_k(t, x))
    @inbounds begin
    #begin
       α = (x - t[k-1]) / (t[p+k-1] - t[k-1])
       d[p] = 1-α
       d[p + 1] = α
       for j ∈ 2:p
           α = (x - t[k-j]) / (t[p+k-j] - t[k-j])
           d[(p-1)j+2] = (1-α)
           d[(p-1)j+3] = α
       end
       for r ∈ 2:p
           α = (x - t[k-1]) / (t[p+k-r] - t[k-1])
           d[p + 1 - r] = (1-α)*d[2+2p-r]
           for i ∈ 2+p-r:p
               d[i] = α*d[i] + (1-α)*d[i+p+1]
           end
           d[p + 1] *= α
           for j ∈ 2:1+p-r
               α = (x - t[k-j]) / (t[1+p+k-r-j] - t[k-j])
               d[(p-1)j-r+3] = (1-α)*d[(p-1)*(j+1)+4-r]
               for i ∈ 3+p-j-r:p-j+1
                   d[i+(j-1)p+1] = α*d[i+(j-1)p+1] + (1-α)*d[i+j*p+1]
               end
               d[(p-1)j+3] *= α
           end
       end
       for i ∈ 1:p+1
            out[i+k-p-2,l] = d[i]
       end
   end
end
function fillΦ_core_threaded!(d::AbstractArray, t::Knots, l::Int, x, p::Int, k::Int = find_k(t, x))
    @inbounds begin
    #begin
       α = (x - t[k-1]) / (t[p+k-1] - t[k-1])
       d[p] = 1-α
       d[p + 1] = α
       for j ∈ 2:p
           α = (x - t[k-j]) / (t[p+k-j] - t[k-j])
           d[(p-1)j+2] = (1-α)
           d[(p-1)j+3] = α
       end
       for r ∈ 2:p
           α = (x - t[k-1]) / (t[p+k-r] - t[k-1])
           d[p + 1 - r] = (1-α)*d[2+2p-r]
           for i ∈ 2+p-r:p
               d[i] = α*d[i] + (1-α)*d[i+p+1]
           end
           d[p + 1] *= α
           for j ∈ 2:1+p-r
               α = (x - t[k-j]) / (t[1+p+k-r-j] - t[k-j])
               d[(p-1)j-r+3] = (1-α)*d[(p-1)*(j+1)+4-r]
               for i ∈ 3+p-j-r:p-j+1
                   d[i+(j-1)p+1] = α*d[i+(j-1)p+1] + (1-α)*d[i+j*p+1]
               end
               d[(p-1)j+3] *= α
           end
       end
       out = d[1:p+1]
   end
   (k-p-1, k-1), out
end
# + 2 + p - k










