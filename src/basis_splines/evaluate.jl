

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

function deBoor!(d::AbstractVector, x, t::Knots{q}, c, vp::Val{p}, k = find_k(t, x)) where {p, q}
    for j ∈ 1:p+1
        d[p+2-j] = c[j+k-q-2]
    end
    deBoorCore!(d, x, t, c, vp, k)
    d[1]
end

deBoorCore(d::AbstractArray, x, t::Knots{p}, c, k = find_k(t, x)) where p = deBoorCore(d, x, t, c, Val{p}(), k)

#would @inline improve performance?
@generated function deBoorCore!(x, t::DiscreteKnots, c, ::Val{p}, k = find_k(t, x)) where p
    quote
        @nexprs $p r -> begin
            @nexprs $p+1-r j -> begin
                α = (x - t[k-j]) / (t[p+1+k-r-j] - t[k-j])
                d_{j} = α*d_{j} + (1-α)*d_{j+1}
            end
        end
        d_1
    end
end
@generated function deBoorCore!(x, t::CardinalKnots, c, ::Val{p}, k = find_k(t, x)) where p
    quote
        denom = t.v * p
        @nexprs $p r -> begin
            @nexprs 1+$p-r j -> begin
                α = (x - t[k-j]) / denom
                d_{j} = α*d_{j} + (1-α)*d_{j+1}
            end
            denom -= t.v
        end
        d_1
    end
end


@generated function deBoor!(out::AbstractVector, x::AbstractVector, t::DiscreteKnots{q}, x_member, c, Vp::Val{p}) where {q, p}
    m = p + 1
    quote
        for i ∈ eachindex(x)
            xᵢ = x[i]
            k = x_member[i]
            @nexprs $m j -> begin
                d_{p+2-j} = c[j+k-q-2]
            end
            @nexprs $p r -> begin
                @nexprs $p+1-r j -> begin
                    α = (x - t[k-j]) / (t[p+1+k-r-j] - t[k-j])
                    d_{j} = α*d_{j} + (1-α)*d_{j+1}
                end
            end
            out[i] = d_1
        end
        out
    end
end
@generated function deBoor!(out::AbstractVector, x::AbstractVector, t::CardinalKnots{q}, x_member, c, Vp::Val{p}) where {q, p}
    m = p + 1
    quote
        for i ∈ eachindex(x)
            xᵢ = x[i]
            k = x_member[i]
            @nexprs $m j -> begin
                d_{p+2-j} = c[j+k-q-2]
            end
            denom = t.v * p
            @nexprs $p r -> begin
                @nexprs 1+$p-r j -> begin
                    α = (x - t[k-j]) / denom
                    d_{j} = α*d_{j} + (1-α)*d_{j+1}
                end
                denom -= t.v
            end
            out[i] = d_1
        end
        out
    end
end

function sorteddeBoor!(out::AbstractVector, d::AbstractArray, x::AbstractVector, t::Knots{q}, c, Vp::Val{p}) where {q, p}
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
    Threads.@threads for i ∈ eachindex(x)
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
function fillΦ_sorted!(Φᵗ, x, t::CardinalKnots, x_member)
    for i ∈ eachindex(x)
        xᵢ = x[i]
        k = x_member[i]
        (k > 2p) && (p + k < t.n) ? fillΦ_flatKGP!(Φᵗ, t, i, xᵢ, k) : fillΦ_flat!(Φᵗ, t, i, xᵢ, k)
    end
end
function fillΦ_sorted!(Φᵗ, x, t::DiscreteKnots, x_member)
    for i ∈ eachindex(x)
        fillΦ_flat!(Φᵗ, t, i, x[i], x_member[i])
    end
end
@generated function fillΦ_flat!(out::AbstractMatrix, t::Knots{p}, l::Int, x, k::Int = find_k(t, x)) where p
    pm1 = p - 1
    m = p + 1
    dp = Symbol("d_", p)
    dm = Symbol("d_", m)
    quote
        @inbounds begin
            #begin
            α = (x - t[k-1]) / (t[$pm1+k] - t[k-1])
            $dp = 1-α
            $dm = α
            @nexprs $pm1 j -> begin
                α = (x - t[k-j+1]) / (t[$p+k-j+1] - t[k-j+1])
                d_{$pm1*(j+1)+2} = (1-α)
                d_{$pm1*(j+1)+3} = α
            end
            @nexprs $pm1 r -> begin
                α = (x - t[k-1]) / (t[$pm1+k-r] - t[k-1])
                d_{$p-r} = (1-α)*d_{$m+$p-r}
                @nexprs r i -> begin
                    d_{i+$p-r} = (1-α)*d_{$m+i+$p-r} + α*d_{i+$p-r}
                end
                $dm *= α
                @nexprs $pm1-r j -> begin
                    α = (x - t[k-j-1]) / (t[pm1+k-r-j] - t[k-j-1])
                    d_{$pm1*j-r+$m} = (1-α)*d_{$pm1*(j+1)+1-r+$m}
                    @nexprs r i -> begin
                        d_{i-r+j*$pm1+$m} = α*d_{i-r+j*$pm1+$m} + (1-α)*d_{i+1-r+(j+1)*$pm1+$m}
                    end
                    d_{$pm1*j+1+$m} *= α
                end
            end
            @nexprs $m i -> begin
                out[i,l] = d_{i}
            end
        end
    end
end

@generated function texpr(A, ::Val{m}, ::Val{n}) where {m, n}
    quote
        out = zero(eltype(A))
        @nexprs $m i -> begin
            @nexprs $n-i+1 j -> begin
                out += A[j+i-1, i]
            end
        end
        out
    end
end

@generated function fillΦ_flatKGP!(out::AbstractMatrix, t::CardinalKnots{p}, l::Int, x, k::Int = find_k(t, x)) where p
    pm1 = p - 1
    m = p + 1
    dp = Symbol("d_", p)
    dm = Symbol("d_", m)
    quote
        @inbounds begin
        #begin
            denom = t.v * $p
            α = (x - t[k-1]) / denom
            $dp = 1-α
            $dm = α
            @nexprs $pm1 j -> begin
                α = (x - t[k-j-1]) / denom
                d_{$pm1*j+$m} = (1-α)
                d_{$pm1*j+1+$m} = α
            end
            denom -= t.v
            @nexprs $pm1 r -> begin
                α = (x - t[k-1]) / denom
                d_{$p-r} = (1-α)*d_{$p-r+$m}
                @nexprs r i -> begin
                    d_{i+$p-r} = (1-α)*d_{i+$p-r+$m} + α*d_{i+$p-r}
                end
                $dm *= α
                @nexprs $pm1-r j -> begin
                    α = (x - t[k-j-1]) / denom
                    d_{$pm1*j-r+$m} = (1-α)*d_{$pm1*(j+1)+1-r+$m}
                    @nexprs r i -> begin
                        d_{i-r+j*$pm1+$m} = α*d_{i-r+j*$pm1+$m} + (1-α)*d_{i+1-r+(j+1)*$pm1+$m}
                    end
                    d_{$pm1*j+1+$m} *= α
                end
                denom -= t.v
            end
            @nexprs $m i -> begin
                out[i,l] = d_{i}
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


@generated function fill_band!(band::AbstractMatrix{T}, Φ::AbstractMatrix{T},  min_k, max_k, ::Val{p}) where {T,p}
    m = p + 1
    quote
        n = size(band,2)
        @nexprs $p i -> begin
            minᵢ = min_k[i]
            for j ∈ $m-i+1:$p
                jind = i+j-$m
                t = zero(T)
                for k ∈ minᵢ:max_k[jind]
                    t += Φ[k,i]*Φ[k,jind]
                end
                band[j,i] = t
            end
            t = zero(T)
            for k ∈ minᵢ:max_k[i]
                t += abs2(Φ[k,i])
            end
            band[$m,i] = t
        end
        for i ∈ $m:n
            minᵢ = min_k[i]
            @nexprs $p j -> begin
                jind = i+j-$m
                t = zero(T)
                for k ∈ minᵢ:max_k[jind]
                    t += Φ[k,i]*Φ[k,jind]
                end
                band[j,i] = t
            end
            t = zero(T)
            for k ∈ minᵢ:max_k[i]
                t += abs2(Φ[k,i])
            end
            band[$m,i] = t
        end
    end
end







