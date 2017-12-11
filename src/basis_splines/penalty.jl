
### Okay, for penalty, we can forgo calculation of G altogether.
### J = β'Sβ, S = G'WG, therefore J = (Gβ)'WGβ. You have already implemented the method for calculating Gβ!!! Alternatively, should be straightforward to do with fillΦ! as well.
### Question is -- what is easiest? Actual calculation of Gβ would aid solving via linear algebra.
###Fix later for cardinal knots. Basically, replace first for loop.
@generated function BSplinePenalty(knots::Knots{p,T}, k::Int, ::Val{d}) where {d, p, T}
    m = p + 1
    ∂ = p - d
    ∂p1 = ∂ + 1
    ∂m1 = ∂ - 1
    quote
        knotsₗ = length(knots)
        q = count_intervals!(knots)
        n = muladd(q,$∂,1)
        xv = Vector{T}(n)
        lb = knots[1]
        ∂Φ = Vector{T}[T[] for i ∈ 1:k]
        Δknots = diff(knots)
        if isa(knots, DiscreteKnots)
            skip = 0
        end
        for i ∈ 1:knotsₗ-1
            ub = knots[i+$m]
            if isa(knots, DiscreteKnots)
                if lb == ub
                    skip += 1
                    continue
                end
                Δ = Δknots[i-skip]
            else
                Δ = Δknots[i]
            end
            @nexprs $∂m1 j -> begin
                ϕ = fillΦ_flat(knots, i, lb + Δ * (j-1) / $∂p1, Val{$d}(), i+$m)
                @nexprs $m l -> begin
                    push!(∂Φ[(i-1)*$∂+l], ϕ[l])
                end
            end
            ϕ = fillΦ_flat(knots, i, lb + Δ * $∂m1 / $∂p1, Val{$d}(), i+$m)
            @nexprs $p l -> begin
                push!(∂Φ[(i-1)*$∂+l], ϕ[l])
            end
            lb = ub
        end
        ϕ = fillΦ_flat(knots, knotsₗ, knots[knotsₗ+$m], Val{$d}(), knotsₗ+$p)
        @nexprs $p l -> begin
            push!(∂Φ[(knotsₗ-1)*$∂+l], ϕ[l+1])
        end
#        fillΦ_flat!(∂Φᵗ, knots, length(knots), knots[length(knots)+$m], Val{$d}(), length(knots)+$p)
        W = gen_W( Δknots, q, Val{$∂}(), Val{$∂p1}() )
  #      ∂Φ, W

        pen_mat = fill(zero(T), $m, k)
        pen_gen_mult!(pen_mat, ∂Φ, W, Val{$p}(), Val{$∂}())
        pen_mat
 #       W, ∂Φᵗ
    end
end

@generated function pen_gen_mult!(C, A::Vector{Vector{T}}, B, ::Val{p}, ::Val{∂}) where {p,∂,T}
    m = p+1
    quote
        @nexprs $p i -> begin
            Aᵢ = A[i]
            @nexprs i-1 j -> begin
                t = zero(T)
                Aⱼ = A[j]
                for k ∈ eachindex(Aᵢ), l ∈ eachindex(Aⱼ)
                    t += Aᵢ[k] * Aⱼ[l] * B[l,k]
                end
                C[$m-i+j,i] = t
            end
            t = zero(T)
            for k ∈ eachindex(Aᵢ)
                for l ∈ 1:k-1
                    t += 2Aᵢ[k] * Aᵢ[l] * B[l,k]
                end
                t += abs2(Aᵢ[k]) * B[k,k]
            end
            C[$m,i] = t
        end
            
        for i ∈ $m:length(A)
            Aᵢ = A[i]
            @nexprs $m j -> begin
                t = zero(T)
                jᵢ = i+j-$m
                Aⱼ = A[jᵢ]
                for k ∈ eachindex(Aᵢ), l ∈ eachindex(Aⱼ)
                    t += Aᵢ[k] * Aⱼ[l] * B[max(0, 1 + $∂*(jᵢ-$m)) + l, 1 + $∂*(i-$m) + k]
                end
                C[j,i] = t
            end
        end
    end
end



function ginv!(X)
    X, piv, inf = Base.LinAlg.LAPACK.getrf!(X)
    Base.LinAlg.LAPACK.getri!(X, piv)
end

@inline function Base.setindex!(w::BandedMatrices.SymBandedMatrix, v, i, j)
    BandedMatrices.symbanded_setindex!(w.data,w.k, v, i, j)
end
# When you want to turn off bounds checking:
# syminbands_setindex!

@generated function H_mat(::Val{∂p1}) where {∂p1}
    ∂ = ∂p1 - 1
    H = Matrix{Float64}(∂p1,∂p1)
    for i ∈ 1:∂p1, j ∈ 1:∂p1
        H[j,i] = (-1  + 2(j-1)/∂ )^i 
    end
    P = inv(H)
    for i ∈ 1:∂p1
        for j ∈ 1:i
            H[j,i] = ( 1 + (-1)^(i+j-2) ) / (i+j-1)
        end
    end
    Base.LinAlg.LAPACK.potrf!('U', H)
    Base.LinAlg.BLAS.trmm!('L', 'U', 'N', 'N', 1.0, H, P) #Updates P
    Base.LinAlg.BLAS.syrk!('U', 'T', 0.5, P, 0.0, H) #Updates H

    for i ∈ 2:∂p1, j ∈ 1:i-1 #Should pay off in terms of fast indexing.
        H[i,j] = H[j,i]
    end
#    SMatrix{∂p1,∂p1}(H)
    H
end

@generated function gen_W( Δ, q::Int, ::Val{∂}, ::Val{∂p1} ) where {∂,∂p1} #q = l - 1
    ∂m1 = ∂ - 1
    quote
        H = H_mat(Val{$∂p1}())
    #  println(H)

        n = 1+q*$∂
        #Wants upper triangular index pattern, of decreasing abs(i - j) on each step
        W = BandedMatrices.sbzeros(Float64, n, $∂)


        for qᵢ ∈ 0:q-1
            @nexprs $∂p1 j -> begin
                @nexprs j-1 i -> begin
                    W[i + $∂*qᵢ, j + $∂*qᵢ] += 2Δ[qᵢ+1] * H[i, j]
                end
                W[j + $∂*qᵢ, j + $∂*qᵢ] += Δ[qᵢ+1] * H[j, j]
            end
        end
        W
    end
end

@generated function gen_W2( Δ, q::Int, ::Val{∂}, ::Val{∂p1} ) where {∂,∂p1} #q = l - 1
    ∂m1 = ∂ - 1
    quote
        H = H_mat(Val{$∂p1}())
    #  println(H)

        n = 1+q*$∂
        #Wants upper triangular index pattern, of decreasing abs(i - j) on each step
        W = SymBandedMatrix(Float64, n, $∂)


        for qᵢ ∈ 0:q-1
            @nexprs $∂p1 j -> begin
                @nexprs $∂p1 i -> begin
                    W[i + $∂*qᵢ, j + $∂*qᵢ] += Δ[qᵢ+1]
                end
            end
        end
        @nexprs $∂ i -> begin
            @nexprs i j -> begin
                W[j, i] = H[j,i] * Δ[1]
            end
        end
        @nexprs $∂ j -> begin
            W[j, $∂p1] = H[j,$∂p1] * Δ[1]
        end
        for qᵢ ∈ 1:q-1 #do it for q = 0 above this.
            indⱼ = 1+qᵢ*∂
            W[indⱼ, indⱼ] = H[1,1] * Δ[qᵢ+1] + H[∂p1, ∂p1] * Δ[qᵢ]
            @nexprs $∂ j -> begin
                W[1+j+qᵢ*∂, indⱼ] = H[1+j,1] * Δ[qᵢ+1] #+ H[j, ∂p1]
            end
            @nexprs $∂m1 i -> begin
                indᵢ = i+1+qᵢ*∂
    #            W[indᵢ, indⱼ] = H[i,1] + H[i,∂p1]
                @nexprs i j -> begin
                    W[1+j+qᵢ*∂, indᵢ] = H[j+1,i+1] * Δ[qᵢ+1]
                end
            end
        end
        @nexprs $∂p1 j -> begin
            W[j+(q-1)*$∂, n] = H[j,∂p1]*Δ[q]
        end
        W
    end


end

@inline Base.unique(s::BSpline{K} where K <: CardinalKnots) = s.knots.n
@inline Base.unique(s::BSpline{K} where K <: DiscreteKnots) = s.knots.n


###Use exp(λ)*(max-min)^(2m-1) as penalty coefficient.


function create_breaks(x::AbstractVector{T}, p::Int) where {T}
    breaks = p-1
    bm1 = breaks - 1
    nₓ = length(x)
    nᵢ = nₓ ÷ breaks
    n = nₓ - breaks
    out = Vector{T}(uninitialized, n)
    for i ∈ eachindex(out)
        out[i] = x[ nₓ * i ÷ n ]
    end
    out
end


"""
B-spline basis with a penalty proportional to λ placed on the d-th derivative, specified by passing Val(d). 
The default behavior is to evenly place knots on the data points, until there are the correct number of coefficients.
"""
function BSpline(x::Vector{T}, y::Vector, λ::Real, ::Val{d} = Val{2}(), k::Int = div(length(x),2), knots::Knots{p} = DiscreteKnots(create_breaks(x, 3+k), Val{3}())) where {T, p, d}
    m = p+1
    β = Vector{T}(k)
    x_member, min_k, max_k = k_structure!(x, y, knots, p, k)

    Φᵗ = zeros(promote_type(T, Float64), m, length(x))
    fillΦ_sorted!(Φᵗ, x, knots, x_member)
    Φ = jagged_semiband_transpose(Φᵗ, x_member, min_k, max_k, p)

    ΦᵗΦ = Array{Float64, 2}(uninitialized, m, k)
    fill_band!(ΦᵗΦ, Φ, min_k, max_k, Val{p}())

    pen = BSplinePenalty(knots, k, Val{d}())

    Base.LinAlg.BLAS.axpy!(λ, pen, ΦᵗΦ)

    β, Φᵗy, ΦᵗΦ⁻ = solve!(Array{Float64, 2}(uninitialized, k, k), ΦᵗΦ, β, Φ, y, min_k, max_k, Val{p}())

    BSpline(knots, [β], ΦᵗΦ⁻, Φ, Φᵗ, Φᵗy, y, Val{p}(), k)

end

function BSpline(x::Vector{T}, y::Vector, λ::AbstractVector, ::Val{d} = Val{2}(), k::Int = div(length(x),2), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p, d}


 
    m = p+1
    β = Vector{T}(k)
    x_member, min_k, max_k = k_structure!(x, y, knots, p, β)

    Φᵗ = zeros(promote_type(T, Float64), m, length(x))
    fillΦ_sorted!(Φᵗ, x, knots, x_member)
    Φ = jagged_semiband_transpose(Φᵗ, x_member, min_k, max_k, p)

    ΦᵗΦ = Array{Float64, 2}(uninitialized, m, k)
    fill_band!(ΦᵗΦ, Φ, min_k, max_k, Val{p}())

    pen = BSplinePenalty(knots, k, Val{d}())
    
    BandedMatrices.pbstf!('U', k, p, pointer(ΦᵗΦ), m)
    BandedMatrices.sbgst!(vect::Char, uplo::'U', n::Int, ka::Int, kb::Int,
    AB::Ptr{$elty}, ldab::Int, BB::Ptr{$elty}, ldbb::Int,
    X::Ptr{$elty}, ldx::Int, work::Ptr{$elty})

    Base.LinAlg.BLAS.axpy!(λ, pen, ΦᵗΦ)

    β, Φᵗy, ΦᵗΦ⁻ = solve!(Array{Float64, 2}(uninitialized, k, k), ΦᵗΦ, β, Φ, y, min_k, max_k, Val{p}())

    BSpline(knots, [β], ΦᵗΦ⁻, Φ, Φᵗ, Φᵗy, y, Val{p}(), k)
end

#generalized eigen
#Base.LinAlg.LAPACK.sygvd!