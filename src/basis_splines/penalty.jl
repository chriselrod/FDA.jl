
### Okay, for penalty, you can forgo calculation of G altogether.
### J = β'Sβ, S = G'WG, therefore J = (Gβ)'WGβ. You have already implemented the method for calculating Gβ!!! Alternatively, should be straightforward to do with fillΦ! as well.
### Question is -- what is easiest? Actual calculation of Gβ would aid solving via linear algebra.
###Fix later for cardinal knots. Basically, replace first for loop.
function BSplinePenalty!(buffer::AbstractMatrix{T}, knots::Knots{p}, k::Int, d::Int) where {p, T}
    q = unique(knots)-1
    ∂ = p - d
    ∂p1 = ∂ + 1
    n = q*∂+1
    xv = Vector{T}(n)
    ∂Φ = zeros(k, length(xv))
    lb = knots[1]
    for i ∈ 0:length(knots)-2
      ub = knots[i+p+2]
      isa(knots, DiscreteKnots) && lb == ub && continue
      Δ = (ub - lb) / ∂p1
      for j ∈ 0:∂-1
          xv[i*∂+j+1] = lb + Δ * j
      end
#      [i*∂+1:(i+1)*∂] .= lb:Δ:ub-Δ
      lb = ub
    end
    xv[end] = lb
    println(xv)
    println(knots.min, "\n", knots.v, "\n", knots.max)
    fillΦ!(∂Φ, buffer, xv, knots, ∂)
    expandΦ!(∂Φ, knots, d) ###Produces G. Gβ == fillΦ! * s.coefs[d+1]
    W, H, P = gen_W( ∂, q, ∂p1 )
#   g1 = ∂Φ * W
#   A_mul_Bt!(H, g1, ∂Φ)
#   Should test which method is faster.
    println(W)
    cholfact!(W)
    U = BandedMatrix( W.data[1:∂p1,:], n, 0, ∂)
    g = A_mul_Bt(U, ∂Φ)
    println(size(g), k, q)
    Base.LinAlg.BLAS.syrk!('T', 'U', 1.0, g, 0.0, P)
    g, P
end

expandΦ!(args...) = Threads.nthreads() == 1 ? expandΦ_unthreaded!(args...) : expandΦ_threaded!(args...)
function expandΦ_threaded!(Φ, t::Knots, d::Int)
    for i ∈ 1:size(Φ,2)#Threads.@threads 
        for j ∈ 1:d
            expandΦcore!(Φ, t, d, i, j)
        end
    end    
end
function expandΦ_unthreaded!(Φ, t::Knots, d::Int)
    for i ∈ 1:size(Φ,2)#Threads.@threads 
        for j ∈ 1:d
            expandΦcore!(Φ, t, d, i, j)
        end
    end        
end

function expandΦcore!(Φ, t::Knots{p}, d::Int, i::Int, j::Int) where p
    pjmd = p + j - d
    oldscaled = Φ[1,i] * pjmd / range(t, 2+p, 2+d-j)
    Φ[1,i] = - oldscaled
    for k ∈ 2:size(Φ,1) - d + j - 1
        newscaled = Φ[k,i] * pjmd / range(t, 1+k+p, 1+k+d-j)
        Φ[k,i] = oldscaled - newscaled
        oldscaled = newscaled
    end
    Φ[end-d+j,i] = oldscaled

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

gen_W(∂::Int, q::Int, ∂p1::Int = ∂+1) = gen_W!(Matrix{Float64}(∂p1,∂p1), ∂, q, ∂p1)
function gen_W!(H, ∂::Int, q::Int, ∂p1::Int = ∂+1) #q = l - 1
    for i ∈ 1:∂p1, j ∈ 1:∂p1
        H[j,i] = (-1  + 2(i-1)/∂ )^j
    end
    P = inv(H)
    for i ∈ 1:∂p1
        for j ∈ 1:i
            H[j,i] = ( 1 + (-1)^(i+j-2) ) / (i+j-1)
        end
    end
    Base.LinAlg.LAPACK.potrf!('U', H)
    Base.LinAlg.BLAS.trmm!('L', 'U', 'N', 'N', 1.0, H, P) #Updates P
    Base.LinAlg.BLAS.syrk!('U', 'T', 1.0, P, 0.0, H) #Updates H

    for i ∈ 2:∂p1, j ∈ 1:i-1 #Should pay off in terms of fast indexing.
        H[i,j] = H[j,i]
    end

    println(H)

    n = 1+q*∂
    #Wants upper triangular index pattern, of decreasing abs(i - j) on each step
    W = SymBandedMatrix(Float64, n, ∂)


    for i ∈ 1:∂p1, j ∈ 1:i
        W[j, i] = H[j,i]
    end
    for qᵢ ∈ 1:q-1 #do it for q = 0 above this.
        indⱼ = 1+qᵢ*∂
        W[indⱼ, indⱼ] = H[1,1] + H[∂p1, ∂p1]
        for j ∈ 2:∂+1
            W[j+qᵢ*∂, indⱼ] = H[j,1] #+ H[j, ∂p1]
        end
        for i ∈ 2:∂
            indᵢ = i+qᵢ*∂
#            W[indᵢ, indⱼ] = H[i,1] + H[i,∂p1]
            for j ∈ 2:i
                W[j+qᵢ*∂, indᵢ] = H[j,i]
            end
        end
    end
    for j ∈ 1:∂+1
        W[j+(q-1)*∂, n] = H[j,∂p1]
    end
    W, H, P

#    W = Base.LinAlg.BLAS.symm('L', 'U')

end

@inline unique(s::BSpline{K} where K <: CardinalKnots) = s.knots.n
@inline unique(s::BSpline{K} where K <: DiscreteKnots) = s.knots.n


###Use exp(λ)*(max-min)^(2m-1) as penalty coefficient.


function BSpline(x::Vector{T}, y::Vector, λ::Real, d::Int = 2, k::Int = div(length(x),2), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    m = p+1
    n = length(x)
    Φᵗ = zeros(promote_type(T, Float64), k, n)
    buffer = Matrix{T}(m,m)
    g, R = BSplinePenalty!(buffer, knots, k, d)
    fillΦ!(Φᵗ, buffer, x, knots)
    
    β, ΦᵗΦ⁻ = solve!(R, Φᵗ, y, λ)

    BSpline(knots, [β], ΦᵗΦ⁻, buffer, Φᵗ, y, Val{p}(), k)
end

function BSpline(x::Vector{T}, y::Vector, λ::AbstractVector, d::Int = 2, k::Int = div(length(x),2), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    m = p+1
    n = length(x)
    Φᵗ = zeros(promote_type(T, Float64), k, n)
    buffer = Matrix{T}(m,m)
    g, R = BSplinePenalty!(buffer, knots, k, d)
    fillΦ!(Φᵗ, buffer, x, knots)
    Base.LinAlg.BLAS.syrk!('U', 'N', 1.0, Φᵗ, 0.0, buffer)
    Λ, R, buffer = Base.LinAlg.LAPACK.sygvd!( 1, 'V', 'U', R, buffer )
    D = Diagonal(similar(Λ))
    for λᵢ ∈ λ
        for i ∈ eachindex(Λ)
            D.diag[i] = 1 / (1 + Λ[i])
        end

    end
    
    BSpline(knots, [β], ΦᵗΦ⁻, buffer, Φᵗ, y, Val{p}(), k)
end

#generalized eigen
#Base.LinAlg.LAPACK.sygvd!