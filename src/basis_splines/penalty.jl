

### Okay, for penalty, you can forgo calculation of G altogether.
### J = β'Sβ, S = G'WG, therefore J = (Gβ)'WGβ. You have already implemented the method for calculating Gβ!!! Alternatively, should be straightforward to do with fillΦ! as well.
### Question is -- what is easiest? Actual calculation of Gβ would aid solving via linear algebra.
###Fix later for cardinal knots. Basically, replace first for loop.
function BSplinePenalty(s::BSpline{K, p, T} where K <: DiscreteKnots, d::Int) where {p, T, d}
    xv = Vector{T}((unique(s)-1)*p+1)
    ∂Φ = zeros(s.k, length(xv))
    lb = s.knots[1]
    ∂ = p - d
    for i ∈ 0:length(s.knots)-2
      ub = s.knots[i]
      isa(s.knots, DiscreteKnots) && lb == ub && continue
      Δ = (ub - lb) / (∂+1)
      for j ∈ 0:∂-1
          xv[i*∂+j+1] = lb + Δ * j
      end
#      [i*∂+1:(i+1)*∂] .= lb:Δ:ub-Δ
      lb = ub
    end
    fillΦ!(∂Φ, s.buffer, xv, s.knots, ∂)
    expandΦ!(∂Φ, s.knots, d) ###Produces G. Gβ == fillΦ! * s.coefs[d+1]
end

function expandΦ!(Φ, t::Knots{p}, d::Int) where p
    Threads.@threads for i ∈ 1:size(Φ,2)
        for j ∈ 1:d
            pjmd = p + j - d
            oldscaled = Φ[1,i] * pjmd / range(t, 2+p, 2+d-j)
            Φ[1,i] = - oldscaled
            for k ∈ 2:size(Φ,1) - d + j - 1
                newscaled = Φ[k,i] * pjmd / range(t, 1+k+p, 1+k+d-j)
                Φ[k,i] = oldscaled - newscaled
                oldscaled = newscaled
            end
            Φ[end+d-j,i] = oldscaled
        end

    end

end
function ginv!(X)
    X, piv, inf = Base.LinAlg.LAPACK.getrf!(X)
    Base.LinAlg.LAPACK.getri!(X, piv)
end
function gen_Lw(∂)
    H = Matrix{Float64}(∂+1,∂+1)
#    H = similar(P)
    for i ∈ 1:∂+1, j ∈ 1:∂+1
        P[j,i] = (-1  + 2(i-1)/∂ )^j
    end
    P = inv(H)
    for i ∈ 1:∂+1
        H[i,i] = ( 1 + (-1)^(2i-2) ) / (2i-1)
        for j ∈ i+1:∂+1
            H[j,i] = ( 1 + (-1)^(i+j-2) ) / (i+j-1)
        end
    end
    Base.LinAlg.LAPACK.potrf!('U', H)
    Base.LinAlg.BLAS.trmm!('L', 'U', 'N', 'N', 1.0, H, P) #Updates P
    Base.LinAlg.BLAS.syrk!('U', 'T', 1.0, P, 0.0, H)

    W = Base.LinAlg.BLAS.symm('L', 'U')

end

@inline unique(s::BSpline{K} where K <: CardinalKnots) = s.knots.n
@inline unique(s::BSpline{K} where K <: DiscreteKnots) = s.knots.n


###Use exp(λ)*(max-min)^(2m-1) as penalty coefficient.


function BSpline(x::Vector{T}, y::Vector, k::Int = div(length(x),10), knots::Knots{p} = CardinalKnots(x, k, Val{3}())) where {T, p}
    n = length(x)
    Φᵗ = zeros(promote_type(T, Float64), k, n)
    vector_buffer = Vector{T}(p+1)
    matrix_buffer = Matrix{T}(p,p)
    fillΦ!(Φᵗ, matrix_buffer, x, knots)
#    Φt, y, knots, vector_buffer
#    println("P: $p, size of Φ: $(size(Φᵗ)), y: $(size(y))")
    β, ΦᵗΦ⁻ = solve(Φᵗ, y)
    BSpline(knots, [β], ΦᵗΦ⁻, vector_buffer, matrix_buffer, Φᵗ, y, Val{p}(), k)
end
