

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
      xv[i*∂+1:(i+1)*∂] .= lb:Δ:ub-Δ
      lb = ub
    end
    fillΦ!(∂Φ, s.buffer, xv, s.knots, ∂)
    expandΦ!(∂Φ)

end

function expandΦ!(Φ)

end

function gen_w()

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
