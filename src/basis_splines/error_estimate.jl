
function ssr!(s::BSpline)
    sum(abs2, Base.LinAlg.BLAS.gemm!('T', 'N', 1.0, s.Φᵗ, s.coefficients[1], -1.0, s.y))
end
function ssr(s::BSpline{K, p, T}) where {K, p, T}
    out = zero(T)
    for i ∈ eachindex(s.y)
        δ = s.y[i]
        for j ∈ eachindex(s.coefficients[1])
            δ -= s.coefficients[1][j] * s.Φᵗ[j,i]
        end
        out += abs2(δ)
    end
    out
end
function confidence_width(s::BSpline{K, p, T}, x) where {K, p, T}
    k = find_k(s.knots, x)
    fillΦ3!(s.buffer, s.mat_buffer, x, s.knots, p, k)
    out = zero(T)
    for i ∈ 1:p+1
        bufferᵢ = s.buffer[i]
        out += s.ΦᵗΦ⁻.data[i, i] * bufferᵢ^2
        for j ∈ 1:i-1
            out += 2s.ΦᵗΦ⁻.data[j+p+1, i+p+1] * bufferᵢ * s.buffer[j]
        end
    end
    out
end

function scaleVar!(s::BSpline{K, p}) where {K, p}
    σ = ssr(s) / (length(s.y) - s.knots.n - p + 1)
    s.ΦᵗΦ⁻ ./= σ
end

function gcv(s::BSpline{K, p}) where {K, p}
    ssr(s) * length(s.y) / (length(s.y) - s.knots.n - p + 1)^2
end
function gcv!(s::BSpline{K, p}) where {K, p}
    nmdf = length(s.y) - s.knots.n - p + 1
    σ = ssr(s) / nmdf
    s.ΦᵗΦ⁻.data ./= σ
    σ * length(s.y) / (length(s.y) - s.knots.n - p + 1)
end