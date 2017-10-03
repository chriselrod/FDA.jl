
"""
p is the penalty degree, and k the number of coefficients in your model.
"""
function penalty_matrix!(out::Matrix{T}, buffer::Vector{T}, k::Int, p::Int) where T
    fill!(out, zero(T))
    buffer[1] = 1
    next = last =  one(T)
    for i ∈ 1:p
        for j ∈ 1:i-1
            last, next = next, buffer[j+1]
            buffer[j+1] -= last
        end
        last, next = next, one(T)
        buffer[i+1] -= last
    end
    dk2 = div(k,2)
    for i ∈ 1:dk2, j ∈ max(1,i-p):i, l ∈ 1:min(j,p+1+j-i)
        out[j,i] += buffer[l]*buffer[l+i-j]
    end
    for i ∈ dk2+1:k, j ∈ max(1,i-p):i
        out[j,i] = out[k+1-i,k+1-j]
    end
    out
end
penalty_matrix(k, p) = penalty_matrix!(Matrix{Float64}(k, k), zeros(p+1), k, p)

"""
Solve a linear system AXt' = B if t = 'N' (the default), or AXt = B if t = 'T'.
Requires Xt to be full row rank (ie, X = Xt' to be full column rank). This is the default because it is often more natural to produce the transpose of the design matrix when using column-major arrays.

Returns the solution, as well as the inverse of XᵗX (ie matrrix proportional to the sampling covariance of β-hat given independent errors).
"""
function solve(Xᵗ, y, t = 'N')
    XᵗXⁱ = Base.LinAlg.BLAS.syrk('U', t, 1.0, Xᵗ)
    Base.LinAlg.LAPACK.potrf!('U', XᵗXⁱ);
    Base.LinAlg.LAPACK.potri!('U', XᵗXⁱ);
    Base.LinAlg.BLAS.symv('U', XᵗXⁱ, Xᵗ * y), Symmetric(XᵗXⁱ)
end
