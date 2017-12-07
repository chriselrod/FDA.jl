
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
function solve(Xᵗ, y, t::Char = 'N')
    XᵗXⁱ = Base.LinAlg.BLAS.syrk('U', t, 1.0, Xᵗ)
    Base.LinAlg.LAPACK.potrf!('U', XᵗXⁱ);
    Base.LinAlg.LAPACK.potri!('U', XᵗXⁱ);
    β = Vector{eltype(Xᵗ)}(size(Xᵗ,1))
    A_mul_B!(β, Xᵗ, y)
    Base.LinAlg.BLAS.symv('U', XᵗXⁱ, Xᵗ * y), Symmetric(XᵗXⁱ)
end
function solve!(XᵗXⁱ, Xᵗ, y, t::Char = 'N', coef = 0.0)
    Base.LinAlg.BLAS.syrk!('U', t, 1.0, Xᵗ, coef, XᵗXⁱ)
    Base.LinAlg.LAPACK.potrf!('U', XᵗXⁱ);
    Base.LinAlg.LAPACK.potri!('U', XᵗXⁱ);
    Base.LinAlg.BLAS.symv('U', XᵗXⁱ, Xᵗ * y), Symmetric(XᵗXⁱ)
end

function inv2!(A, B)
    copy!(A, B)
    Base.LinAlg.LAPACK.potrf!('U', A)
    Base.LinAlg.LAPACK.potri!('U', A)
end
function inv3!(A, B)
    copy!(A, B)
    Base.LinAlg.LAPACK.potri!('U', A)
end


function simultaneous_sort!(x::AbstractArray{T}, y::AbstractArray{T}) where T
    issorted(x, Base.Order.ForwardOrdering()) || sort!(StructOfArrays{Tuple{T,T},1,Tuple{Vector{T},Vector{T}}}((x,y)), QuickSort, Base.Order.ForwardOrdering())
end

function chol!(U::AbstractArray{<:Real,2}, Σ::AbstractArray{<:Real,2})
#    @inbounds for i ∈ 1:size(U,1)
    for i ∈ 1:size(U,1)
        U[i,i] = Σ[i,i]
        for j ∈ 1:i-1
            U[j,i] = Σ[j,i]
            for k ∈ 1:j-1
                U[j,i] -= U[k,i] * U[k,j]
            end
            U[j,i] /= U[j,j]
            U[i,i] -= U[j,i]^2
        end
        U[i,i] = √U[i,i]
    end
end
function inv!(A::AbstractArray{T,2}, U::AbstractArray{T,2}) where {T <: Real}
#    @inbounds for i ∈ 1:size(U,1)
    for i ∈ 1:size(U,1)
        A[i,i] = 1 / U[i,i]
        for j ∈ i+1:size(U,1)
            Aⱼᵢ = zero(T)
            for k ∈ i:j-1
                Aⱼᵢ += U[k,j] * A[k,i]
            end
            A[j,i] = - Aⱼᵢ / U[j,j]
        end
    end
    A
end


function chol!(U::AbstractArray{<:Real,2}, Σ::AbstractArray{<:Real,2})
    #    @inbounds for i ∈ 1:size(U,1)
        for i ∈ 1:size(U,1)
            U[i,i] = Σ[i,i]
            for j ∈ 1:i-1
                U[j,i] = Σ[j,i]
                for k ∈ 1:j-1
                    U[j,i] -= U[k,i] * U[k,j]
                end
                U[j,i] /= U[j,j]
                U[i,i] -= U[j,i]^2
            end
            U[i,i] = √U[i,i]
        end
    end
@generated function chol!(U::AbstractArray{T,2}, Σ::AbstractArray{T,2}, ::Val{p}) where {T,p}
    m = p+1
    quote
    #    @inbounds for i ∈ 1:size(U,2)
        @nexprs $p i -> begin
            Uᵢᵢ = Σ[$m,i]
            @nexprs i-1 l -> begin
                j = l - i + $m
                Uⱼᵢ = Σ[j,i]
                @nexprs l-1 k -> begin
                    Uⱼᵢ -= U[k-i+$m,i] * U[k-l+$m,l]
                end
                Uⱼᵢ /= U[$m,l]
                Uᵢᵢ -= Uⱼᵢ^2
                U[j,i] = Uⱼᵢ
            end
            U[$m,i] = √Uᵢᵢ
        end

        for i ∈ $m:size(U,2)
            
            Uᵢᵢ = Σ[$m,i]
            @nexprs $p j -> begin
                l = j + i - $m
                Uⱼᵢ = Σ[j,i]
                @nexprs j-1 k -> begin
                    Uⱼᵢ -= U[k,i] * U[k-j+$m,l]
                end
                Uⱼᵢ /= U[$m,l]
                Uᵢᵢ -= Uⱼᵢ^2
                U[j,i] = Uⱼᵢ
            end
            U[$m,i] = √Uᵢᵢ
        end
    end
end

@generated function inv!(A::AbstractArray{T,2}, U::AbstractArray{T,2}, ::Val{p}) where {T, p}
    m = p+1
    quote
        n = size(A,1)
        @inbounds for i ∈ 1:n-$p
            A[i,i] = 1 / U[$m,i]
            @nexprs $p l -> begin
                j = l+i
                Aⱼᵢ = zero(T)
                @simd for k ∈ i:j-1
                    Aⱼᵢ += U[k+$m-j,j] * A[k,i]
                end
                A[j,i] = - Aⱼᵢ / U[$m,j]
            end
            for j ∈ i+$m:n
                Aⱼᵢ = zero(T)
                @simd for k ∈ 1:$p
                    Aⱼᵢ += U[k,j] * A[k+j-$m,i]
                end
                A[j,i] = - Aⱼᵢ / U[$m,j]
            end
        end
        @nexprs $p l -> begin
            i = l + n - $p
            @inbounds A[i,i] = 1 / U[$m,i]
            @inbounds for j ∈ i+1:n
                Aⱼᵢ = zero(T)
                @simd for k ∈ i:j-1
                    Aⱼᵢ += U[k+$m-j,j] * A[k,i]
                end
                A[j,i] = - Aⱼᵢ / U[$m,j]
            end
        end
        A
    end
end

function inv2!(A::AbstractArray{T,2}, U::AbstractArray{T,2}, ::Val{p}) where {T, p}
#    @inbounds for i ∈ 1:size(U,1)
    for i ∈ 1:size(A,1)
        A[i,i] = 1 / U[$m,i]
        for j ∈ i+1:size(A,1)
            Aⱼᵢ = zero(T)
            for k ∈ max(i,j-p):j-1
                Aⱼᵢ += U[k+$m-j,j] * A[k,i]
            end
            A[j,i] = - Aⱼᵢ / U[$m,j]
        end
    end
    A
end

function sym_mul!(β, S)
    
end

n = 20;
function check_symv(n)
    A = randn(n, 2n) |> X -> X * X';
    x = rand(n);
    gemv = A * x
    symv = Base.LinAlg.BLAS.symv('U', A, x)
    correct = gemv .≈ symv
    correct_count = sum(correct)
    eight = min(8,n)
    correct_in_first_8 = sum(@view(correct[1:eight]))
    n, correct_count, n - correct_count, correct_in_first_8, eight - correct_in_first_8
end
function check_symv2(n)
    A = randn(n, 2n) |> X -> X * X';
    x = rand(n);
    gemv = A * x
    symv = Base.LinAlg.BLAS.symv('L', A, x)
    gemv .≈ symv
end