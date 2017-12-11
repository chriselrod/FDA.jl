
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
function solve!(XᵗXⁱ, XᵗX, β, X::Vector{Vector{T}}, y, min_k, max_k, ::Val{p}) where {p,T}
#    BandedMatrices.pbtrf!('U', size(XᵗX,2), p, pointer(XᵗX), p+1)
    chol!(XᵗX, Val{p}())
    inv!(XᵗXⁱ, XᵗX, Val{p}())###
    triangle_crossprod!(XᵗXⁱ)
#    Xᵗy = reinterpret(T, max_k)
    Xᵗy = similar(β)
    semiband_mul_y!(Xᵗy, X, y, min_k)
    Base.LinAlg.BLAS.symv!('L', 1.0, XᵗXⁱ, Xᵗy, 0.0, β)
    β, Xᵗy, Symmetric(XᵗXⁱ, :L)
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

function triangle_crossprod!(A::AbstractMatrix{T}) where T
    n = size(A,1)
    @inbounds for i ∈ 1:n
        temp = zero(T)
        @inbounds for k ∈ i:n
            temp += abs2(A[k,i])
        end
        @inbounds A[i,i] = temp
        for j ∈ i+1:n
            temp = zero(T)
            @inbounds for k ∈ j:n
                temp += A[k,j] * A[k,i]
            end
            @inbounds A[j,i] = temp
        end
    end
end
@generated function triangle_crossprod!(A::AbstractMatrix{T}, ::Val{c}) where {T,c}
    quote
        @nexprs $c i -> begin
            temp = zero(T)
            @nexprs $c-i+1 k -> begin
                @inbounds temp += abs2(A[k+i-1,i])
            end
            @inbounds A[i,i] = temp
            @nexprs $c-i j -> begin
                temp = zero(T)
                @nexprs $c-j+1 k -> begin
                    @inbounds temp += A[k+j-1,j+i] * A[k+j-1,i]
                end
                @inbounds A[j+i,i] = temp
            end
        end
    end
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

@generated function chol!(U::AbstractArray{T,2}, ::Val{p}) where {T,p}
    m = p+1
    quote
    #    @inbounds for i ∈ 1:size(U,2)
        @nexprs $p i -> begin
            @inbounds Uᵢᵢ = U[$m,i]
            @nexprs i-1 l -> begin
                j = l - i + $m
                @inbounds Uⱼᵢ = U[j,i]
#                @simd for k ∈  1:l-1
#                    @inbounds Uⱼᵢ -= U[k-i+$m,i] * U[k-l+$m,l]
#                end
                @nexprs l-1 k -> begin
                    @inbounds Uⱼᵢ -= U[k-i+$m,i] * U[k-l+$m,l]
                end
                @inbounds Uⱼᵢ /= U[$m,l]
                Uᵢᵢ -= Uⱼᵢ^2
                @inbounds U[j,i] = Uⱼᵢ
            end
            @inbounds U[$m,i] = √Uᵢᵢ
        end

        for i ∈ $m:size(U,2)
            
            Uᵢᵢ = U[$m,i]
            @nexprs $p j -> begin
                l = j + i - $m
                @inbounds Uⱼᵢ = U[j,i]
#                @simd for k ∈  1:j-1
#                    @inbounds Uⱼᵢ -= U[k,i] * U[k-j+$m,l]
#                end
                @nexprs j-1 k -> begin
                    @inbounds Uⱼᵢ -= U[k,i] * U[k-j+$m,l]
                end
        #        if j == 2
        #            Uⱼᵢ -= U[1,i] * U[$p,l]
        #        elseif j > 2
        #            @inbounds Uⱼᵢ -= sum( SIMD.Vec{j-1,T}( ( @ntuple j-1 k -> U[k,i] ) ) * SIMD.Vec{j-1,T}( ( @ntuple j-1 k -> U[k-j+$m,l] ) )  )
        #        end
        ##        vload(Vec{j-1,T}, U, k + $m*(i-1))
        ##        vload(Vec{j-1,T}, U, k-j+$m + $m*(l-1))
                @inbounds Uⱼᵢ /= U[$m,l]
                Uᵢᵢ -= Uⱼᵢ^2
                @inbounds U[j,i] = Uⱼᵢ
            end
            @inbounds U[$m,i] = √Uᵢᵢ
        end
    end
end

@generated function inv!(A::AbstractArray{T,2}, U::AbstractArray{T,2}, ::Val{p}) where {T, p}
    m = p+1
    quote
        n = size(A,1)
        @inbounds for i ∈ 1:n-$p
            @inbounds A[i,i] = 1 / U[$m,i]
            @nexprs $p l -> begin
                j = l+i
                Aⱼᵢ = zero(T)
                @simd for k ∈ i:j-1
                    @inbounds Aⱼᵢ += U[k+$m-j,j] * A[k,i]
                end
                @inbounds A[j,i] = - Aⱼᵢ / U[$m,j]
            end
            for j ∈ i+$m:n
                Aⱼᵢ = zero(T)
                @simd for k ∈ 1:$p
                    @inbounds Aⱼᵢ += U[k,j] * A[k+j-$m,i]
                end
                @inbounds A[j,i] = - Aⱼᵢ / U[$m,j]
            end
        end
        @nexprs $p l -> begin
            i = l + n - $p
            @inbounds A[i,i] = 1 / U[$m,i]
            @inbounds for j ∈ i+1:n
                Aⱼᵢ = zero(T)
                @simd for k ∈ i:j-1
                    @inbounds Aⱼᵢ += U[k+$m-j,j] * A[k,i]
                end
                @inbounds A[j,i] = - Aⱼᵢ / U[$m,j]
            end
        end
        A
    end
end


semiband_mul_y(X, y::AbstractVector{T}, min_k) where T = semiband_mul_y!(Vector{T}(length(min_k)), X, y, min_k)
function semiband_mul_y!(out, X::Vector{Vector{T}}, y::AbstractVector, min_k) where {T}
    for i ∈ eachindex(out)
        Xᵢ = X[i]
        temp = zero(T)
        minₖm1 = min_k[i] - 1
        for j ∈ eachindex(Xᵢ)
            temp += y[j + minₖm1] * Xᵢ[j]
        end
        out[i] = temp
    end
    out
end

function jagged_semiband_transpose(Xᵗ::AbstractMatrix{T}, x_member::AbstractVector{Int}, min_k::AbstractVector{Int}, max_k::AbstractVector{Int}, p::Int) where {T}
    k = length(min_k)
    out = Vector{Vector{T}}(uninitialized, k)
    for i ∈ 1:k
        min_k_ind = min_k[i]
        last_x_member = x_member[min_k_ind]
        temp_l = 1 + max_k[i] - min_k_ind
        temp = Vector{T}(uninitialized, temp_l)
        row = min(1 + p, i)
        for j ∈ eachindex(temp)
            ind = min_k_ind + j - 1
            current_x_member = x_member[ind]
            if last_x_member != current_x_member
                row -= current_x_member - last_x_member
                last_x_member = current_x_member
            end
            temp[j] = Xᵗ[ row , ind ]
        end
        out[i] = temp
    end
 #   println("Total size of out:", sum(length, out))
    out
end
