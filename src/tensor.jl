using LinearAlgebra
using LoopVectorization
using StaticArrays

@inline function _tuple_prod(a, nonmodes::NTuple{M,Int}) where {M}
    return prod(a[i] for i in nonmodes)
end

@inline function _perm_tuple(i::Int,n::Int)
    if i == 1
        return n
    elseif i - 1 < n
        return i - 1
    else
        return i
    end
end

@inline function _nonmode_tuple(i::Int,n::Int)
    return i < n ? i : i + 1
end


@generated function get_tuple(::Val{N}, f::F, type, args...) where {N,F}
    els = :()
    # Add each element directly
    for i in 1:N
        els =  :($els...,f($(i),args...))
    end

    return quote
        NTuple{N,type}($els)
    end
end

# @generated function get_TTN_tuple(::Val{N}, ttns::Vector{<:AbstractTTN{T}}) where {N,T}
#     exprs = [:(ttns[$i]) for i in 1:N]
#     return :(tuple($(exprs...)))
# end

@inline function _dims_tuple(i::Int,n::Int,dims::NTuple{N,Int}) where N
    return @inbounds i==1 ? dims[n] : i - 1 < n ? dims[i - 1] : dims[i]
end

function matricize(X::AbstractArray{T,N}, n::Int) where {T, N}
    @assert N > 1 "Vectors can't be matricized."
    s = size(X)
    if n == 0
        n = last(ndims(X))
    end
    @assert 1 ≤ n ≤ N "Mode n must be between 1 and N."

    # Create perm tuple without closures using a generated function
    perm = get_tuple(Val(N),_perm_tuple,Int,n)

    X_perm = permutedims(X, perm)

    # Create nonmode tuple without closures
    nonmode = get_tuple(Val(N-1),_nonmode_tuple,Int,n)

    return reshape(X_perm, s[n], _tuple_prod(s, nonmode))
end

function tensorize(X::AbstractMatrix{T}, n::Int, dims::NTuple{M,Int}) where {T,M}
    if n == 0
        n = length(dims)
    end

    s = size(X)
    @assert dims[n] == s[begin] "Dimension Mismatch"
    m = get_tuple(Val(M-1),_nonmode_tuple,Int,n)
    @assert _tuple_prod(dims,m) == s[end] "Dimension Mismatch"
    d = get_tuple(Val(M),_dims_tuple,Int,n,dims)
    perm = get_tuple(Val(M),(i::Int,n::Int)-> i==1 ? n : i - 1 < n ? i - 1 : i,Int,n)
    Xres = reshape(X,d)
    permutedims(Xres,invperm(perm))
end

function tmul!(C, A, B)
    # Get sizes and verify compatibility.
    m, nA = size(A)
    nB, p  = size(B)
    @assert nA == nB "Inner dimensions of A and B must match for matrix multiplication."
    @assert size(C, 1) == m && size(C, 2) == p "C must have dimensions m × p to store the result."
    @assert !Base.mightalias(C, A) && !Base.mightalias(C, B) "C must not alias A or B."

    @turbo for i ∈ axes(A, 1), j ∈ axes(B, 2)
        sum = zero(eltype(C))
        for k ∈ axes(A, 2)
            sum += A[i, k] * B[k, j]
        end
        C[i, j] = sum
    end
    return C
end

function khatri_rao(A::AbstractMatrix{T}, B::AbstractMatrix{S}) where {T,S}
    n = size(A, 2)
    @assert size(B, 2) == n "Matrices must have the same number of columns"
    R = promote_type(T, S)
    m, p = size(A, 1), size(B, 1)
    result = Matrix{R}(undef, m * p, n)
    @turbo for k in 1:n
        for i in 1:m
            a = A[i, k]
            for j in 1:p
                result[(i-1)*p + j, k] = a * B[j, k]
            end
        end
    end
    return result
end

@inline function khatri_rao(A::AbstractMatrix{T}, B::AbstractMatrix{T}, flag::Char) where T
    if flag == 't'
        return _t_khatri_rao(A,B)
    else
        throw("Flag is not supported for Khatri-Rao Products")
    end
end

@inline function _t_khatri_rao(A::AbstractMatrix{T}, B::AbstractMatrix{S}) where {T,S}
    n = size(A, 1)
    @assert size(B, 1) == n "Matrices must have the same number of rows"
    R = promote_type(T, S)
    m, p = size(A, 2), size(B, 2)
    result = Matrix{R}(undef, n, m * p)
    @turbo for j in 1:p
        for i in 1:m
            for k in 1:n
                result[k, (i-1)*p + j] = A[k, i] * B[k, j]
            end
        end
    end
    return result
end

function kronecker(At::AbstractArray{T}, Bt::AbstractArray{S}) where {T,S}
    # Ensure both tensors have the same number of dimensions
    n_A = ndims(At)
    n_B = ndims(Bt) 
    n = max(n_A,n_B)
    
    dim_diff = abs2(n_A - n_B)
    @assert dim_diff <= 1 "Tensors can differ by at most 1 dimension"
    R = promote_type(T, S)

    if n_A == n_B
        A = At 
        B = Bt    
    elseif n_A < n_B
        # A needs padding - add trailing dimensions of size 1
        A = reshape(At, size(At)..., 1)
        B = Bt
    else
        # B needs padding - add trailing dimensions of size 1
        A = At
        B = reshape(Bt, size(Bt)..., 1)
    end

    s1 = size(A)
    s2 = size(B)
    # Compute the output size
    result_size = @inbounds ntuple(i->s1[i]*s2[i],n)# for i in 1:ndims(A)]
    result = Array{R}(undef,result_size)

    # Iterate over each element in A using Cartesian indices
    @inbounds @simd for I_A in CartesianIndices(A)#pairs(IndexCartesian(), A)
        val_A = A[I_A]
        # Compute the corresponding block in the result tensor
        block_slices = ntuple(n) do d
            start = (I_A[d] - 1) * size(B, d) + 1
            return start:start+size(B, d)-1
        end

        # Assign val_A * B to the block (using views for efficiency)
        result[block_slices...] .= val_A .* B
    end

    return result
end


function n_mode_product(X::AbstractArray{T}, U::AbstractMatrix, n::Int) where T
    dims = size(X)
    N = length(dims)
#     n = n + 1
    # Check that the inner dimension matches: dims[n] must equal the number of columns of U.
    p = size(U)

    if n == 0
        mode = last(ndims(X))
    else
        mode = n
    end

    @assert dims[mode] == p[2] "Dimension mismatch: size(X, n) = $(dims[mode]) must equal size(U,2) = $(p[2])."

    # Create a lazy matricized view of X along mode n.
    Xmat = matricize(X, mode)  # This returns a ModeNMatrix which is an AbstractMatrix without allocating new memory.
    Ymat = @inbounds Matrix{T}(undef,p[1],size(Xmat,2))

    # Compute the matrix product. This uses BLAS and no extra allocation is needed for Xmat.
#     Ymat = U * Xmat  # Ymat has size (size(U,1), prod(dims)/dims[n])
    tmul!(Ymat,U,Xmat)
    # The new dimensions are the same as X, except that the n‑th dimension is replaced by size(U,1).
#     newdims = @inbounds ntuple(i -> i == n ? p[1] : dims[i], N)
    newdims = get_tuple(Val(N),(i,n,U,dims) -> i == n ? size(U, 1) : (@inbounds dims[i]),Int,mode,U,dims)

    # Create a lazy tensorized view from the matricized result.
    Y = tensorize(Ymat, mode, newdims)
    return Y
end


function tucker_hosvd(X; tol=1e-8)
    ndims_X = ndims(X)
    U = Vector{Matrix{eltype(X)}}(undef, ndims_X)
    tol_sq = tol^2
    # Precompute the factor matrices
    @inbounds for n in 1:ndims_X
        F = svd(matricize(X, n))

        # Precompute cumulative sum of squared singular values in reverse
        cumsum_rev = cumsum(abs2.(reverse(F.S)))

        # Find the smallest r such that the discarded variance is <= tol_abs
        r = findfirst(cumsum_rev .>= tol_sq * sum(abs2, F.S))
        r = isnothing(r) ? length(F.S) : length(F.S) - r + 1

        U[n] = F.U[:, 1:r]
    end

    # Compute core tensor using in-place operations
    G = copy(X)
    @inbounds for n in 1:ndims_X
        U_n = U[n]
        # Perform n-mode product in-place
        G = n_mode_product(G, transpose(U_n), n)
    end

    return G, U
end
