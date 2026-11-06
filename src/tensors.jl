# N-dimensional tensors:
#   n_mode_product   Y = X ×ₙ U                       (U: p × dₙ)
#   matricize        Mat_n(X): mode n in the rows, the other modes in the columns with the
#                    lower modes fastest (Kolda's ordering); `transposed = true`: Mat_n(X)ᵀ
#   tensorize        the inverse of `matricize`
#   nonmodes         the modes other than n
#
# Inside the solvers a tensor is a flat preallocated vector plus its dims d = (d₁, ..., d_N)
# (column-major), because the same buffers hold tensors of changing size.  The in-place routines
# (`n_mode_product!`, `n_mode_products!`, `matricize!`)
# view such a tensor as the 3-D array (I_pre, dₙ, I_post), I_pre = d₁⋯dₙ₋₁, I_post = dₙ₊₁⋯d_N:
# a mode-n product is then one BLAS product per index of I_post, for any N, with no allocation.
# The allocating `matricize` / `tensorize` / `n_mode_product` (permutedims) are for problem setup
# and the exact references only.

@views flatmat(v, a, b) = reshape(v[1:a*b], a, b)

"""The modes other than n, in increasing order (N = length of the dims tuple d)."""
nonmodes(::NTuple{N,Int}, n::Int) where {N} = ntuple(q -> q < n ? q : q + 1, Val(N - 1))

"""(I_pre, I_post) = (d₁⋯dₙ₋₁, dₙ₊₁⋯d_N)."""
function pre_post(d::NTuple{N,Int}, n::Int) where {N}
    pre, post = 1, 1
    for m in 1:N
        m < n && (pre *= d[m])
        m > n && (post *= d[m])
    end
    return pre, post
end

"""
    n_mode_product!(Y, X, d, U, n; α = 1, β = 0) -> dims of Y

Y = α X ×ₙ U + β Y for flat tensors (X has dims d; U is p × dₙ and may be a transpose).
"""
@views function n_mode_product!(Y, X, d::NTuple{N,Int}, U, n::Int; α = true, β = false) where {N}
    pre, post = pre_post(d, n); dn = d[n]; p = size(U, 1)
    if n == 1
        mul!(flatmat(Y, p, post), U, flatmat(X, dn, post), α, β)
    elseif n == N
        mul!(flatmat(Y, pre, p), flatmat(X, pre, dn), transpose(U), α, β)
    else                                          # one product per index of the modes after n
        for q in 1:post
            mul!(reshape(Y[(q-1)*pre*p+1:q*pre*p], pre, p), reshape(X[(q-1)*pre*dn+1:q*pre*dn], pre, dn),
                 transpose(U), α, β)
        end
    end
    return Base.setindex(d, p, n)
end

"""
    n_mode_products!(dest, X, d, Us, (T1, T2); skip = 0) -> dims of dest

dest = X ×₁ Us[1] ×₂ Us[2] ⋯ ×_N Us[N] for flat tensors, leaving out mode `skip` (if any), through
the scratch vectors T1, T2 (both distinct from dest).
"""
function n_mode_products!(dest, X, d::NTuple{N,Int}, Us, (T1, T2); skip::Int = 0) where {N}
    last = skip == N ? N - 1 : N
    cur, nxt = X, T1
    for n in 1:N
        n == skip && continue
        out = n == last ? dest : nxt
        d = n_mode_product!(out, cur, d, Us[n], n)
        cur, nxt = out, (out === T1 ? T2 : T1)
    end
    return d
end

"""
    matricize!(W, X, d, n; transposed = false) -> W

W = Mat_n(X) (dₙ × the product of the other dims) for a flat tensor X of dims d, or its transpose.
"""
function matricize!(W, X, d::NTuple{N,Int}, n::Int; transposed::Bool = false) where {N}
    pre, post = pre_post(d, n); dn = d[n]
    @inbounds for q in 1:post, k in 1:dn, i in 1:pre
        v = X[i + pre * ((k - 1) + dn * (q - 1))]
        col = i + pre * (q - 1)
        if transposed
            W[col, k] = v
        else
            W[k, col] = v
        end
    end
    return W
end

"""Mat_n(X) of an array (allocating): mode n first, then the other modes in order."""
matricize(X::AbstractArray{T,N}, n::Int) where {T,N} =
    reshape(permutedims(X, (n, nonmodes(size(X), n)...)), size(X, n), :)

"""The array of dims d whose Mat_n is W (allocating; inverse of `matricize`)."""
function tensorize(W::AbstractMatrix, n::Int, d::NTuple{N,Int}) where {N}
    perm = (n, nonmodes(d, n)...)
    return permutedims(reshape(W, map(m -> d[m], perm)), invperm(perm))
end

"""Y = X ×ₙ U (allocating; problem setup and references only)."""
n_mode_product(X::AbstractArray{T,N}, U::AbstractMatrix, n::Int) where {T,N} =
    tensorize(U * matricize(X, n), n, Base.setindex(size(X), size(U, 1), n))

"""The full tensor G ×₁ U[1] ⋯ ×_N U[N] (references only)."""
full_tensor(U, G) = foldl((X, k) -> n_mode_product(X, U[k], k), 1:length(U); init = G)

"""Sum of squares of the first n entries of v."""
function sumsq(v, n)
    s = 0.0
    @inbounds for i in 1:n
        s += abs2(v[i])
    end
    return s
end
