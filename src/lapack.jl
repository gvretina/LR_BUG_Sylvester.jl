# Thin wrappers around the LAPACK routines we need, with the workspace passed in by the caller,
# so that a solve allocates nothing.  (Julia's `qr!`, `svd!`, `schur!`, `eigen!` allocate
# their workspace and result on every call; FastLapackInterface.jl reuses its workspace but
# reallocates its output arrays whenever the matrix size changes, and ours change every
# iteration.)  Every wrapper works on views of larger preallocated buffers.
#
# Workspace sizes: calling a routine with lwork = -1 makes LAPACK only write the optimal
# workspace length into work[1]; the `*_lwork` functions do that once, at setup.

using LinearAlgebra: BlasInt
using LinearAlgebra.LAPACK: liblapack, chklapackerror, @blasfunc

# -- QR:  A = Q R ------------------------------------------------------------

"""Householder QR in place: R in the upper triangle of A, reflectors below (LAPACK dgeqrf)."""
function geqrf!(A::StridedMatrix{Float64}, tau, work; lwork = length(work))
    m, n = size(A); info = Ref{BlasInt}()
    ccall((@blasfunc(dgeqrf_), liblapack), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ref{BlasInt}, Ptr{BlasInt}),
          m, n, A, max(1, stride(A, 2)), tau, work, lwork, info)
    chklapackerror(info[])
    return A
end

"""Overwrite the output of `geqrf!` with the thin Q factor (LAPACK dorgqr)."""
function orgqr!(A::StridedMatrix{Float64}, tau, work; lwork = length(work))
    m, n = size(A); info = Ref{BlasInt}()
    ccall((@blasfunc(dorgqr_), liblapack), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}),
          m, n, n, A, max(1, stride(A, 2)), tau, work, lwork, info)
    chklapackerror(info[])
    return A
end

"""A ← Q, the orthonormal factor of the thin QR of A (m × n, n ≤ m)."""
orthonormalize!(A, tau, work) = orgqr!(geqrf!(A, tau, work), tau, work)

"""R ← the triangular factor of the QR of X (min(m, n) × n for X m × n; X is destroyed)."""
function rfactor!(R, X, tau, work)
    geqrf!(X, tau, work)
    fill!(R, 0.0)
    @inbounds for j in axes(R, 2), i in 1:min(j, size(R, 1))
        R[i, j] = X[i, j]
    end
    return R
end

function qr_lwork(m::Int, n::Int)
    n = min(m, n)                                     # dorgqr needs n ≤ m
    A = zeros(m, n); tau = zeros(n); w = zeros(1)
    geqrf!(A, tau, w; lwork = -1); lw = w[1]
    orgqr!(A, tau, w; lwork = -1)
    return ceil(Int, max(lw, w[1]))
end

# -- SVD:  A = U Σ Vᵀ (thin) -------------------------------------------------

"""Thin SVD by divide and conquer (LAPACK dgesdd, JOBZ = 'S'); A is destroyed."""
function gesdd!(A::StridedMatrix{Float64}, σ, U, VT, work, iwork; lwork = length(work))
    m, n = size(A); info = Ref{BlasInt}()
    ccall((@blasfunc(dgesdd_), liblapack), Cvoid,
          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt}, Ptr{BlasInt}, Clong),
          'S', m, n, A, max(1, stride(A, 2)), σ, U, max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
          work, lwork, iwork, info, 1)
    chklapackerror(info[])
    return σ
end

function svd_lwork(m::Int, n::Int)
    k = min(m, n); w = zeros(1)
    gesdd!(zeros(m, n), zeros(k), zeros(m, k), zeros(k, n), w, zeros(BlasInt, 8k); lwork = -1)
    return ceil(Int, w[1])
end

# -- Schur:  A = Z T Zᵀ (real, T quasi-triangular) and A = Z T Zᴴ (complex) -----

"""Real Schur form in place: A ← T, Z ← Schur vectors, eigenvalues in wr + i·wi (LAPACK dgees)."""
function gees!(A::StridedMatrix{Float64}, wr, wi, Z, work; lwork = length(work))
    n = size(A, 1); sdim = Ref{BlasInt}(); info = Ref{BlasInt}()
    ccall((@blasfunc(dgees_), liblapack), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Cvoid}, Ptr{BlasInt}, Clong, Clong),
          'V', 'N', C_NULL, n, A, max(1, stride(A, 2)), sdim, wr, wi, Z, max(1, stride(Z, 2)),
          work, lwork, C_NULL, info, 1, 1)
    chklapackerror(info[])
    return A
end

function gees_lwork(n::Int)
    w = zeros(1)
    gees!(zeros(n, n), zeros(n), zeros(n), zeros(n, n), w; lwork = -1)
    return ceil(Int, w[1])
end

"""Complex Schur form in place: A ← T (upper triangular), Z ← unitary Schur vectors (LAPACK zgees)."""
function zgees!(A::StridedMatrix{ComplexF64}, w, Z, work, rwork; lwork = length(work))
    n = size(A, 1); sdim = Ref{BlasInt}(); info = Ref{BlasInt}()
    ccall((@blasfunc(zgees_), liblapack), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt}, Ref{BlasInt},
           Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt},
           Ptr{Float64}, Ptr{Cvoid}, Ptr{BlasInt}, Clong, Clong),
          'V', 'N', C_NULL, n, A, max(1, stride(A, 2)), sdim, w, Z, max(1, stride(Z, 2)),
          work, lwork, rwork, C_NULL, info, 1, 1)
    chklapackerror(info[])
    return A
end

function zgees_lwork(n::Int)
    w = zeros(ComplexF64, 1)
    zgees!(zeros(ComplexF64, n, n), zeros(ComplexF64, n), zeros(ComplexF64, n, n), w, zeros(n); lwork = -1)
    return ceil(Int, real(w[1]))
end

# -- symmetric eigenproblem:  A = Q Λ Qᵀ -------------------------------------

"""Symmetric eigendecomposition in place (LAPACK dsyev): A ← eigenvectors (or destroyed if
`vectors = false`), λ ← eigenvalues in ascending order."""
function syev!(A::StridedMatrix{Float64}, λ, work; vectors::Bool = true, lwork = length(work))
    n = size(A, 1); info = Ref{BlasInt}()
    ccall((@blasfunc(dsyev_), liblapack), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
          vectors ? 'V' : 'N', 'U', n, A, max(1, stride(A, 2)), λ, work, lwork, info, 1, 1)
    chklapackerror(info[])
    return A
end

function syev_lwork(n::Int)
    n = max(n, 1); w = zeros(1)
    syev!(zeros(n, n), zeros(n), w; lwork = -1)
    return ceil(Int, w[1])
end
