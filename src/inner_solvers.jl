# Inner solvers: the large linear systems inside each method.
#
#   K-step    A K + K Mᵀ = R                 (BUG; K, R: n × r, M: r × r)
#   shifted   X = (A − ξI)⁻¹ W,  ξ real      (RKSM, TBRK)
#   pair      Y = [Re y  Im y],  y = (A − ξI)⁻¹ W,  ξ complex   (RKSM, TBRK: a conjugate pair of
#             poles in real arithmetic)
#
# Four solvers:
#
#   BandedLDLt       tridiagonal A (1-D Laplacian): LDLᵀ factorisation of A + σI (LinearAlgebra's
#                    `ldlt!` on a SymTridiagonal), O(n) per right-hand side.  K-step: M = E Λ Eᵀ
#                    (symmetric) ⇒ r independent shifted systems (A + λⱼ I) k̃ⱼ = r̃ⱼ, K̃ = K E.
#   BandedLU         non-symmetric tridiagonal A (convection-diffusion): LU with partial pivoting
#                    of A + σI (LAPACK gtsv), O(n) per right-hand side.  K-step: complex Schur
#                    form M = Q T Qᴴ ⇒ r shifted systems (A + Tⱼⱼ I) k̃ⱼ = ..., coupled
#                    triangularly, solved from the last one back.
#   GMRES            dense A (random problems), shifted and pair solves: restarted GMRES
#                    (Krylov.jl) on the whole block, the equation A X + X Mᵀ = W with M = −ξ I
#                    (shifted) or M = [aI −bI; bI aI], a + ib = −ξ (pair).
#   BlockKrylovKStep dense A, BUG only: every K-step right-hand side lies in span(C) (C = Cu or
#                    Ubₖ, the RHS factor), so ONE block Krylov basis of K_m(A, C), built once per
#                    solve and extended only when needed, serves all K-steps by Galerkin projection.

# -- tridiagonal LDLᵀ (1-D Laplacian) -----------------------------------------

struct BandedLDLt
    d::Vector{Float64}; e::Vector{Float64}                       # diagonal, off-diagonal of A
    T::SymTridiagonal{Float64,Vector{Float64}}                   # A + σI, then its LDLᵀ factors
    Tz::SymTridiagonal{ComplexF64,Vector{ComplexF64}}            # the same for a complex σ
    xz::Vector{ComplexF64}                                       # one complex right-hand side
    E::Matrix{Float64}; λ::Vector{Float64}; syevwork::Vector{Float64}   # K-step: M = E Λ Eᵀ
    Kt::Matrix{Float64}                                          # K-step: R E, then K E (n × rmax)
end

"""BandedLDLt(A; rmax = 0): workspace for K-steps of rank ≤ rmax (0: shifted solves only)."""
function BandedLDLt(A::SparseMatrixCSC; rmax::Int = 0)
    n = size(A, 1); d = Vector(diag(A)); e = Vector(diag(A, 1))
    BandedLDLt(d, e, SymTridiagonal(copy(d), copy(e)), SymTridiagonal(complex(d), complex(e)),
               zeros(ComplexF64, n), zeros(rmax, rmax), zeros(rmax), zeros(syev_lwork(rmax)), scratch(n, rmax))
end

"""Overwrite T with A + σI and factor it in place (LDLᵀ; no pivoting needed: A + σI is
definite for every shift used here).  Returns the factorisation, for `ldiv!`."""
function factor_shifted!(T, s::BandedLDLt, σ)
    T.dv .= s.d .+ σ
    copyto!(T.ev, s.e)
    return ldlt!(T)
end

"""K-step A K + K Mᵀ = R (M symmetric up to rounding): with M = E Λ Eᵀ, the columns of
K̃ = K E solve the independent tridiagonal systems (A + λⱼ I) k̃ⱼ = r̃ⱼ, R̃ = R E."""
@views function kstep!(s::BandedLDLt, K, R, M)
    r = size(M, 1)
    E = s.E[1:r, 1:r]; λ = s.λ[1:r]
    for j in 1:r, i in 1:r
        E[i, j] = 0.5 * (M[i, j] + M[j, i])
    end
    syev!(E, λ, s.syevwork)                          # M = E Λ Eᵀ
    Kt = s.Kt[:, 1:r]
    mul!(Kt, R, E)                                   # R̃ = R E
    for j in 1:r
        ldiv!(factor_shifted!(s.T, s, λ[j]), Kt[:, j])    # k̃ⱼ = (A + λⱼ I)⁻¹ r̃ⱼ
    end
    mul!(K, Kt, E')                                  # K = K̃ Eᵀ
    return K
end

"""X = (A − ξI)⁻¹ W, ξ real: one factorisation for all columns."""
function shifted_solve!(s::BandedLDLt, X, W, ξ::Float64)
    copyto!(X, W)
    return ldiv!(factor_shifted!(s.T, s, -ξ), X)
end

"""Y = [Re y  Im y], y = (A − ξI)⁻¹ W, ξ complex: complex factorisation, column by column."""
@views function pair_solve!(s::BandedLDLt, Y, W, ξ::ComplexF64)
    F = factor_shifted!(s.Tz, s, -ξ); w = size(W, 2)
    for j in 1:w
        s.xz .= W[:, j]
        ldiv!(F, s.xz)
        Y[:, j] .= real.(s.xz); Y[:, w + j] .= imag.(s.xz)
    end
    return Y
end

# -- tridiagonal LU (convection-diffusion: A not symmetric) ------------------

struct BandedLU
    dl::Vector{Float64}; d::Vector{Float64}; du::Vector{Float64}           # the diagonals of A
    wl::Vector{Float64}; wd::Vector{Float64}; wu::Vector{Float64}           # A + σI, σ real
    zl::Vector{ComplexF64}; zd::Vector{ComplexF64}; zu::Vector{ComplexF64}  # A + σI, σ complex
    Z::Matrix{ComplexF64}; Y::Matrix{ComplexF64}                           # n × width
    T::Matrix{ComplexF64}; Q::Matrix{ComplexF64}; λ::Vector{ComplexF64}    # K-step: M = Q T Qᴴ
    zwork::Vector{ComplexF64}; rwork::Vector{Float64}
end

"""BandedLU(A; width): workspace for K-steps of rank ≤ width and blocks of ≤ width columns."""
function BandedLU(A::SparseMatrixCSC; width::Int)
    n = size(A, 1); dl, d, du = Vector(diag(A, -1)), Vector(diag(A)), Vector(diag(A, 1))
    cmat(m, k) = Matrix{ComplexF64}(undef, m, k)
    BandedLU(dl, d, du, similar(dl), similar(d), similar(du), complex(dl), complex(d), complex(du),
             cmat(n, width), cmat(n, width), cmat(width, width), cmat(width, width), zeros(ComplexF64, width),
             zeros(ComplexF64, zgees_lwork(width)), zeros(width))
end

"""(l, d, u) ← the diagonals of A + σI.  LAPACK's gtsv overwrites them with its LU factors,
so they are reloaded before every solve."""
function load_shifted!(l, d, u, s::BandedLU, σ)
    copyto!(l, s.dl); d .= s.d .+ σ; copyto!(u, s.du)
    return nothing
end

"""
K-step A K + K Mᵀ = R for a general M.  With the complex Schur form M = Q T Qᴴ (T upper
triangular), K̃ = K Q̄ solves A K̃ + K̃ Tᵀ = R̃ = R Q̄, whose column j reads
(A + Tⱼⱼ I) k̃ⱼ = r̃ⱼ − Σ_{i>j} Tⱼᵢ k̃ᵢ: solved for j = r, ..., 1.  Then K = K̃ Qᵀ (real).
"""
@views function kstep!(s::BandedLU, K, R, M)
    r = size(M, 1)
    T = s.T[1:r, 1:r]; Q = s.Q[1:r, 1:r]
    copyto!(T, M)
    zgees!(T, s.λ[1:r], Q, s.zwork, s.rwork[1:r])         # M = Q T Qᴴ
    Q .= conj.(Q)                                                       # Q ← Q̄
    Y = s.Y[:, 1:r]; Z = s.Z[:, 1:r]
    copyto!(Y, R); mul!(Z, Y, Q)                                        # R̃ = R Q̄
    for j in r:-1:1
        z = Z[:, j]
        j < r && mul!(z, Z[:, j+1:r], T[j, j+1:r], -1, 1)  # r̃ⱼ − Σ_{i>j} Tⱼᵢ k̃ᵢ
        load_shifted!(s.zl, s.zd, s.zu, s, T[j, j])
        LAPACK.gtsv!(s.zl, s.zd, s.zu, z)                               # k̃ⱼ
    end
    mul!(Y, Z, Q')                                                      # K = K̃ Q̄ᴴ = K̃ Qᵀ
    K .= real.(Y)
    return K
end

"""X = (A − ξI)⁻¹ W, ξ real: one factorisation for all columns."""
function shifted_solve!(s::BandedLU, X, W, ξ::Float64)
    copyto!(X, W)
    load_shifted!(s.wl, s.wd, s.wu, s, -ξ)
    return LAPACK.gtsv!(s.wl, s.wd, s.wu, X)
end

"""Y = [Re y  Im y], y = (A − ξI)⁻¹ W, ξ complex: one complex factorisation for all columns."""
@views function pair_solve!(s::BandedLU, Y, W, ξ::ComplexF64)
    w = size(W, 2); Z = s.Z[:, 1:w]
    copyto!(Z, W)
    load_shifted!(s.zl, s.zd, s.zu, s, -ξ)
    LAPACK.gtsv!(s.zl, s.zd, s.zu, Z)
    Y[:, 1:w] .= real.(Z); Y[:, w+1:2w] .= imag.(Z)
    return Y
end

# -- GMRES (Krylov.jl) for shifted and pair solves with a dense A -------------

"""The linear map x ↦ vec(A X + X Mᵀ), X = the n × w matrix with vec(X) = x (the form
Krylov.jl needs: a vector in, a vector out)."""
struct SylvesterOperator{TA}
    A::TA; M::Matrix{Float64}
    X::Matrix{Float64}; Y::Matrix{Float64}            # n × w buffers
end
SylvesterOperator(A, w) = SylvesterOperator(A, zeros(w, w), zeros(size(A, 1), w), zeros(size(A, 1), w))
Base.size(op::SylvesterOperator) = (length(op.X), length(op.X))
Base.eltype(::SylvesterOperator) = Float64
function LinearAlgebra.mul!(y::AbstractVector, op::SylvesterOperator, x::AbstractVector)
    copyto!(op.X, x)
    mul!(op.Y, op.A, op.X)
    mul!(op.Y, op.X, op.M', 1.0, 1.0)
    return copyto!(y, op.Y)
end

struct GMRES{TA,WS}
    shifted::SylvesterOperator{TA}; pair::SylvesterOperator{TA}  # b and 2b columns
    ws_shifted::WS; ws_pair::WS                                  # Krylov.jl workspaces
    rhs_shifted::Vector{Float64}; rhs_pair::Vector{Float64}
    rtol::Float64; itmax::Int
end

"""
    GMRES(A; b, rtol, memory = 40, itmax = 2000)

Shifted and pair solves with A for blocks of up to b columns: restarted GMRES(memory) from
Krylov.jl on the whole block as one vector, stopping at relative residual rtol.
"""
function GMRES(A; b::Int, rtol::Float64, memory::Int = 40, itmax::Int = 2000)
    N = size(A, 1) * b
    g = GMRES(SylvesterOperator(A, b), SylvesterOperator(A, 2b),
              GmresWorkspace(N, N, Vector{Float64}; memory), GmresWorkspace(2N, 2N, Vector{Float64}; memory),
              zeros(N), zeros(2N), rtol, itmax)
    # Krylov.jl creates one more vector on the first restarted solve; a solve with a zero
    # right-hand side (it returns at once) does that here, in the setup.
    run_gmres!(g, g.ws_shifted, g.shifted, g.rhs_shifted); run_gmres!(g, g.ws_pair, g.pair, g.rhs_pair)
    return g
end

"""Solve op(x) = rhs from x = 0 (restart from the true residual, re-orthogonalised basis)."""
run_gmres!(g::GMRES, ws, op, rhs) =
    gmres!(ws, op, rhs; restart = true, reorthogonalization = true, atol = 0.0, rtol = g.rtol, itmax = g.itmax)

"""X = (A − ξI)⁻¹ W, ξ real:  A X + X Mᵀ = W with M = −ξ I.  A block narrower than b is
padded with zero columns, whose solution is zero."""
function shifted_solve!(g::GMRES, X, W, ξ::Float64)
    op = g.shifted
    fill!(op.M, 0.0)
    for i in axes(op.M, 1)
        op.M[i, i] = -ξ
    end
    fill!(g.rhs_shifted, 0.0); copyto!(g.rhs_shifted, 1, W, 1, length(W))
    run_gmres!(g, g.ws_shifted, op, g.rhs_shifted)
    copyto!(X, 1, solution(g.ws_shifted), 1, length(X))
    return X
end

"""Y = [Re y  Im y], y = (A − ξI)⁻¹ W:  A Z + Z Mᵀ = [W 0] for Z = [Re y  Im y] with
M = [aI −bI; bI aI], a + ib = −ξ (b × b blocks; a narrower W is zero-padded)."""
@views function pair_solve!(g::GMRES, Y, W, ξ::ComplexF64)
    op = g.pair; b = size(op.M, 1) ÷ 2; w = size(W, 2)
    a, c = -real(ξ), -imag(ξ)
    fill!(op.M, 0.0)
    for i in 1:b
        op.M[i, i] = a; op.M[b+i, b+i] = a; op.M[i, b+i] = -c; op.M[b+i, i] = c
    end
    fill!(g.rhs_pair, 0.0); copyto!(g.rhs_pair, 1, W, 1, length(W))     # [W 0]
    run_gmres!(g, g.ws_pair, op, g.rhs_pair)
    Z = op.X; copyto!(Z, solution(g.ws_pair))                            # n × 2b
    copyto!(Y[:, 1:w], Z[:, 1:w]); copyto!(Y[:, w+1:2w], Z[:, b+1:b+w])
    return Y
end

# -- block Krylov K-step (BUG, dense A) --------------------------------------

mutable struct BlockKrylovKStep{TA}
    basis::KrylovBasis{TA}
    C::Matrix{Float64}                                # starting block (the RHS factor)
    rtol::Float64
    built::Bool                                       # basis built in this solve?
    sylv::SmallSylvester
    Ch::Matrix{Float64}; Y::Matrix{Float64}; E::Matrix{Float64}; c::Matrix{Float64}
end

"""
    BlockKrylovKStep(A, C; rmax, rtol, Kmax = min(n, 64·size(C, 2)))

K-steps A K + K Mᵀ = R with R ∈ span(C): Galerkin on Q = basis of K_m(A, C),
(QᵀAQ) Y + Y Mᵀ = QᵀR, K = Q Y.  The residual of that solution is ‖Γ Y‖_F
((I − QQᵀ) A Q = Z Γ: only the newest block sticks out), and the basis grows by one block
while it exceeds rtol ‖R‖_F.  Rebuilt at the start of every BUG solve (`reset!`).
"""
function BlockKrylovKStep(A, C; rmax::Int, rtol::Float64, Kmax::Int = min(size(A, 1), 64size(C, 2)))
    b = size(C, 2)
    BlockKrylovKStep(KrylovBasis(A, b, Kmax; deflation = :relative), Matrix{Float64}(C), rtol, false,
                     SmallSylvester(Kmax, rmax), zeros(Kmax, rmax), zeros(Kmax, rmax), zeros(2b, rmax),
                     zeros(Kmax, 2rmax))
end

reset!(s::BlockKrylovKStep) = (s.built = false; s)
reset!(s) = s

@views function kstep!(s::BlockKrylovKStep, K, R, M)
    kb = s.basis; r = size(R, 2)
    if !s.built
        start!(kb, s.C); s.built = true
    end
    nR = norm(R)
    while true
        Q = basis(kb); H = kb.H[1:kb.K, 1:kb.K]
        Ch = s.Ch[1:kb.K, 1:r]; mul!(Ch, Q', R)
        Y = s.Y[1:kb.K, 1:r]
        solve_sylvester!(s.sylv, Y, H, M, Ch)          # (QᵀAQ) Y + Y Mᵀ = QᵀR
        Γ = stickout!(kb, newest_block(kb))
        E = s.E[1:size(Γ, 1), 1:r]; mul!(E, Γ, Y)
        norm(E) ≤ s.rtol * nR && break
        J = newest_block(kb)                            # extend: A · (newest block)
        W = kb.W[:, 1:length(J)]; copyto!(W, kb.AU[:, J])
        add_block!(kb, W) == 0 && break                 # invariant subspace or no room
    end
    mul!(K, basis(kb), s.Y[1:kb.K, 1:r])
    return K
end

# BUG products A·X for a basis X of the BUG iteration.  With the block-Krylov K-step every BUG
# basis lies in span(Q), so A X = (A Q)(QᵀX) from the stored A·Q: O(n K w) instead of O(n² w).
apply!(s, Y, A, X) = mul!(Y, A, X)
@views function apply!(s::BlockKrylovKStep, Y, A, X)
    s.built || return mul!(Y, A, X)                   # before the first K-step of the solve
    kb = s.basis
    c = s.c[1:kb.K, 1:size(X, 2)]
    mul!(c, basis(kb)', X)
    return mul!(Y, kb.AU[:, 1:kb.K], c)
end

# Keep an augmented basis X inside span(Q), where `apply!` is exact: X ← orth(Q QᵀX).  (Near
# convergence orth[K U] turns rounding noise into basis vectors outside span(Q).)  Returns the width.
restrict_to_basis!(s, X, tau, work) = size(X, 2)
@views function restrict_to_basis!(s::BlockKrylovKStep, X, tau, work)
    s.built || return size(X, 2)
    kb = s.basis; Q = basis(kb); w = size(X, 2)
    if w ≥ kb.K                                       # X would span all of span(Q): take Q itself
        copyto!(X[:, 1:kb.K], Q)
        return kb.K
    end
    c = s.c[1:kb.K, 1:w]
    mul!(c, Q', X); mul!(X, Q, c)
    orthonormalize!(X, tau[1:w], work)
    return w
end
