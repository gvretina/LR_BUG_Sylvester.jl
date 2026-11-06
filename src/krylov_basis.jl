# Block Krylov basis for one operator A, shared by RKSM, TBRK and the block-Krylov K-step of BUG.
#
# Kept alongside the orthonormal basis U = [U₁ U₂ ... ] (blocks of columns):
#   AU = A·U            (one product per new column, never recomputed)
#   H  = Uᵀ A U         (the projected operator, grown block by block)
#   UC = Uᵀ C           (C = the starting block; only the first block is nonzero)
#
# New blocks are orthogonalised against U by block Gram-Schmidt (repeated while a pass removes
# more than half of the block), then QR + SVD of the small R factor orders their directions.
# Which directions are kept (`deflation`):
#   :none     keep all                                (Casulli-Robol's RKSM: deflation_tol = 0)
#   :eps      σ > eps · max(σ₁, ‖coefficients‖)       (rktoolbox default, used by TBRK)
#   :relative σ > 1e-10 · ‖W‖                         (block-Krylov K-step of BUG)

mutable struct KrylovBasis{TA}
    A::TA
    n::Int; Kmax::Int; b::Int                         # b = width of the starting block
    U::Matrix{Float64}; AU::Matrix{Float64}; H::Matrix{Float64}; UC::Matrix{Float64}
    K::Int                                            # current dimension
    blocks::Vector{UnitRange{Int}}                    # column ranges of the blocks
    deflation::Symbol
    W::Matrix{Float64}; Z::Matrix{Float64}            # n × 2b block scratch
    coef::Matrix{Float64}; Γ::Matrix{Float64}         # Kmax × 2b, 2b × Kmax
    Rb::Matrix{Float64}; σ::Vector{Float64}; P::Matrix{Float64}; VT::Matrix{Float64}
    tau::Vector{Float64}; rd::Vector{Float64}
    qrwork::Vector{Float64}; svdwork::Vector{Float64}; iwork::Vector{BlasInt}
end

function KrylovBasis(A, b::Int, Kmax::Int; deflation::Symbol)
    n = size(A, 1); w = 2b
    KrylovBasis(A, n, Kmax, b, scratch(n, Kmax), scratch(n, Kmax), scratch(Kmax, Kmax), zeros(Kmax, b), 0,
                sizehint!(UnitRange{Int}[], Kmax), deflation, scratch(n, w), scratch(n, w),
                zeros(Kmax, w), zeros(w, Kmax), zeros(w, w), zeros(w), zeros(w, w), zeros(w, w),
                zeros(w), zeros(w), zeros(qr_lwork(n, w)), zeros(svd_lwork(w, w)), zeros(BlasInt, 8w))
end

@views basis(kb::KrylovBasis) = kb.U[:, 1:kb.K]
newest_block(kb::KrylovBasis) = kb.blocks[end]

"""Start the basis from the columns of C (n × b): U₁ = orth(C), UC = UᵀC."""
@views function start!(kb::KrylovBasis, C)
    kb.K = 0; empty!(kb.blocks)
    W = kb.W[:, 1:kb.b]; copyto!(W, C)
    add_block!(kb, W)
    J = kb.blocks[1]
    fill!(kb.UC, 0.0)
    mul!(kb.UC[J, :], kb.U[:, J]', C)
    return kb
end

"""
    add_block!(kb, W) -> number of columns added

Orthogonalise W (n × w, destroyed) against the basis and append its kept directions,
with their images A·u and the new rows/columns of H.
"""
@views function add_block!(kb::KrylovBasis, W)
    K = kb.K; w = size(W, 2)
    room = kb.Kmax - K
    room ≤ 0 && return 0
    nrm0 = norm(W)
    U = kb.U[:, 1:K]; c = kb.coef[1:K, 1:w]
    if K > 0                                          # block Gram-Schmidt, repeated if needed
        nprev = nrm0
        for pass in 1:4
            mul!(c, U', W); mul!(W, U, c, -1.0, 1.0)
            nn = norm(W)
            (pass ≥ 2 && nn > 0.5nprev) && break
            nprev = nn
        end
    end
    hnrm = sqrt(max(nrm0^2 - norm(W)^2, 0.0))        # norm of the removed coefficients
    # W = Q R, R = P Σ Vᵀ  ⇒  the directions Q·P[:, j] in decreasing order of importance σⱼ
    tau = kb.tau[1:w]
    geqrf!(W, tau, kb.qrwork)
    R = kb.Rb[1:w, 1:w]; fill!(R, 0.0)
    for j in 1:w, i in 1:j
        R[i, j] = W[i, j]
    end
    orgqr!(W, tau, kb.qrwork)
    σ = kb.σ[1:w]; P = kb.P[1:w, 1:w]
    gesdd!(R, σ, P, kb.VT[1:w, 1:w], kb.svdwork, kb.iwork)
    keep = kb.deflation === :none ? w :
           kb.deflation === :eps  ? count(>(eps() * max(σ[1], hnrm)), σ) :
                                    count(>(1e-10 * nrm0), σ)
    keep = min(keep, room)
    keep == 0 && return 0
    new = K+1:K+keep
    Unew = kb.U[:, new]
    mul!(Unew, W, P[:, 1:keep])
    if K > 0          # weak directions carry U-components ~ eps/σ after scaling: orthogonalise again
        c = kb.coef[1:K, 1:keep]
        for _ in 1:2
            mul!(c, U', Unew); mul!(Unew, U, c, -1.0, 1.0)
        end
        orthonormalize!(Unew, kb.tau[1:keep], kb.qrwork)
    end
    mul!(kb.AU[:, new], kb.A, Unew)
    Kn = K + keep
    mul!(kb.H[1:Kn, new], kb.U[:, 1:Kn]', kb.AU[:, new])
    K > 0 && mul!(kb.H[new, 1:K], Unew', kb.AU[:, 1:K])
    kb.K = Kn
    push!(kb.blocks, new)
    return keep
end

"""
    stickout!(kb, J) -> Γ

The part of A·U outside span(U).  Only the columns J have images leaving the space
(RKSM / TBRK: the starting block; polynomial Krylov: the newest block), so
(I − UUᵀ) A U = Z Γ with Z = orth((I − UUᵀ) A U[:, J]) and Γ = Zᵀ A U  (|J| × K).
The Galerkin residual of a projected solution Y is then ‖Γ Y‖ per side.
"""
@views function stickout!(kb::KrylovBasis, J::UnitRange{Int})
    K = kb.K; w = length(J)
    Γ = kb.Γ[1:w, 1:K]
    w == 0 && return Γ
    Z = kb.Z[:, 1:w]
    copyto!(Z, kb.AU[:, J])
    nrm0 = norm(Z)
    U = kb.U[:, 1:K]; c = kb.coef[1:K, 1:w]
    for _ in 1:2
        mul!(c, U', Z); mul!(Z, U, c, -1.0, 1.0)
    end
    geqrf!(Z, kb.tau[1:w], kb.qrwork)
    rd = kb.rd[1:w]
    for i in 1:w
        rd[i] = abs(Z[i, i])
    end
    orgqr!(Z, kb.tau[1:w], kb.qrwork)
    mul!(Γ, Z', kb.AU[:, 1:K])
    for i in 1:w                                      # rounding-noise directions: no stick-out
        rd[i] ≤ 1e-12 * nrm0 && fill!(Γ[i, :], 0.0)
    end
    return Γ
end
