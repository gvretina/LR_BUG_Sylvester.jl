# Relative residuals ‖𝓛(Y) − B‖_F / ‖B‖_F of a low-rank approximation Y, computed the same way
# for every method, without forming Y.  QR based: accurate down to ~eps (a Gram-matrix
# formula would stop at ~√eps ≈ 1e-8).
#
# Matrix: Y = U S Vᵀ.
#   A Y + Y Bᵀ − Cu Cs Cvᵀ = L · Mid · Rᵀ,   L = [AU  U  Cu],  R = [V  BV  Cv],
#   Mid = blkdiag(S, S, −Cs);  with L = Q_L R_L and R = Q_R R_R,  ‖·‖_F = ‖R_L Mid R_Rᵀ‖_F.
#   (R_L has min(n, w) rows: when the factors are wider than tall, w > n, it is n × w.)
#
# Tucker (N modes): Y = G ×ᵢ Uᵢ.
#   Σᵢ Y ×ᵢ Aᵢ − Gb ×ᵢ Ubᵢ = Ĉ ×ᵢ [Uᵢ  AᵢUᵢ  Ubᵢ],  Ĉ = G in the N blocks with block index 2 in
#   one mode and 1 in the others (for N = 3: (2,1,1), (1,2,1), (1,1,2)), −Gb in block (3, ..., 3);
#   with [Uᵢ AᵢUᵢ Ubᵢ] = Qᵢ Rᵢ,  ‖·‖_F = ‖Ĉ ×ᵢ Rᵢ‖_F  (Rᵢ has min(nᵢ, wᵢ) rows).

struct MatrixResidual
    L::Matrix{Float64}; R::Matrix{Float64}            # n × w, m × w   (w = 2rmax + rc)
    RL::Matrix{Float64}; RR::Matrix{Float64}; Mid::Matrix{Float64}; P::Matrix{Float64}
    tau::Vector{Float64}; work::Vector{Float64}
end

function MatrixResidual(n, m, rmax, rc)
    w = 2rmax + rc
    MatrixResidual(scratch(n, w), scratch(m, w), zeros(w, w), zeros(w, w), zeros(w, w), zeros(w, w),
                   zeros(w), zeros(max(qr_lwork(max(n, w), w), qr_lwork(max(m, w), w))))
end

"""‖A U S Vᵀ + U S (B V)ᵀ − Cu Cs Cvᵀ‖_F / ‖Cs‖_F, given AU = A·U and BV = B·V."""
@views function relative_residual!(ws::MatrixResidual, U, S, V, AU, BV, Cu, Cs, Cv)
    r = size(U, 2); rc = size(Cs, 1); w = 2r + rc
    L = ws.L[:, 1:w]; R = ws.R[:, 1:w]
    copyto!(L[:, 1:r], AU); copyto!(L[:, r+1:2r], U); copyto!(L[:, 2r+1:w], Cu)
    copyto!(R[:, 1:r], V);  copyto!(R[:, r+1:2r], BV); copyto!(R[:, 2r+1:w], Cv)
    mL, mR = min(size(L, 1), w), min(size(R, 1), w)
    RL = ws.RL[1:mL, 1:w]; RR = ws.RR[1:mR, 1:w]
    rfactor!(RL, L, ws.tau[1:mL], ws.work)
    rfactor!(RR, R, ws.tau[1:mR], ws.work)
    Mid = ws.Mid[1:w, 1:w]; fill!(Mid, 0.0)
    copyto!(Mid[1:r, 1:r], S); copyto!(Mid[r+1:2r, r+1:2r], S)
    for j in 1:rc, i in 1:rc
        Mid[2r+i, 2r+j] = -Cs[i, j]
    end
    P = ws.P[1:mL, 1:w]; E = ws.Mid[1:mL, 1:mR]
    mul!(P, RL, Mid); mul!(E, P, RR')
    return norm(E) / norm(Cs)                       # Cu, Cv orthonormal: ‖C‖_F = ‖Cs‖_F
end

struct TuckerResidual{N}
    V::NTuple{N,Matrix{Float64}}                      # nᵢ × wᵢ   (wᵢ = 2rmax + rb)
    R::Matrix{Float64}; C1::Vector{Float64}; C2::Vector{Float64}
    tau::Vector{Float64}; work::Vector{Float64}
end

function TuckerResidual(n::NTuple{N,Int}, rmax, rb) where {N}
    w = 2rmax + rb
    TuckerResidual(ntuple(k -> scratch(n[k], w), N), zeros(w, w), scratch(w^N), scratch(w^N),
                   zeros(w), zeros(maximum(qr_lwork(max(n[k], w), w) for k in 1:N)))
end

"""
‖Σᵢ Y ×ᵢ Aᵢ − Gb ×ᵢ Ubᵢ‖_F / ‖Gb‖_F for Y = G ×ᵢ Uᵢ (G flat with dims d, Gb flat with dims db),
given AU[i] = Aᵢ·Uᵢ.  U and AU are tuples of n × d[i] matrices (views).
"""
@views function relative_residual!(ws::TuckerResidual{N}, U, AU, G, d::NTuple{N,Int}, Ub, Gb, db::NTuple{N,Int}) where {N}
    w = ntuple(m -> 2d[m] + db[m], Val(N))
    st = ntuple(m -> prod(q -> w[q], 1:m-1; init = 1), Val(N))       # linear strides of Ĉ
    C = ws.C1
    fill!(C[1:prod(w)], 0.0)
    @inbounds for i in 1:N, (L, I) in enumerate(CartesianIndices(d))      # G in block i ↦ AᵢUᵢ
        C[1 + sum(ntuple(m -> (I[m] - 1 + (m == i ? d[m] : 0)) * st[m], Val(N)))] = G[L]
    end
    @inbounds for (L, I) in enumerate(CartesianIndices(db))                  # −Gb in block (3, ..., 3)
        C[1 + sum(ntuple(m -> (I[m] - 1 + 2d[m]) * st[m], Val(N)))] = -Gb[L]
    end
    cur, nxt = ws.C1, ws.C2; cd = w
    for k in 1:N
        V = ws.V[k][:, 1:w[k]]
        copyto!(V[:, 1:d[k]], U[k])
        copyto!(V[:, d[k]+1:2d[k]], AU[k])
        copyto!(V[:, 2d[k]+1:w[k]], Ub[k])
        mk = min(size(V, 1), w[k])                    # Rₖ is mk × wₖ
        Rk = ws.R[1:mk, 1:w[k]]
        rfactor!(Rk, V, ws.tau[1:mk], ws.work)
        cd = n_mode_product!(nxt, cur, cd, Rk, k)
        cur, nxt = nxt, cur
    end
    return sqrt(sumsq(cur, prod(cd)) / sumsq(Gb, prod(db)))
end
