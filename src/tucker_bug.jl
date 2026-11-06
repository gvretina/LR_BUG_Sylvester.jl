# Rank-adaptive Tucker BUG for the tensor Sylvester equation with N modes (the paper: N = 3)
#     ∑ᵢ₌₁ᴺ X ×ᵢ Aᵢ = B = Gb ×₁ Ub₁ ⋯ ×_N Ub_N.
#
# One iteration, from the current Tucker factors X = G ×ᵢ Uᵢ (Uᵢ: n × rᵢ):
#   K-steps   for each mode k, all from the OLD factors:
#               Qₖ = orth(Mat_k(G)ᵀ)                 (∏_{j≠k} rⱼ × rₖ)
#               Mₖ = Qₖᵀ (Σ_{j≠k} I ⊗ ⋯ ⊗ Mⱼ ⊗ ⋯ ⊗ I) Qₖ,   Mⱼ = UⱼᵀAⱼUⱼ
#               Rₖ = Ubₖ · Mat_k(Gb ×_{j≠k} UⱼᵀUbⱼ) · Qₖ
#               Aₖ Kₖ + Kₖ Mₖᵀ = Rₖ                  (inner solver: matrix K-step)
#   augment   Ûₖ = orth[Kₖ Uₖ]
#   core      Σᵢ Ĝ ×ᵢ M̂ᵢ = Gb ×ᵢ ÛᵢᵀUbᵢ,  M̂ᵢ = ÛᵢᵀAᵢÛᵢ   (CoreSolver)
#   truncate  HOSVD of Ĝ, per mode the rank of truncation_rank(σ of Mat_k(Ĝ)):
#               G = Ĝ ×ᵢ Pᵢᵀ,  Uᵢ = Ûᵢ Pᵢ
# Start: Uᵢ = Ubᵢ.  Stop: relative residual ≤ tol, or stagnation (truncation.jl).
# Small tensors live in flat buffers with explicit dims (tensors.jl).

mutable struct TuckerBUG{TA,I,N}
    As::NTuple{N,TA}; inners::NTuple{N,I}
    rmax::Int; r::NTuple{N,Int}
    U::NTuple{N,Matrix{Float64}}; AU::NTuple{N,Matrix{Float64}}     # n × rmax
    Uh::NTuple{N,Matrix{Float64}}; AUh::NTuple{N,Matrix{Float64}}   # n × 2rmax
    RK::NTuple{N,Matrix{Float64}}                                   # K-step right-hand sides
    M::NTuple{N,Matrix{Float64}}; Mh::NTuple{N,Matrix{Float64}}    # UᵀAU, ÛᵀAÛ
    UUb::NTuple{N,Matrix{Float64}}                                  # UᵀUb or ÛᵀUb
    G::Vector{Float64}; Gh::Vector{Float64}; T1::Vector{Float64}; T2::Vector{Float64}; gb::Vector{Float64}
    Q::Vector{Float64}; SQ::Vector{Float64}; Mk::Matrix{Float64}    # K-step projection (Q, SQ flat)
    Wk::Matrix{Float64}; WQ::Matrix{Float64}
    P::NTuple{N,Matrix{Float64}}; hosvd::HOSVDWork
    core::CoreSolver{N}
    tau::Vector{Float64}; qrwork::Vector{Float64}
    residual::TuckerResidual{N}
    res_hist::Vector{Float64}; rank_hist::Vector{NTuple{N,Int}}
end

"""
    TuckerBUG(p::TuckerProblem, inners; rmax = 64, maxit = 50, general)

`general = true` for non-symmetric operators (complex-Schur core solve).
"""
function TuckerBUG(p::TuckerProblem{TA,N}, inners; rmax::Int = 64, maxit::Int = 50, general::Bool) where {TA,N}
    n = ntuple(k -> size(p.As[k], 1), N); rb = size(p.Gb)
    rmax = min(rmax, minimum(n) ÷ 2); R2 = 2rmax; rbm = maximum(rb)
    big = max(R2, rbm)^N                             # largest intermediate of a chain of mode products
    TuckerBUG(p.As, Tuple(inners), rmax, ntuple(_ -> 0, N),
              ntuple(k -> scratch(n[k], rmax), N), ntuple(k -> scratch(n[k], rmax), N),
              ntuple(k -> scratch(n[k], R2), N), ntuple(k -> scratch(n[k], R2), N),
              ntuple(k -> scratch(n[k], rmax), N),
              ntuple(_ -> zeros(rmax, rmax), N), ntuple(_ -> zeros(R2, R2), N), ntuple(_ -> zeros(R2, rbm), N),
              scratch(rmax^N), scratch(big), scratch(big), scratch(big), zeros(prod(rb)),
              scratch(rmax^N), scratch(rmax^N), zeros(rmax, rmax),
              scratch(rbm, rmax^(N-1)), zeros(rbm, rmax),
              ntuple(_ -> zeros(R2, R2), N), HOSVDWork(R2, N), CoreSolver(R2, N; general),
              zeros(max(R2, rmax)), zeros(maximum(qr_lwork(max(n[k], rmax^(N-1)), R2) for k in 1:N)),
              TuckerResidual(n, rmax, rbm), zeros(maxit), fill(ntuple(_ -> 0, N), maxit))
end

"""
    solve!(t, p; tol, theta) -> (; status, iters, residual, ranks)

Solution: `tucker_factors(t)` = (U, G) with X ≈ G ×ᵢ Uᵢ.
"""
@views function solve!(t::TuckerBUG{TA,I,N}, p::TuckerProblem; tol::Float64, theta::Float64,
                       maxit::Int = length(t.res_hist)) where {TA,I,N}
    for k in 1:N
        reset!(t.inners[k])
    end
    db = size(p.Gb); copyto!(t.gb, p.Gb)
    t.r = size(p.Gb)
    for k in 1:N                                      # start: Uₖ = Ubₖ
        rk = t.r[k]
        U = t.U[k][:, 1:rk]; AU = t.AU[k][:, 1:rk]
        copyto!(U, p.Ub[k]); mul!(AU, t.As[k], U); mul!(t.M[k][1:rk, 1:rk], U', AU)
    end
    project_rhs!(t, t.G, t.U, t.r, p.Ub, db)          # G = Gb ×ᵢ UᵢᵀUbᵢ
    status = :maxiter; res = Inf; it = 0
    for k in 1:min(maxit, length(t.res_hist))
        it = k
        tucker_iteration!(t, p, db, theta)
        d = t.r
        res = relative_residual!(t.residual, factor_views(t.U, d), factor_views(t.AU, d), t.G, d, p.Ub, t.gb, db)
        t.res_hist[k] = res; t.rank_hist[k] = d
        if res ≤ tol
            status = :converged; break
        end
        if k > 1 && stagnated(res, t.res_hist[k-1])
            status = :stagnated; break
        end
    end
    return (; status, iters = it, residual = res, ranks = t.r)
end

@views factor_views(U::NTuple{N}, d) where {N} = ntuple(k -> U[k][:, 1:d[k]], Val(N))
@views tucker_factors(t::TuckerBUG) = (factor_views(t.U, t.r), reshape(t.G[1:prod(t.r)], t.r))

"""UUbₖ = Basesₖ[:, 1:dₖ]ᵀ Ubₖ for every mode; returns them as views."""
@views function project_bases!(t::TuckerBUG{TA,I,N}, Bases, d, Ub, db) where {TA,I,N}
    UUb = ntuple(k -> t.UUb[k][1:d[k], 1:db[k]], Val(N))
    for k in 1:N
        mul!(UUb[k], Bases[k][:, 1:d[k]]', Ub[k])
    end
    return UUb
end

"""dest = Gb ×ᵢ (Basesᵢ[:, 1:dᵢ]ᵀ Ubᵢ)."""
project_rhs!(t::TuckerBUG, dest, Bases, d, Ub, db) =
    n_mode_products!(dest, t.gb, db, project_bases!(t, Bases, d, Ub, db), (t.T1, t.T2))

@views function tucker_iteration!(t::TuckerBUG{TA,I,N}, p::TuckerProblem, db, θ) where {TA,I,N}
    d = t.r

    # K-steps (all from the old factors)
    UUb = project_bases!(t, t.U, d, p.Ub, db)         # UⱼᵀUbⱼ
    for k in 1:N
        rk = d[k]; m = prod(d) ÷ rk; others = nonmodes(d, k)
        Q = flatmat(t.Q, m, rk)
        matricize!(Q, t.G, d, k; transposed = true)
        orthonormalize!(Q, t.tau[1:rk], t.qrwork)           # Qₖ = orth(Mat_k(G)ᵀ)
        # SQ = Σ_{j≠k} (I ⊗ ⋯ ⊗ Mⱼ ⊗ ⋯ ⊗ I) Qₖ: the rows of Qₖ run over the other modes (lower
        # fastest), so Qₖ is the flat tensor of dims (rⱼ for j ≠ k ..., rₖ) and each term is a mode product
        dq = ntuple(q -> q < N ? d[others[q]] : rk, Val(N))
        for (q, j) in enumerate(others)
            n_mode_product!(t.SQ, t.Q, dq, t.M[j][1:d[j], 1:d[j]], q; β = q > 1)
        end
        Mk = t.Mk[1:rk, 1:rk]; mul!(Mk, Q', flatmat(t.SQ, m, rk))
        # Rₖ = Ubₖ · Mat_k(Gb ×_{j≠k} UⱼᵀUbⱼ) · Qₖ
        c = n_mode_products!(t.Gh, t.gb, db, UUb, (t.T1, t.T2); skip = k)
        Wk = t.Wk[1:db[k], 1:m]; matricize!(Wk, t.Gh, c, k)
        WQ = t.WQ[1:db[k], 1:rk]; mul!(WQ, Wk, Q)
        Rk = t.RK[k][:, 1:rk]; mul!(Rk, p.Ub[k], WQ)
        kstep!(t.inners[k], t.Uh[k][:, 1:rk], Rk, Mk)        # Kₖ → Uh[k][:, 1:rk]
    end

    # augmentation Ûₖ = orth[Kₖ Uₖ], projected operators M̂ₖ = ÛₖᵀAₖÛₖ
    dh = map(x -> 2x, d)
    for k in 1:N
        copyto!(t.Uh[k][:, d[k]+1:dh[k]], t.U[k][:, 1:d[k]])
        Uh = t.Uh[k][:, 1:dh[k]]; AUh = t.AUh[k][:, 1:dh[k]]
        orthonormalize!(Uh, t.tau[1:dh[k]], t.qrwork)
        mul!(AUh, t.As[k], Uh)
        mul!(t.Mh[k][1:dh[k], 1:dh[k]], Uh', AUh)
    end

    # core:  Σᵢ Ĝ ×ᵢ M̂ᵢ = Gb ×ᵢ ÛᵢᵀUbᵢ
    project_rhs!(t, t.Gh, t.Uh, dh, p.Ub, db)
    solve_core!(t.core, ntuple(k -> t.Mh[k][1:dh[k], 1:dh[k]], Val(N)), dh, t.Gh, t.Gh)

    # truncation: HOSVD of Ĝ,  G = Ĝ ×ᵢ Pᵢᵀ
    rn = ntuple(k -> min(mode_svd!(t.hosvd, t.P[k][1:dh[k], 1:dh[k]], t.Gh, dh, k, θ), t.rmax), Val(N))
    n_mode_products!(t.G, t.Gh, dh, ntuple(k -> t.P[k][1:dh[k], 1:rn[k]]', Val(N)), (t.T1, t.T2))
    for k in 1:N
        Pk = t.P[k][1:dh[k], 1:rn[k]]
        U = t.U[k][:, 1:rn[k]]; AU = t.AU[k][:, 1:rn[k]]
        mul!(U, t.Uh[k][:, 1:dh[k]], Pk)
        mul!(AU, t.AUh[k][:, 1:dh[k]], Pk)
        mul!(t.M[k][1:rn[k], 1:rn[k]], U', AU)
    end
    t.r = rn
    return nothing
end
