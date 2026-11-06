# TBRK: tensorized block rational Krylov method for the tensor Sylvester equation with N modes
# (the paper: N = 3)
#     ∑ᵢ₌₁ᴺ X ×ᵢ Aᵢ = Gb ×₁ Ub₁ ⋯ ×_N Ub_N.
# Port of A. A. Casulli's `Tuck_Sylvester_adaptive.m` (github.com/numpi/TBRK-Sylvester,
# "Tensorized block rational Krylov methods for tensor Sylvester equations", arXiv:2306.00705),
# options as in their experiments: ADM poles, real arithmetic, residual checked every iteration.
#
#   spaces    one block rational Krylov space Vᵢ per mode, started from Ubᵢ; every iteration
#             adds (Aᵢ − ξᵢI)⁻¹ u to each (u = first b columns of the newest block; a complex
#             pole adds the real pair block [Re y  Im y]); blocks are deflated as rktoolbox
#             does by default (`:eps`, krylov_basis.jl)
#   Galerkin  Σᵢ Y ×ᵢ Hᵢ = Gb ×ᵢ VᵢᵀUbᵢ,  Hᵢ = VᵢᵀAᵢVᵢ,   X ≈ Y ×ᵢ Vᵢ
#   poles     (their `poles_selection_ADM` / `newpole_adaptive_ADM`) for mode i the candidates
#             lie in −spec(⊕_{j≠i} Aⱼ) ⊂ [−Mᵢ, −mᵢ] (mᵢ, Mᵢ = sums of the other modes' extreme
#             eigenvalues): 20 points between consecutive sorted nodes {−mᵢ, −Mᵢ, −(sums of one
#             Ritz value of each other mode)}, using every k-th sorted Ritz value of each other mode
#             (k = ⌊K/min(1000^(1/N), K)⌋, 1000^(1/3) = 10, as their `nthroot(1e3, d)`);
#             complex sums ⇒ the edges of their convex hull.  The new pole maximises
#                 ∏ₛ |t − s|^{bs} / ∏ⱼ |t − θⱼ|      (θ = Ritz values of mode i,
#             s = mode i's pole list, every pole with weight bs = rank of Ubᵢ, one copy of the
#             placeholder first entry dropped, as in their `ratfun`)
#   output    HOSVD truncation of Y (truncation_rank per mode):  X ≈ G ×ᵢ (Vᵢ Pᵢ)
#
# Stopping as RKSM (rksm.jl): residual of the TRUNCATED output ≤ tol, or its stagnation once
# the truncation limits it (untruncated Galerkin residual ≤ 0.1 × that residual).
# Equivalent but not identical to the MATLAB code: Druskin-Simoncini basis order instead of
# pole swapping, exact spectral intervals instead of estimates.

mutable struct TBRK{TA,I,N}
    bases::NTuple{N,KrylovBasis{TA}}; inners::NTuple{N,I}
    general::Bool
    m::NTuple{N,Float64}; M::NTuple{N,Float64}       # mᵢ = Σ_{j≠i} λmin(Aⱼ), Mᵢ = Σ_{j≠i} λmax(Aⱼ)
    poles::NTuple{N,Vector{ComplexF64}}              # [placeholder far end, near end, ...]
    core::CoreSolver{N}
    C::Vector{Float64}; Y::Vector{Float64}; T1::Vector{Float64}; T2::Vector{Float64}   # flat Kc^N
    hosvd::HOSVDWork; P::NTuple{N,Matrix{Float64}}; r::NTuple{N,Int}
    ritz::NTuple{N,Vector{ComplexF64}}               # sorted Ritz values of each mode
    Hc::Matrix{Float64}; Z::Matrix{Float64}; wr::Vector{Float64}; wi::Vector{Float64}
    geeswork::Vector{Float64}; syevwork::Vector{Float64}
    sums::Vector{ComplexF64}; nodes::Vector{Float64}; hull_in::Vector{ComplexF64}; hull::Vector{ComplexF64}
    G::Vector{Float64}; gb::Vector{Float64}          # truncated output core, flat RHS core
    Uo::NTuple{N,Matrix{Float64}}; AUo::NTuple{N,Matrix{Float64}}    # truncated output factors
    residual::TuckerResidual{N}
    res_hist::Vector{Float64}
end

"""
    TBRK(p::TuckerProblem, inners; general, Kmax = min(n, 256), rcap = 64, maxit = 100)

`inners[i]` provides `shifted_solve!` / `pair_solve!` for Aᵢ.  Kmax = capacity of each space,
rcap = largest output rank.
"""
function TBRK(p::TuckerProblem{TA,N}, inners; general::Bool, Kmax::Int = min(minimum(size(A, 1) for A in p.As), 256),
              rcap::Int = 64, maxit::Int = 100) where {TA,N}
    rb = size(p.Gb); n = ntuple(i -> size(p.As[i], 1), N)
    lo = ntuple(i -> p.specs[i][1], N); hi = ntuple(i -> p.specs[i][2], N)
    Kc = Kmax; KN = Kc^N
    # adm_pole subsamples every k-th Ritz value of a mode, k = ⌊K / min(m, K)⌋, m = 1000^(1/N):
    # at most ⌈2m⌉ − 1 of them (19 for N = 3), so at most that to the power N − 1 sums over the
    # other modes, plus the 2 interval ends.
    ns = (ceil(Int, 2exp10(3 / N)) - 1)^(N - 1)
    TBRK(ntuple(i -> KrylovBasis(p.As[i], rb[i], Kc; deflation = :eps), N), Tuple(inners), general,
         ntuple(i -> sum(lo) - lo[i], N), ntuple(i -> sum(hi) - hi[i], N),
         ntuple(_ -> sizehint!(ComplexF64[], 2maxit + 4), N), CoreSolver(Kc, N; general),
         scratch(KN), scratch(KN), scratch(KN), scratch(KN), HOSVDWork(Kc, N), ntuple(_ -> zeros(Kc, Kc), N),
         ntuple(_ -> 0, N), ntuple(_ -> zeros(ComplexF64, Kc), N), scratch(Kc, Kc), scratch(Kc, Kc), zeros(Kc), zeros(Kc),
         zeros(gees_lwork(Kc)), zeros(syev_lwork(Kc)),
         zeros(ComplexF64, ns), zeros(ns + 2), zeros(ComplexF64, ns + 2), zeros(ComplexF64, 2(ns + 2) + 1),
         scratch(rcap^N), zeros(prod(rb)), ntuple(i -> scratch(n[i], rcap), N), ntuple(i -> scratch(n[i], rcap), N),
         TuckerResidual(n, rcap, maximum(rb)), zeros(maxit))
end

"""
    solve!(t, p; tol, theta) -> (; status, iters, residual, ranks, dims)

Output: `tbrk_factors(t)` = (U, G), the truncated approximation X ≈ G ×ᵢ Uᵢ.
"""
function solve!(t::TBRK{TA,I,N}, p::TuckerProblem; tol::Float64, theta::Float64,
                maxit::Int = length(t.res_hist)) where {TA,I,N}
    for i in 1:N
        start!(t.bases[i], p.Ub[i])
        mi, Mi = t.m[i], t.M[i]                       # S = [−Mᵢ, −mᵢ]: [far end, near end]
        empty!(t.poles[i])
        abs(mi) < abs(Mi) ? push!(t.poles[i], -Mi, -mi) : push!(t.poles[i], -mi, -Mi)
    end
    db = size(p.Gb); copyto!(t.gb, p.Gb)
    status = :maxiter; res = Inf; prev = Inf; it = 0
    for k in 1:min(maxit, length(t.res_hist))
        it = k
        for i in 1:N                                  # extend every space with its current pole
            extend!(t, i)
        end
        res, res_galerkin = check!(t, p, db, theta)
        t.res_hist[k] = res
        if res ≤ tol
            status = :converged; break
        end
        if stagnated(res, prev) && res_galerkin ≤ 0.1res
            status = :stagnated; break
        end
        prev = res
        for i in 1:N                                  # next poles
            ritz_values!(t, i)
        end
        for i in 1:N
            ξ = adm_pole(t, i)
            imag(ξ) == 0 ? push!(t.poles[i], ξ) : push!(t.poles[i], ξ, conj(ξ))
        end
    end
    return (; status, iters = it, residual = res, ranks = t.r, dims = ntuple(i -> t.bases[i].K, Val(N)))
end

@views tbrk_factors(t::TBRK) = (factor_views(t.Uo, t.r), reshape(t.G[1:prod(t.r)], t.r))

"""Add (Aᵢ − ξI)⁻¹ u to space i, ξ = its newest pole (the pair block if complex)."""
@views function extend!(t::TBRK, i)
    kb = t.bases[i]; ps = t.poles[i]
    pair = length(ps) ≥ 2 && imag(ps[end]) != 0 && ps[end] == conj(ps[end-1])
    ξ = pair ? ps[end-1] : ps[end]
    J = newest_block(kb); w = min(length(J), kb.b)
    u = kb.U[:, first(J):first(J)+w-1]
    if imag(ξ) == 0
        W = kb.W[:, 1:w]; shifted_solve!(t.inners[i], W, u, real(ξ))
    else
        W = kb.W[:, 1:2w]; pair_solve!(t.inners[i], W, u, ξ)
    end
    return add_block!(kb, W)
end

"""Galerkin solve; returns (residual of the truncated output, untruncated Galerkin residual
‖Σᵢ Y ×ᵢ Γᵢ‖ / ‖Gb‖ from the stick-out of the first blocks).  Leaves the output in Uo, G, r."""
@views function check!(t::TBRK{TA,I,N}, p::TuckerProblem, db, θ) where {TA,I,N}
    d = ntuple(i -> t.bases[i].K, Val(N))
    # projected RHS  Gb ×ᵢ (VᵢᵀUbᵢ)
    n_mode_products!(t.C, t.gb, db, ntuple(i -> t.bases[i].UC[1:d[i], :], Val(N)), (t.T1, t.T2))
    solve_core!(t.core, ntuple(i -> t.bases[i].H[1:d[i], 1:d[i]], Val(N)), d, t.C, t.Y)
    r2 = 0.0
    for i in 1:N
        kb = t.bases[i]
        Γ = stickout!(kb, kb.blocks[1])
        e = n_mode_product!(t.T1, t.Y, d, Γ, i)
        r2 += sumsq(t.T1, prod(e))
    end
    res_galerkin = sqrt(r2 / sumsq(t.gb, prod(db)))
    # truncated output: HOSVD of Y,  Uoᵢ = Vᵢ Pᵢ,  G = Y ×ᵢ Pᵢᵀ
    r = ntuple(i -> mode_svd!(t.hosvd, t.P[i][1:d[i], 1:d[i]], t.Y, d, i, θ), Val(N))
    maximum(r) ≤ size(t.Uo[1], 2) || error("TBRK output rank $r exceeds rcap")
    for i in 1:N
        Pi = t.P[i][1:d[i], 1:r[i]]
        mul!(t.Uo[i][:, 1:r[i]], basis(t.bases[i]), Pi)
        mul!(t.AUo[i][:, 1:r[i]], t.bases[i].AU[:, 1:d[i]], Pi)
    end
    n_mode_products!(t.G, t.Y, d, ntuple(i -> t.P[i][1:d[i], 1:r[i]]', Val(N)), (t.T1, t.T2))
    t.r = r
    res = relative_residual!(t.residual, factor_views(t.Uo, r), factor_views(t.AUo, r), t.G, r, p.Ub, t.gb, db)
    return res, res_galerkin
end

"""Ritz values of mode i (eigenvalues of Hᵢ), sorted as MATLAB's `sort` does (complex: by modulus, then angle)."""
@views function ritz_values!(t::TBRK, i)
    kb = t.bases[i]; K = kb.K; θ = t.ritz[i]
    Hc = t.Hc[1:K, 1:K]; H = kb.H[1:K, 1:K]
    if t.general
        copyto!(Hc, H)
        gees!(Hc, t.wr[1:K], t.wi[1:K], t.Z[1:K, 1:K], t.geeswork)
        for j in 1:K
            θ[j] = complex(t.wr[j], t.wi[j])
        end
    else
        for c in 1:K, r in 1:K
            Hc[r, c] = 0.5 * (H[r, c] + H[c, r])
        end
        syev!(Hc, t.wr[1:K], t.syevwork; vectors = false)
        for j in 1:K
            θ[j] = t.wr[j]
        end
    end
    v = θ[1:K]
    if all(z -> imag(z) == 0, v)
        sort!(v; by = real, alg = QuickSort)
    else
        sort!(v; by = z -> (abs(z), angle(z)), alg = QuickSort)
    end
    return v
end

"""log of their objective ∏ₛ|x − s|^{bs} / ∏ⱼ|x − θⱼ| (first, placeholder pole: weight bs − 1)."""
function adm_objective(x::ComplexF64, poles, θ, nθ::Int, bs::Int)
    v = 0.0
    for k in eachindex(poles)
        v += (k == 1 ? bs - 1 : bs) * log(abs(x - poles[k]))
    end
    for j in 1:nθ
        v -= log(abs(x - θ[j]))
    end
    return v
end

"""Their `poles_selection_ADM` + `newpole_adaptive_ADM` for mode i."""
@views function adm_pole(t::TBRK{TA,I,N}, i) where {TA,I,N}
    others = nonmodes(t.r, i)
    K = ntuple(q -> t.bases[others[q]].K, Val(N - 1))
    step = ntuple(q -> max(1, floor(Int, K[q] / min(exp10(3 / N), K[q]))), Val(N - 1))   # their nthroot(1e3, d)
    c = 0; allreal = true
    for J in CartesianIndices(ntuple(q -> length(1:step[q]:K[q]), Val(N - 1)))
        s = zero(ComplexF64)                          # −(sum of one subsampled Ritz value per other mode)
        for q in 1:N-1
            s += t.ritz[others[q]][1 + (J[q] - 1) * step[q]]
        end
        c += 1; t.sums[c] = -s
        imag(s) != 0 && (allreal = false)
    end
    a, b = -t.m[i], -t.M[i]
    ps = t.poles[i]; θ = t.ritz[i]; nθ = t.bases[i].K; bs = t.bases[i].b
    best = -Inf; ξ = complex(a)
    if allreal                                       # 20 points per interval of the sorted nodes
        nodes = t.nodes[1:c+2]
        nodes[1] = a; nodes[2] = b
        for j in 1:c
            nodes[j+2] = real(t.sums[j])
        end
        sort!(nodes; alg = QuickSort)
        for q in 1:c+1, x in range(nodes[q], nodes[q+1]; length = 20)
            v = adm_objective(complex(x), ps, θ, nθ, bs)
            v > best && (best = v; ξ = complex(x))
        end
    else                                             # edges of the convex hull in ℂ
        for j in 1:c
            t.hull_in[j] = t.sums[j]
        end
        t.hull_in[c+1] = a; t.hull_in[c+2] = b
        m = convex_hull!(t.hull, t.hull_in, c + 2)
        for q in 1:m-1, x in LinRange(t.hull[q], t.hull[q+1], 20)
            v = adm_objective(x, ps, θ, nθ, bs)
            v > best && (best = v; ξ = x)
        end
    end
    return ξ
end
