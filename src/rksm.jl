# RKSM: block rational Krylov subspace method with adaptive poles for A X + X Bᵀ = Cu Cs Cvᵀ.
# Port of Casulli & Robol's `rk_adaptive_sylvester.m` (github.com/numpi/rk_adaptive_sylvester,
# "An efficient block rational Krylov solver for Sylvester equations with adaptive pole
# selection", arXiv:2301.08103), with their default sADM poles and options.real = true.
#
#   spaces    U ⊂ ℝⁿ for A, V ⊂ ℝᵐ for B, both started from the RHS factors (Cu, Cv);
#             each iteration adds (A − ξI)⁻¹ u to U (and (B − ηI)⁻¹ v to V), u = first b columns
#             of the newest block (Druskin-Simoncini continuation, b = rank of C);
#             a complex pole adds its conjugate too: the real block [Re y  Im y]
#   Galerkin  (UᵀAU) Y + Y (VᵀBV)ᵀ = (UᵀCu) Cs (VᵀCv)ᵀ,   X ≈ U Y Vᵀ
#   poles     for the space of A, ξ lies in S = −spec(B) (and vice versa).  sADM:
#             maximise over candidates t ∈ S
#                 ∏_{j≥2} |t − s_j| / ( |t − θ_(1)| · ∏_{m≥1} |t − θ_(m·bs+1)| )
#             s = this space's pole list (its first entry is a placeholder: the far end of S),
#             θ_(i) = this space's Ritz values sorted by distance to t, bs = #θ ÷ #s.
#             Candidates: 20 points on each interval between consecutive sorted points of
#             {ends of S, −(Ritz values of the other space)}, or on each edge of their convex
#             hull in ℂ when those Ritz values are complex.  First pole: the end of S nearest 0.
#   output    Y = P Σ Qᵀ truncated (truncation_rank):  X ≈ (U P_r) Σ_r (V Q_r)ᵀ
#
# Stopping (paper policy, truncation.jl): the relative residual of the TRUNCATED output (the
# approximation RKSM would return) ≤ tol, or its stagnation once the truncation limits it
# (the untruncated Galerkin residual ≤ 0.1 × that residual).
#
# Equivalent but not identical to the MATLAB code (same iterates in exact arithmetic): the
# basis is kept in Druskin-Simoncini order instead of rktoolbox's pole swapping; the
# projected equation is solved by real Schur + trsyl (+ one refinement step); the objective
# is evaluated in logarithms; the spectral intervals are exact problem data.

mutable struct RKSMSide{TA,I}
    basis::KrylovBasis{TA}
    inner::I
    spec_other::Tuple{Float64,Float64}               # spectrum of the OTHER operator: S = −spec_other
    poles::Vector{ComplexF64}                        # [placeholder far end of S, near end, ...]
    ritz::Vector{ComplexF64}; T::Matrix{Float64}; Z::Matrix{Float64}   # Ritz values of UᵀAU
    wr::Vector{Float64}; wi::Vector{Float64}; geeswork::Vector{Float64}
    dist::Vector{Float64}                            # distances |t − θᵢ|
    pts::Vector{Float64}; hull_in::Vector{ComplexF64}; hull::Vector{ComplexF64}   # candidate search
end

function RKSMSide(A, inner, b, Kmax, spec_other)
    RKSMSide(KrylovBasis(A, b, Kmax; deflation = :none), inner, spec_other,
             sizehint!(ComplexF64[], Kmax + 4), zeros(ComplexF64, Kmax), scratch(Kmax, Kmax), scratch(Kmax, Kmax),
             zeros(Kmax), zeros(Kmax), zeros(gees_lwork(Kmax)), zeros(Kmax),
             zeros(Kmax + 2), zeros(ComplexF64, Kmax + 2), zeros(ComplexF64, 2Kmax + 5))
end

mutable struct RKSM{SA,SB}
    A::SA; B::SB                                      # the two RKSMSides
    sylv::SmallSylvester
    Y::Matrix{Float64}; Ch::Matrix{Float64}; T1::Matrix{Float64}; T2::Matrix{Float64}
    σ::Vector{Float64}; P::Matrix{Float64}; QT::Matrix{Float64}; Ysvd::Matrix{Float64}
    svdwork::Vector{Float64}; iwork::Vector{BlasInt}
    r::Int; Uo::Matrix{Float64}; Vo::Matrix{Float64}; So::Matrix{Float64}   # truncated output
    AUo::Matrix{Float64}; BVo::Matrix{Float64}
    residual::MatrixResidual
    res_hist::Vector{Float64}
end

"""
    RKSM(p::MatrixProblem, innerA, innerB; Kmax = min(n, 1000), rmax_out = 256, maxit = 100)

`innerA`, `innerB` provide `shifted_solve!` and `pair_solve!` (BandedLDLt or GMRES).
"""
function RKSM(p::MatrixProblem, innerA, innerB; Kmax::Int = min(size(p.A, 1), size(p.B, 1), 1000),
              rmax_out::Int = 256, maxit::Int = 100)
    n, m = size(p.A, 1), size(p.B, 1); rc = size(p.Cs, 1)
    sA = RKSMSide(p.A, innerA, rc, Kmax, p.specB)
    sB = RKSMSide(p.B, innerB, rc, Kmax, p.specA)
    RKSM(sA, sB, SmallSylvester(Kmax, Kmax), scratch(Kmax, Kmax), scratch(Kmax, Kmax), scratch(Kmax, Kmax),
         scratch(Kmax, Kmax), zeros(Kmax), scratch(Kmax, Kmax), scratch(Kmax, Kmax), scratch(Kmax, Kmax),
         zeros(svd_lwork(Kmax, Kmax)), zeros(BlasInt, 8Kmax),
         0, scratch(n, rmax_out), scratch(m, rmax_out), zeros(rmax_out, rmax_out),
         scratch(n, rmax_out), scratch(m, rmax_out), MatrixResidual(n, m, rmax_out, rc), zeros(maxit))
end

"""
    solve!(rk, p; tol, theta) -> (; status, iters, residual, rank, dim)

Output: `rksm_factors(rk)` = (U, S, V), the truncated approximation X ≈ U S Vᵀ.
"""
function solve!(rk::RKSM, p::MatrixProblem; tol::Float64, theta::Float64, maxit::Int = length(rk.res_hist))
    A, B = rk.A, rk.B
    start!(A.basis, p.Cu); start!(B.basis, p.Cv)
    initial_poles!(A); initial_poles!(B)
    status = :maxiter; res = Inf; prev = Inf; it = 0
    kmax = min(maxit, length(rk.res_hist))
    for k in 1:kmax
        it = k
        res, res_galerkin = check!(rk, p, theta)
        rk.res_hist[k] = res
        if res ≤ tol
            status = :converged; break
        end
        if stagnated(res, prev) && res_galerkin ≤ 0.1res
            status = :stagnated; break
        end
        prev = res
        k == kmax && break
        # next poles: both from the Ritz values of THIS step, then both spaces are extended
        if k == 1
            ξA = pop!(A.poles); ξB = pop!(B.poles)    # first pole: near end of S (re-pushed below)
        else
            ritz_values!(A); ritz_values!(B)
            ξA = sadm_pole(A, B.ritz, B.basis.K)
            ξB = sadm_pole(B, A.ritz, A.basis.K)
        end
        (extend!(A, ξA) == 0 || extend!(B, ξB) == 0) && (status = :maxdim; break)
    end                                               # (the output was truncated by the last check)
    return (; status, iters = it, residual = res, rank = rk.r, dim = A.basis.K)
end

@views rksm_factors(rk::RKSM) = (rk.Uo[:, 1:rk.r], rk.So[1:rk.r, 1:rk.r], rk.Vo[:, 1:rk.r])

"""Their initial pole list [far end of S, near end of S], S = −spec(other operator)."""
function initial_poles!(s::RKSMSide)
    lo, hi = -s.spec_other[2], -s.spec_other[1]
    near, far = abs(lo) < abs(hi) ? (lo, hi) : (hi, lo)
    empty!(s.poles); push!(s.poles, far, near)
    return s
end

"""
Galerkin solve on the current spaces; returns (residual of the truncated output, untruncated
Galerkin residual).  The latter is exact from the stick-out of the first blocks:
‖A UYVᵀ + UYVᵀBᵀ − C‖² = ‖Γ_A Y‖² + ‖Y Γ_Bᵀ‖².
"""
@views function check!(rk::RKSM, p::MatrixProblem, θ)
    kA, kB = rk.A.basis, rk.B.basis; KA, KB = kA.K, kB.K; rc = size(p.Cs, 1)
    T1 = rk.T1[1:KA, 1:rc]; mul!(T1, kA.UC[1:KA, :], p.Cs)
    Ch = rk.Ch[1:KA, 1:KB]; mul!(Ch, T1, kB.UC[1:KB, :]')
    Y = rk.Y[1:KA, 1:KB]
    solve_sylvester!(rk.sylv, Y, kA.H[1:KA, 1:KA], kB.H[1:KB, 1:KB], Ch)
    ΓA = stickout!(kA, kA.blocks[1]); ΓB = stickout!(kB, kB.blocks[1])
    E1 = rk.T1[1:size(ΓA, 1), 1:KB]; mul!(E1, ΓA, Y)
    E2 = rk.T2[1:KA, 1:size(ΓB, 1)]; mul!(E2, Y, ΓB')
    res_galerkin = sqrt(norm(E1)^2 + norm(E2)^2) / norm(p.Cs)
    truncate_output!(rk, θ)
    r = rk.r
    res = relative_residual!(rk.residual, rk.Uo[:, 1:r], rk.So[1:r, 1:r], rk.Vo[:, 1:r],
                             rk.AUo[:, 1:r], rk.BVo[:, 1:r], p.Cu, p.Cs, p.Cv)
    return res, res_galerkin
end

"""Truncated output from Y = P Σ Qᵀ: Uo = U P_r, Vo = V Q_r, So = Σ_r (with A·Uo, B·Vo from A·U, B·V)."""
@views function truncate_output!(rk::RKSM, θ)
    kA, kB = rk.A.basis, rk.B.basis; KA, KB = kA.K, kB.K; k = min(KA, KB)
    W = rk.Ysvd[1:KA, 1:KB]; copyto!(W, rk.Y[1:KA, 1:KB])
    σ = rk.σ[1:k]; P = rk.P[1:KA, 1:k]; QT = rk.QT[1:k, 1:KB]
    gesdd!(W, σ, P, QT, rk.svdwork, rk.iwork)
    r = min(truncation_rank(σ, θ), size(rk.Uo, 2))
    Pr = P[:, 1:r]; Qr = QT[1:r, :]'
    mul!(rk.Uo[:, 1:r], basis(kA), Pr); mul!(rk.AUo[:, 1:r], kA.AU[:, 1:KA], Pr)
    mul!(rk.Vo[:, 1:r], basis(kB), Qr); mul!(rk.BVo[:, 1:r], kB.AU[:, 1:KB], Qr)
    S = rk.So[1:r, 1:r]; fill!(S, 0.0)
    for i in 1:r
        S[i, i] = σ[i]
    end
    rk.r = r
    return r
end

"""Ritz values of this space: the eigenvalues of UᵀAU (real Schur form)."""
@views function ritz_values!(s::RKSMSide)
    K = s.basis.K
    T = s.T[1:K, 1:K]; copyto!(T, s.basis.H[1:K, 1:K])
    gees!(T, s.wr[1:K], s.wi[1:K], s.Z[1:K, 1:K], s.geeswork)
    for i in 1:K
        s.ritz[i] = complex(s.wr[i], s.wi[i])
    end
    return s
end

"""log of their sADM objective at t (own Ritz values θ, own pole list s incl. the placeholder)."""
@views function sadm_objective(t::ComplexF64, s::RKSMSide, K::Int, bs::Int)
    v = 0.0
    for j in 2:length(s.poles)
        v += log(abs(t - s.poles[j]))
    end
    dist = s.dist[1:K]
    for i in 1:K
        dist[i] = abs(t - s.ritz[i])
    end
    sort!(dist; alg = QuickSort)
    v -= log(dist[1])
    for m in bs+1:bs:K
        v -= log(dist[m])
    end
    return v
end

"""Their `newpole_adaptive_sADM` for side s; φ = Ritz values of the other space (nφ of them)."""
@views function sadm_pole(s::RKSMSide, φ, nφ::Int)
    K = s.basis.K; bs = max(1, K ÷ length(s.poles))
    a, b = -s.spec_other[1], -s.spec_other[2]        # ends of S
    best = -Inf; ξ = complex(a)
    if all(i -> imag(φ[i]) == 0, 1:nφ)               # real case: 20 points per interval
        pts = s.pts[1:nφ+2]
        pts[1] = a; pts[2] = b
        for j in 1:nφ
            pts[j+2] = -real(φ[j])
        end
        sort!(pts; alg = QuickSort)
        for i in 1:nφ+1, t in range(pts[i], pts[i+1]; length = 20)
            v = sadm_objective(complex(t), s, K, bs)
            v > best && (best = v; ξ = complex(t))
        end
    else                                             # complex case: edges of the convex hull
        for j in 1:nφ
            s.hull_in[j] = -φ[j]
        end
        s.hull_in[nφ+1] = a; s.hull_in[nφ+2] = b
        m = convex_hull!(s.hull, s.hull_in, nφ + 2)
        for i in 1:m-1, t in LinRange(s.hull[i], s.hull[i+1], 20)
            v = sadm_objective(t, s, K, bs)
            v > best && (best = v; ξ = t)
        end
    end
    return ξ
end

"""Add (A − ξI)⁻¹ u (u = first b columns of the newest block); a complex ξ adds the pair
[Re y  Im y].  Returns the number of columns added (0: the space is full)."""
@views function extend!(s::RKSMSide, ξ::ComplexF64)
    kb = s.basis
    J = newest_block(kb); w = min(length(J), kb.b)
    u = kb.U[:, first(J):first(J)+w-1]
    if imag(ξ) == 0
        push!(s.poles, ξ)
        W = kb.W[:, 1:w]
        shifted_solve!(s.inner, W, u, real(ξ))
    else
        push!(s.poles, ξ, conj(ξ))
        W = kb.W[:, 1:2w]
        pair_solve!(s.inner, W, u, ξ)
    end
    return add_block!(kb, W)
end

"""Convex hull of the points P[1:n] (monotone chain) into H, counter-clockwise and closed
(H[end] = H[1]); returns its length.  P is sorted in place."""
@views function convex_hull!(H::Vector{ComplexF64}, P::Vector{ComplexF64}, n::Int)
    sort!(P[1:n]; by = z -> (real(z), imag(z)), alg = QuickSort)
    cross(o, a, b) = (real(a) - real(o)) * (imag(b) - imag(o)) - (imag(a) - imag(o)) * (real(b) - real(o))
    m = 0
    for j in 1:n                                     # lower hull
        while m ≥ 2 && cross(H[m-1], H[m], P[j]) ≤ 0
            m -= 1
        end
        m += 1; H[m] = P[j]
    end
    lo = m + 1
    for j in n-1:-1:1                                # upper hull
        while m ≥ lo && cross(H[m-1], H[m], P[j]) ≤ 0
            m -= 1
        end
        m += 1; H[m] = P[j]
    end
    return m
end
