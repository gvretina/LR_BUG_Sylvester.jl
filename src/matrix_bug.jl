# Rank-adaptive BUG for the matrix Sylvester equation  A X + X Bᵀ = C = Cu Cs Cvᵀ.
#
# One iteration, from the current factors U (n × r), S (r × r), V (m × r):
#   K-step    A K + K (VᵀBV)ᵀ = C V            → K  (n × r)
#   L-step    B L + L (UᵀAU)ᵀ = Cᵀ U           → L  (m × r)
#   augment   Û = orth[K U],  V̂ = orth[L V]    (n × 2r, m × 2r)
#   S-step    (ÛᵀAÛ) Ŝ + Ŝ (V̂ᵀBV̂)ᵀ = Ûᵀ C V̂   (2r × 2r, dense)
#   truncate  Ŝ = P Σ Qᵀ, keep r' = truncation_rank(Σ, θ):  U = Û P[:, 1:r'], S = Σ[1:r'], V = V̂ Q[:, 1:r']
# Start: U = Cu, V = Cv.  Stop: relative residual ≤ tol, or stagnation (truncation.jl).
#
# Every array is allocated by the constructor (setup); `solve!` allocates nothing.

mutable struct MatrixBUG{TA,TB,IA,IB}
    A::TA; B::TB
    innerA::IA; innerB::IB                            # K-step solvers for A and for B
    rmax::Int; r::Int
    U::Matrix{Float64}; S::Matrix{Float64}; V::Matrix{Float64}
    AU::Matrix{Float64}; BV::Matrix{Float64}          # A·U, B·V of the current factors
    Uh::Matrix{Float64}; Vh::Matrix{Float64}          # augmented bases Û, V̂
    AUh::Matrix{Float64}; BVh::Matrix{Float64}
    RK::Matrix{Float64}; RL::Matrix{Float64}          # K- and L-step right-hand sides
    MA::Matrix{Float64}; MB::Matrix{Float64}          # projected operators
    W1::Matrix{Float64}; W2::Matrix{Float64}; W3::Matrix{Float64}   # rc × 2rmax products with the RHS
    Ch::Matrix{Float64}; Sh::Matrix{Float64}          # projected RHS and solution of the S-step
    sylv::SmallSylvester
    σ::Vector{Float64}; P::Matrix{Float64}; QT::Matrix{Float64}
    svdwork::Vector{Float64}; iwork::Vector{BlasInt}
    tau::Vector{Float64}; qrwork::Vector{Float64}
    residual::MatrixResidual
    res_hist::Vector{Float64}; rank_hist::Vector{Int}
end

"""
    MatrixBUG(p::MatrixProblem, innerA, innerB; rmax = 128, maxit = 50)

Workspace for ranks up to rmax (capped at n/2 so that [K U] fits).
"""
function MatrixBUG(p::MatrixProblem, innerA, innerB; rmax::Int = 128, maxit::Int = 50)
    n, m = size(p.A, 1), size(p.B, 1); rc = size(p.Cs, 1)
    rmax = min(rmax, n ÷ 2, m ÷ 2); R2 = 2rmax
    MatrixBUG(p.A, p.B, innerA, innerB, rmax, 0,
              scratch(n, rmax), zeros(rmax, rmax), scratch(m, rmax), scratch(n, rmax), scratch(m, rmax),
              scratch(n, R2), scratch(m, R2), scratch(n, R2), scratch(m, R2), scratch(n, rmax), scratch(m, rmax),
              zeros(R2, R2), zeros(R2, R2), zeros(rc, R2), zeros(rc, R2), zeros(rc, R2), zeros(R2, R2), zeros(R2, R2),
              SmallSylvester(R2, R2), zeros(R2), zeros(R2, R2), zeros(R2, R2),
              zeros(svd_lwork(R2, R2)), zeros(BlasInt, 8R2), zeros(R2), zeros(qr_lwork(max(n, m), R2)),
              MatrixResidual(n, m, rmax, rc), zeros(maxit), zeros(Int, maxit))
end

"""
    solve!(bug, p; tol, theta) -> (; status, iters, residual, rank)

Stop at relative residual ≤ tol (or stagnation); truncate with θ = theta.
Solution: `bug_factors(bug)` = (U, S, V) with X ≈ U S Vᵀ.
"""
@views function solve!(bug::MatrixBUG, p::MatrixProblem; tol::Float64, theta::Float64, maxit::Int = length(bug.res_hist))
    reset!(bug.innerA); reset!(bug.innerB)
    r = size(p.Cu, 2)
    copyto!(bug.U[:, 1:r], p.Cu); copyto!(bug.V[:, 1:r], p.Cv)
    status = :maxiter; res = Inf; it = 0
    for k in 1:min(maxit, length(bug.res_hist))
        it = k
        r = bug_iteration!(bug, p, r, theta)
        res = relative_residual!(bug.residual, bug.U[:, 1:r], bug.S[1:r, 1:r], bug.V[:, 1:r],
                                 bug.AU[:, 1:r], bug.BV[:, 1:r], p.Cu, p.Cs, p.Cv)
        bug.res_hist[k] = res; bug.rank_hist[k] = r
        if res ≤ tol
            status = :converged; break
        end
        if k > 1 && stagnated(res, bug.res_hist[k-1])
            status = :stagnated; break
        end
    end
    bug.r = r
    return (; status, iters = it, residual = res, rank = r)
end

@views bug_factors(bug::MatrixBUG) = (bug.U[:, 1:bug.r], bug.S[1:bug.r, 1:bug.r], bug.V[:, 1:bug.r])

@views function bug_iteration!(bug::MatrixBUG, p::MatrixProblem, r, θ)
    A, B = bug.A, bug.B; Cu, Cs, Cv = p.Cu, p.Cs, p.Cv; rc = size(Cs, 1)
    U = bug.U[:, 1:r]; V = bug.V[:, 1:r]

    # K-step:  A K + K (VᵀBV)ᵀ = C V = Cu Cs (Cvᵀ V)
    BV = bug.BVh[:, 1:r]; apply!(bug.innerB, BV, B, V)
    MB = bug.MB[1:r, 1:r]; mul!(MB, V', BV)
    W1 = bug.W1[1:rc, 1:r]; W2 = bug.W2[1:rc, 1:r]
    mul!(W1, Cv', V); mul!(W2, Cs, W1)
    RK = bug.RK[:, 1:r]; mul!(RK, Cu, W2)
    K = bug.Uh[:, 1:r]
    kstep!(bug.innerA, K, RK, MB)

    # L-step:  B L + L (UᵀAU)ᵀ = Cᵀ U = Cv Csᵀ (Cuᵀ U)
    AU = bug.AUh[:, 1:r]; apply!(bug.innerA, AU, A, U)
    MA = bug.MA[1:r, 1:r]; mul!(MA, U', AU)
    mul!(W1, Cu', U); mul!(W2, Cs', W1)
    RL = bug.RL[:, 1:r]; mul!(RL, Cv, W2)
    L = bug.Vh[:, 1:r]
    kstep!(bug.innerB, L, RL, MA)

    # augmentation:  Û = orth[K U],  V̂ = orth[L V]
    copyto!(bug.Uh[:, r+1:2r], U); copyto!(bug.Vh[:, r+1:2r], V)
    orthonormalize!(bug.Uh[:, 1:2r], bug.tau[1:2r], bug.qrwork)
    orthonormalize!(bug.Vh[:, 1:2r], bug.tau[1:2r], bug.qrwork)
    ru = restrict_to_basis!(bug.innerA, bug.Uh[:, 1:2r], bug.tau, bug.qrwork)
    rv = restrict_to_basis!(bug.innerB, bug.Vh[:, 1:2r], bug.tau, bug.qrwork)
    Uh = bug.Uh[:, 1:ru]; Vh = bug.Vh[:, 1:rv]

    # S-step:  (ÛᵀAÛ) Ŝ + Ŝ (V̂ᵀBV̂)ᵀ = (ÛᵀCu) Cs (V̂ᵀCv)ᵀ
    AUh = bug.AUh[:, 1:ru]; apply!(bug.innerA, AUh, A, Uh)
    BVh = bug.BVh[:, 1:rv]; apply!(bug.innerB, BVh, B, Vh)
    MA = bug.MA[1:ru, 1:ru]; mul!(MA, Uh', AUh)
    MB = bug.MB[1:rv, 1:rv]; mul!(MB, Vh', BVh)
    W1 = bug.W1[1:rc, 1:ru]; mul!(W1, Cu', Uh)
    W2 = bug.W2[1:rc, 1:rv]; mul!(W2, Cv', Vh)
    W3 = bug.W3[1:rc, 1:rv]; mul!(W3, Cs, W2)
    Ch = bug.Ch[1:ru, 1:rv]; mul!(Ch, W1', W3)
    Sh = bug.Sh[1:ru, 1:rv]
    solve_sylvester!(bug.sylv, Sh, MA, MB, Ch)

    # truncation:  Ŝ = P Σ Qᵀ
    k = min(ru, rv)
    σ = bug.σ[1:k]; P = bug.P[1:ru, 1:k]; QT = bug.QT[1:k, 1:rv]
    gesdd!(Sh, σ, P, QT, bug.svdwork, bug.iwork)
    rn = min(truncation_rank(σ, θ), bug.rmax)
    Pr = P[:, 1:rn]; Qr = QT[1:rn, :]'
    mul!(bug.U[:, 1:rn], Uh, Pr); mul!(bug.AU[:, 1:rn], AUh, Pr)
    mul!(bug.V[:, 1:rn], Vh, Qr); mul!(bug.BV[:, 1:rn], BVh, Qr)
    S = bug.S[1:rn, 1:rn]; fill!(S, 0.0)
    for i in 1:rn
        S[i, i] = σ[i]
    end
    return rn
end
