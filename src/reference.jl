# Exact solutions, used only to measure errors and the truncated-exact residual level in the
# figures (Laplacian: sine transform; random: eigenfactors; convection-diffusion: tridiagonal solves).  Not part of any method, not timed.  They are computed dense, then the matrix ones are
# kept as low-rank factors (exact to rounding) and the tensor ones as their truncation.

"""Dirichlet Laplacians: diagonalised by the discrete sine transform (DST-I, FFTW's RODFT00)."""
function dst_solve(Bfull)
    X = copy(Bfull)
    n = size(X, 1); λ = laplacian_eigenvalues(n)
    FFTW.r2r!(X, FFTW.RODFT00)
    scale = (2(n + 1))^ndims(X)                       # DST-I applied twice = (2(n+1)) × identity
    for I in CartesianIndices(X)
        X[I] /= sum(λ[i] for i in Tuple(I)) * scale
    end
    FFTW.r2r!(X, FFTW.RODFT00)
    return X
end

reference_laplacian(p::MatrixProblem) = dst_solve(p.Cu * p.Cs * p.Cv')
reference_laplacian(p::TuckerProblem) = dst_solve(full_tensor(p.Ub, p.Gb))

"""
Non-normal random problems through the eigenfactors (A = P D P⁻¹):  X = P_A ((P_A⁻¹ C P_B⁻ᵀ) ./
(dA_i + dB_j)) P_Bᵀ, then `refine` steps of iterative refinement against the STORED operators
(forming P D P⁻¹ in floating point perturbs them at the cond(P)·eps level).
"""
function reference_random(p::MatrixProblem, f; refine = 3)
    solve(R) = f.PA * ((f.iPA * R * f.iPB') ./ (f.dA .+ f.dB')) * f.PB'
    C = p.Cu * p.Cs * p.Cv'
    X = solve(C)
    for _ in 1:refine
        X .+= solve(C - p.A * X - X * p.B')
    end
    @printf("  reference: relative residual %.2e\n", norm(C - p.A * X - X * p.B') / norm(C))
    return X
end

function reference_random(p::TuckerProblem{TA,N}, f; refine = 2) where {TA,N}
    function solve(R)
        Z = R
        for k in 1:N; Z = n_mode_product(Z, f[k].iP, k); end
        Z = Z ./ [sum(f[m].d[I[m]] for m in 1:N) for I in CartesianIndices(Z)]
        for k in 1:N; Z = n_mode_product(Z, f[k].P, k); end
        return Z
    end
    residual(X) = B - sum(n_mode_product(X, p.As[k], k) for k in 1:N)
    B = full_tensor(p.Ub, p.Gb)
    X = solve(B)
    for _ in 1:refine
        X .+= solve(residual(X))
    end
    @printf("  reference: relative residual %.2e\n", norm(residual(X)) / norm(B))
    return X
end

"""
Convection-diffusion (2-D), from the structure: A X + X Bᵀ = C is one big K-step (Section 3.2 of
the Linear solves page of the docs) with the diffusive, symmetric tridiagonal B = Q M Qᵀ (LAPACK's tridiagonal
eigensolver, Q orthogonal).  X̃ = X Q solves A X̃ + X̃ M = C Q, column by column
(A + μⱼ I) x̃ⱼ = (CQ)ⱼ: n tridiagonal LU solves (`BandedLU`).  Then X = X̃ Qᵀ.  Orthogonal
transformations and backward-stable solves only: no inverse, no refinement.
"""
@views function reference_convdiff(p::MatrixProblem)
    issymmetric(p.B) || error("reference_convdiff needs a symmetric B")
    M = eigen(SymTridiagonal(Vector(diag(p.B)), Vector(diag(p.B, 1))))
    CQ = p.Cu * (p.Cs * (p.Cv' * M.vectors))
    Xt = similar(CQ); lu = BandedLU(p.A; width = 1)
    for j in axes(Xt, 2)
        shifted_solve!(lu, Xt[:, j:j], CQ[:, j:j], -M.values[j])    # (A + μⱼ I) x̃ⱼ = (CQ)ⱼ
    end
    X = Xt * M.vectors'
    C = p.Cu * p.Cs * p.Cv'
    @printf("  reference: relative residual %.2e\n", norm(C - p.A * X - X * p.B') / norm(C))
    return X
end

"""
Low-rank factors of the exact matrix solution, X ≈ U diag(σ[1:k]) Vᵀ, so that it can be stored
and used without the dense n × n matrix.  Randomised SVD (range finder with `power` subspace
iterations, Halko-Martinsson-Tropp): exact to rounding because σ decays far below eps·σ₁ within
the first K values.  Returns
  σ    : the leading K singular values,
  tail : ‖X − (best rank-K approximation)‖_F, i.e. the ℓ² norm of all the others,
  U, V : the singular vectors up to the smallest rank k with ‖σ_{k+1:end}‖₂ ≤ cutoff·‖X‖_F
         (beyond k the reference itself is rounding noise; see its residual).
"""
function factor_reference(X::Matrix; K = 128, oversample = 32, power = 2, cutoff = 1e-14)
    orth(Y) = Matrix(qr(Y).Q)
    Q = orth(X * randn(MersenneTwister(0), size(X, 2), K + oversample))
    for _ in 1:power
        Q = orth(X * orth(X' * Q))
    end
    F = svd(Q' * X)
    rest2 = 0.0                                       # ‖X − Q Qᵀ X‖_F², by column blocks
    for j0 in 1:1024:size(X, 2)
        js = j0:min(j0 + 1023, size(X, 2))
        Xj = view(X, :, js)
        rest2 += sum(abs2, Xj .- Q * (Q' * Xj))
    end
    σ = F.S[1:K]
    tail = sqrt(sum(abs2, F.S[K+1:end]) + rest2)
    k = truncation_rank([σ; tail], cutoff)
    return (U = Q * F.U[:, 1:k], sigma = σ, V = F.V[:, 1:k], tail = tail)
end

"""Truncation of the factored exact solution `F` with the methods' rule: (U, V, r)."""
function truncate_reference(F::NamedTuple, θ)
    r = truncation_rank([F.sigma; F.tail], θ)       # the tail counts as one more singular value
    r ≤ size(F.U, 2) || error("stored reference rank $(size(F.U, 2)) < truncation rank $r")
    return F.U[:, 1:r], F.V[:, 1:r], r
end

"""HOSVD truncation of the exact tensor solution with the methods' rule: (U, G, ranks)."""
function truncate_reference(X::Array{Float64,N}, θ) where {N}
    U = ntuple(N) do k
        F = svd(matricize(X, k))
        F.U[:, 1:truncation_rank(F.S, θ)]
    end
    G = foldl((Y, k) -> n_mode_product(Y, U[k]', k), 1:N; init = X)
    return U, G, size(G)
end

"""
‖X_r − U S Vᵀ‖_F / ‖X_r‖_F with X_r = Ur diag(σr) Vrᵀ (from `truncate_reference`): QR of the
stacked factors, X_r − Y = Q_a R_a blockdiag(diag(σr), −S) R_bᵀ Q_bᵀ, so no n × n matrix is
formed.  (The difference is not formed as two inner products, which would cancel when Y ≈ X_r.)
"""
function relative_error((Ur, σr, Vr), U, S, V)
    k = length(σr); r = size(U, 2)
    D = zeros(k + r, k + r)
    D[1:k, 1:k] = Diagonal(σr)
    D[k+1:end, k+1:end] = -S
    Ra = qr([Ur U]).R; Rb = qr([Vr V]).R
    return norm(Ra * D * Rb') / norm(σr)
end

"""Relative residual of the matrix approximation U S Vᵀ (allocating convenience wrapper)."""
function relative_residual(p::MatrixProblem, U, S, V)
    ws = MatrixResidual(size(p.A, 1), size(p.B, 1), size(S, 1), size(p.Cs, 1))
    return relative_residual!(ws, U, S, V, p.A * U, p.B * V, p.Cu, p.Cs, p.Cv)
end

"""Relative residual of the Tucker approximation G ×ᵢ Uᵢ (allocating convenience wrapper)."""
function relative_residual(p::TuckerProblem{TA,N}, U, G) where {TA,N}
    n = ntuple(k -> size(p.As[k], 1), N)
    ws = TuckerResidual(n, maximum(size(G)), maximum(size(p.Gb)))
    AU = ntuple(k -> p.As[k] * U[k], N)
    return relative_residual!(ws, U, AU, vec(collect(G)), size(G), p.Ub, vec(copy(p.Gb)), size(p.Gb))
end
