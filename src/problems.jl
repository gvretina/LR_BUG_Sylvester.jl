# The test problems of the paper: Laplacian, convection-diffusion, random; matrix and tensor.
#
#   matrix  A X + X Bᵀ = C,           C = Cu Cs Cvᵀ   (Cu, Cv orthonormal)
#   tensor  ∑ᵢ₌₁ᴺ X ×ᵢ Aᵢ = B,   B = Gb ×₁ Ub₁ ⋯ ×_N Ub_N   (Ubᵢ orthonormal; the paper
#           uses N = 3, the tensor code works for any N)
#
# The right-hand sides are always kept in factored form, so no n × n (or n³) array is formed.
# `spec` holds the interval [λmin, λmax] containing the spectrum of each operator; it is
# exact problem data (needed by the pole selection of RKSM and TBRK).
# Every problem is generated from MersenneTwister(1), so the data are reproducible.

struct MatrixProblem{TA,TB}
    A::TA; B::TB
    Cu::Matrix{Float64}; Cs::Matrix{Float64}; Cv::Matrix{Float64}
    specA::Tuple{Float64,Float64}; specB::Tuple{Float64,Float64}
end

struct TuckerProblem{TA,N}
    As::NTuple{N,TA}
    Ub::NTuple{N,Matrix{Float64}}
    Gb::Array{Float64,N}
    specs::NTuple{N,Tuple{Float64,Float64}}
end

grid(n) = collect(LinRange(0, 4π, n + 1)[1:n])

"""1-D finite-difference Laplacian tridiag(1, −2, 1)/h² with Dirichlet conditions, h = 4π/n."""
function laplacian_1d(n)
    h = 4π / n
    return spdiagm(-1 => ones(n - 1), 0 => fill(-2.0, n), 1 => ones(n - 1)) ./ h^2
end

"""Exact spectrum of `laplacian_1d(n)`: −4/h² sin²(kπ / 2(n+1)), k = 1...n."""
laplacian_eigenvalues(n) = (h = 4π / n; [-4 / h^2 * sin(k * π / (2(n + 1)))^2 for k in 1:n])
laplacian_spectrum(n) = extrema(laplacian_eigenvalues(n))

"""Trigonometric basis [1, cos x, sin x, ..., cos Kx, sin Kx] on the grid (n × (2K+1))."""
trig_basis(x, K = 3) = hcat(ones(length(x)), (hcat(cos.(k .* x), sin.(k .* x)) for k in 1:K)...)

"""
Coefficients G (7 × 7) with  Σ_{k,l=-3..3} a_kl cos(k x + l y + φ_kl) = Φ(x) G Φ(y)ᵀ,
a_kl = randn/(1+|k|+|l|), φ_kl uniform in [0, 2π): the smooth random RHS of the paper.
"""
function cosine_coefficients(rng; K = 3)
    G = zeros(2K + 1, 2K + 1)
    ic(k) = k == 0 ? 1 : 2abs(k)          # column of cos(|k|·) in the trig basis
    is(k) = 2abs(k) + 1                   # column of sin(|k|·)
    for k in -K:K, l in -K:K
        amp = randn(rng) / (1 + abs(k) + abs(l)); φ = 2π * rand(rng)
        sk, sl = sign(k), sign(l)
        # cos(kx + ly + φ) = cos(kx)[cos(ly)cosφ − sin(ly)sinφ] − sin(kx)[sin(ly)cosφ + cos(ly)sinφ]
        G[ic(k), ic(l)] += amp * cos(φ)
        l != 0 && (G[ic(k), is(l)] -= amp * sl * sin(φ))
        if k != 0
            l != 0 && (G[is(k), is(l)] -= amp * sk * sl * cos(φ))
            G[is(k), ic(l)] -= amp * sk * sin(φ)
        end
    end
    return G
end

"""Write Φx G Φyᵀ as Cu Cs Cvᵀ with orthonormal Cu, Cv (thin QRs + SVD of the small core)."""
function factor_rhs(Φx, G, Φy)
    Qx, Rx = qr(Φx); Qy, Ry = qr(Φy)
    F = svd(Rx * G * Ry')
    r = count(>(1e-14 * F.S[1]), F.S)
    return Matrix(Qx) * F.U[:, 1:r], Matrix(Diagonal(F.S[1:r])), Matrix(Qy) * F.V[:, 1:r]
end

"""Eigenvalues: smax, smin and n−2 uniform samples in between, sorted; times −1 if s = −1."""
function uniform_eigen(rng, n, smax, smin, s = 1)
    samples = [smax; rand(rng, n - 2) .* (smax - smin) .+ smin; smin]
    samples = s == 1 ? sort(samples) : sort(samples, rev = true)
    return s .* samples .* sign.(samples)
end

"""
Dense non-normal A = P D P⁻¹: P = rand(n, n) with normalised columns, D = `uniform_eigen`.
Returns (A, P, d, P⁻¹).  `inverse` is `inv` for the matrix problem and `pinv` for the tensor
one (as in the original scripts; identical up to rounding for a square, full-rank P).
"""
function nonnormal_matrix(rng, n, eigargs; inverse = pinv)
    P = rand(rng, n, n)
    P ./= sqrt.(sum(abs2, P; dims = 1))
    d = uniform_eigen(rng, n, eigargs...)
    iP = inverse(P)
    return P * Diagonal(d) * iP, P, d, iP
end

"""
    matrix_laplacian(n)

D X + X Dᵀ = C, D = `laplacian_1d(n)`, C = the rank-7 cosine RHS on the grid.
"""
function matrix_laplacian(n)
    rng = MersenneTwister(1)
    D = laplacian_1d(n)
    Φ = trig_basis(grid(n))
    Cu, Cs, Cv = factor_rhs(Φ, cosine_coefficients(rng), Φ)
    sp = laplacian_spectrum(n)
    return MatrixProblem(D, D, Cu, Cs, Cv, sp, sp)
end

"""
    matrix_random(n) -> (problem, eigenfactors)

Non-normal A (spectrum [12, 13]) and B (spectrum [−2, −1]), rank-7 RHS rand(n,7)·rand(7,n).
The eigenfactors (P, d, P⁻¹ of both operators) give the exact solution (`reference_random`).
"""
function matrix_random(n)
    rng = MersenneTwister(1)
    A, PA, dA, iPA = nonnormal_matrix(rng, n, (13, 12); inverse = inv)
    B, PB, dB, iPB = nonnormal_matrix(rng, n, (2, 1, -1); inverse = inv)
    Cu, Cs, Cv = factor_rhs(rand(rng, n, 7), Matrix{Float64}(I, 7, 7), rand(rng, n, 7))
    p = MatrixProblem(A, B, Cu, Cs, Cv, extrema(dA), extrema(dB))
    return p, (PA = PA, dA = dA, iPA = iPA, PB = PB, dB = dB, iPB = iPB)
end

"""
    tensor_laplacian(n; N = 3, r = 7)

N-D Dirichlet Laplacian (`laplacian_1d(n)` in every mode; the paper uses N = 3), RHS of
multilinear rank (r, ..., r): trig basis in each mode, core = Gaussian damped by 1/(1 + |k|₁).
"""
function tensor_laplacian(n; N = 3, r = 7)
    rng = MersenneTwister(1)
    K = (r - 1) ÷ 2
    D = laplacian_1d(n)
    Q, R = qr(trig_basis(grid(n), K))
    freq = [0; repeat(1:K, inner = 2)]
    G = [randn(rng) / (1 + sum(freq[i] for i in Tuple(I))) for I in CartesianIndices(ntuple(_ -> r, N))]
    for k in 1:N
        G = n_mode_product(G, R, k)
    end
    sp = laplacian_spectrum(n)
    return TuckerProblem(ntuple(_ -> D, N), ntuple(_ -> Matrix(Q), N), G, ntuple(_ -> sp, N))
end

"""
1-D convection-diffusion operator, central differences on the grid points t:
A = −ν·tridiag(−1, 2, −1)/h² + diag(w(t))·tridiag(−1, 0, 1)/(2h).
Returns (A, (λmin, λmax)).  Its off-diagonals ν/h² ± wᵢ/(2h) are positive (|w| h < 2ν), so a
diagonal similarity turns A into the symmetric tridiagonal S with off-diagonals
√(Aᵢ,ᵢ₊₁ Aᵢ₊₁,ᵢ): the eigenvalues are real and are computed from S (to rounding).  That
similarity has condition number ≈ exp(∫ w / 2ν): for small ν, A is far from normal.
"""
function convdiff_1d(t, h, ν, w)
    n = length(t)
    up = [ν / h^2 + w[i] / 2h for i in 1:n-1]                    # A[i, i+1]
    lo = [ν / h^2 - w[i+1] / 2h for i in 1:n-1]                  # A[i+1, i]
    d = fill(-2ν / h^2, n)
    A = spdiagm(-1 => lo, 0 => d, 1 => up)
    return A, extrema(eigvals(SymTridiagonal(d, sqrt.(up .* lo))))
end

# Convection-diffusion: the 3-D test of Casulli's TBRK paper (Section 6, Figure 3), and the same
# problem one dimension lower.  On [0, 1]^d with zero boundary values, in the sign convention of
# that paper's code:  ν Δu + w·∇u = f,  ν = 0.1,  w = (1 + (x₁ + 1)²/4, 0, ...),  f = 1/(1 + x₁ + ... + x_d);
# grid t = linspace(0, 1, n), h = 1/(n − 1).  Mode 1 is convective, the others diffusion only.
cd_grid(n) = collect(range(0, 1, length = n))
cd_convective(t) = convdiff_1d(t, t[2] - t[1], 0.1, @. 1 + (1 + t)^2 / 4)
cd_diffusive(t) = convdiff_1d(t, t[2] - t[1], 0.1, zero(t))

"""
    matrix_convdiff(n)

2-D convection-diffusion: A X + X Bᵀ = F with A the convective and B the diffusive operator,
F = 1/(1 + x + y) on the grid, of rank 8 (`reciprocal_basis`).
"""
function matrix_convdiff(n)
    t = cd_grid(n)
    (A, sA), (B, sB) = cd_convective(t), cd_diffusive(t)
    U = reciprocal_basis(t, 2)
    return MatrixProblem(A, B, U, U' * (1 ./ (1 .+ t .+ t')) * U, U, sA, sB)
end

"""
    tensor_convdiff(n)

3-D convection-diffusion: A₁ convective, A₂ = A₃ diffusive; RHS the HOSVD of
f = 1/(1 + x + y + z) on the grid (multilinear rank (9, 9, 9) at n = 1024, (10, 10, 10) at
n = 4096).  At n = 1024 this is the problem of the file `Convection-Diffusion d=3 n=1024.mat`
(numpi/TBRK-Sylvester), except that the file's A₁ has ν = 0.01 = 0.1² (its code applies ν twice
to the convective modes) where the paper states 0.1.
"""
function tensor_convdiff(n)
    t = cd_grid(n)
    (A1, s1), (A2, s2) = cd_convective(t), cd_diffusive(t)
    U = reciprocal_basis(t, 3)
    return TuckerProblem((A1, A2, A2), (U, U, U), reciprocal_core(t, U), (s1, s2, s2))
end

"""
Orthonormal basis of the column space of every unfolding of F = 1/(1 + x₁ + ... + x_d) on the
uniform grid t^d (d = 2, 3; the same for every mode, F is symmetric), truncated as in both
reference codes: singular values ≥ 1e-10.  F is not formed: a column of Mat₁(F) is the vector
1/(1 + t + s), s = the sum of the other d − 1 coordinates, which takes the values s_m = (m − 1)h,
each for mult_m grid points.  So Mat₁(F) Mat₁(F)ᵀ = K W Kᵀ with K_im = 1/(1 + t_i + s_m),
W = diag(mult), and the basis is the leading left singular vectors of K W^{1/2}.
"""
function reciprocal_basis(t, d)
    n = length(t); h = t[2] - t[1]
    mult = d == 2 ? ones(n) : [min(m, 2n - m) for m in 1:2n-1]   # d = 3: #{(j, k) : j + k − 1 = m}
    s = [(m - 1) * h for m in eachindex(mult)]
    F = svd((1 ./ (1 .+ t .+ s')) .* sqrt.(mult'))
    return F.U[:, 1:count(≥(1e-10), F.S)]
end

"""
Core G = F ×ᵢ Uᵀ of F = 1/(1 + x + y + z) on t³ (t uniform), without forming F: F_ijk = K_{i,m}
with m = j + k − 1 and K_im = 1/(1 + t_i + (m − 1)h), so G_abc = Σ_i U_ia Σ_m K_im V_mbc with
V_mbc = Σ_{j+k−1=m} U_jb U_kc (a convolution of two basis columns): O(r² n²) instead of O(n³).
"""
function reciprocal_core(t, U)
    n, r = size(U); h = t[2] - t[1]
    K = 1 ./ (1 .+ t .+ ((0:2n-2) .* h)')
    V = zeros(2n - 1, r, r)
    @inbounds for c in 1:r, b in 1:r, k in 1:n
        u = U[k, c]
        for j in 1:n
            V[j+k-1, b, c] += U[j, b] * u
        end
    end
    return reshape(U' * K * reshape(V, 2n - 1, r * r), r, r, r)
end

"""
    tensor_random(n; r = 7) -> (problem, eigenfactors)

Non-normal Aᵢ = PᵢDᵢPᵢ⁺ (`nonnormal_matrix`) with spectra [−2, −1], [101, 102], [−202, −201];
random RHS factors Ubᵢ = orth(rand(n, r)) and core rand(r, r, r).
(The original paper code wrote the eigenvalue arguments as `100(i−1) .+ (2, 1, (−1)^i)`: the
broadcast also offsets the third, sign argument of `uniform_eigen`, which then SCALES the
eigenvalues, giving [10201, 10302] and [39999, 40198] instead.  Here the offset is applied to
the two interval ends only.)
"""
function tensor_random(n; r = 7)
    rng = MersenneTwister(1)
    eigargs = ((2, 1, -1), (102, 101, 1), (202, 201, -1))     # (smax, smin, sign) per mode
    ops = [nonnormal_matrix(rng, n, a) for a in eigargs]
    Ub = ntuple(_ -> Matrix(qr(rand(rng, n, r)).Q), 3)
    Gb = rand(rng, r, r, r)
    p = TuckerProblem(ntuple(i -> ops[i][1], 3), Ub, Gb, ntuple(i -> extrema(ops[i][3]), 3))
    return p, [(P = o[2], d = o[3], iP = o[4]) for o in ops]
end
