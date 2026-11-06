# Dense solvers for the SMALL projected problems (sizes ≲ a few hundred), shared by all methods:
#
#   SmallSylvester : A S + S Bᵀ = C          (matrix S-step, RKSM's projected equation, K-step Galerkin)
#   CoreSolver     : Σᵢ Y ×ᵢ Hᵢ = C (N-D)     (Tucker BUG core, TBRK's projected equation)
#   HOSVDWork      : left singular vectors of the matricizations of an N-D core (truncation)

# -- A S + S Bᵀ = C : Bartels-Stewart + one step of iterative refinement ----------

struct SmallSylvester
    TA::Matrix{Float64}; ZA::Matrix{Float64}         # A = ZA TA ZAᵀ (real Schur form)
    TB::Matrix{Float64}; ZB::Matrix{Float64}         # B = ZB TB ZBᵀ
    T::Matrix{Float64}; R::Matrix{Float64}           # p × q scratch
    wr::Vector{Float64}; wi::Vector{Float64}; work::Vector{Float64}
end

SmallSylvester(p, q) = SmallSylvester(scratch(p, p), scratch(p, p), scratch(q, q), scratch(q, q),
                                      scratch(p, q), scratch(p, q), zeros(max(p, q)), zeros(max(p, q)),
                                      zeros(gees_lwork(max(p, q))))

"""
    solve_sylvester!(ws, S, A, B, C) -> S

S = solution of A S + S Bᵀ = C (A: p × p, B: q × q, C: p × q; A, B, C are not modified).
Real Schur forms of A and B, LAPACK's `trsyl` on the transformed (triangular) equation, then one refinement step
S += solve(C − A S − S Bᵀ): the projected matrices are formed in floating point (e.g. UᵀAU),
so they are slightly non-normal and plain Bartels-Stewart can lose digits in the smooth modes;
one step restores full accuracy.
"""
@views function solve_sylvester!(ws::SmallSylvester, S, A, B, C)
    p, q = size(C)
    TA = ws.TA[1:p, 1:p]; ZA = ws.ZA[1:p, 1:p]
    TB = ws.TB[1:q, 1:q]; ZB = ws.ZB[1:q, 1:q]
    copyto!(TA, A); gees!(TA, ws.wr[1:p], ws.wi[1:p], ZA, ws.work)
    copyto!(TB, B); gees!(TB, ws.wr[1:q], ws.wi[1:q], ZB, ws.work)
    T = ws.T[1:p, 1:q]
    mul!(T, ZA', C); mul!(S, T, ZB)                  # S ← ZAᵀ C ZB
    triangular_sylvester!(TA, TB, S)
    mul!(T, ZA, S); mul!(S, T, ZB')                  # S ← ZA S ZBᵀ
    R = ws.R[1:p, 1:q]                         # refinement
    copyto!(R, C); mul!(R, A, S, -1.0, 1.0); mul!(R, S, B', -1.0, 1.0)
    mul!(T, ZA', R); mul!(R, T, ZB)
    triangular_sylvester!(TA, TB, R)
    mul!(T, ZA, R); mul!(S, T, ZB', 1.0, 1.0)
    return S
end

"""S ← the solution X of TA X + X TBᵀ = S, for real Schur forms TA, TB (LAPACK dtrsyl; it
returns the solution times a factor `scale` ≤ 1 that guards against overflow)."""
function triangular_sylvester!(TA, TB, S)
    _, scale = LAPACK.trsyl!('N', 'T', TA, TB, S)
    scale != 1 && rmul!(S, inv(scale))
    return S
end

# -- Σᵢ Y ×ᵢ Hᵢ = C (N-D, Hᵢ: dᵢ × dᵢ) ---------------------------------------

struct CoreSolver{N}
    general::Bool
    X1::Vector{Float64}; X2::Vector{Float64}                 # real Kc^N ping-pong buffers
    E::NTuple{N,Matrix{Float64}}; λ::NTuple{N,Vector{Float64}}; syevwork::Vector{Float64}
    Z1::Vector{ComplexF64}; Z2::Vector{ComplexF64}           # complex Kc^N buffers (general only)
    T::NTuple{N,Matrix{ComplexF64}}; Q::NTuple{N,Matrix{ComplexF64}}; QH::NTuple{N,Matrix{ComplexF64}}
    w::Vector{ComplexF64}; zwork::Vector{ComplexF64}; rwork::Vector{Float64}
end

"""
    CoreSolver(Kc, N; general)

Workspace for N-D cores up to Kc × ⋯ × Kc.  `general = false`: symmetric Hᵢ (eigendecompositions);
`general = true`: any Hᵢ (complex Schur forms).
"""
function CoreSolver(Kc::Int, N::Int; general::Bool)
    Kg = general ? Kc : 1
    CoreSolver(general, scratch(Kc^N), scratch(Kc^N),
               ntuple(_ -> zeros(Kc, Kc), N), ntuple(_ -> zeros(Kc), N), zeros(syev_lwork(Kc)),
               Vector{ComplexF64}(undef, Kg^N), Vector{ComplexF64}(undef, Kg^N),
               ntuple(_ -> zeros(ComplexF64, Kg, Kg), N), ntuple(_ -> zeros(ComplexF64, Kg, Kg), N),
               ntuple(_ -> zeros(ComplexF64, Kg, Kg), N), zeros(ComplexF64, Kg),
               zeros(ComplexF64, zgees_lwork(Kg)), zeros(Kg))
end

"""
    solve_core!(cs, H, d, C, Y) -> Y

Y (flat, dims d) = solution of Y ×₁ H[1] + ⋯ + Y ×_N H[N] = C (flat).
Symmetric Hᵢ = Eᵢ Λᵢ Eᵢᵀ:  Y = (C ×ᵢ Eᵢᵀ ./ (λ₁ + ⋯ + λ_N)) ×ᵢ Eᵢ.
General Hᵢ = Qᵢ Tᵢ Qᵢᴴ (complex Schur):  C̃ = C ×ᵢ Qᵢᴴ, back-substitution over the
triangular Tᵢ, Y = Re(Ỹ ×ᵢ Qᵢ).
"""
@views function solve_core!(cs::CoreSolver{N}, H, d::NTuple{N,Int}, C, Y) where {N}
    len = prod(d)
    if !cs.general
        for k in 1:N
            E = cs.E[k][1:d[k], 1:d[k]]; Hk = H[k]
            for j in 1:d[k], i in 1:d[k]
                E[i, j] = 0.5 * (Hk[i, j] + Hk[j, i])            # symmetric up to rounding
            end
            syev!(E, cs.λ[k][1:d[k]], cs.syevwork)
        end
        copyto!(cs.X1[1:len], C[1:len])
        cur, nxt = cs.X1, cs.X2
        for k in 1:N
            n_mode_product!(nxt, cur, d, transpose(cs.E[k][1:d[k], 1:d[k]]), k); cur, nxt = nxt, cur
        end
        @inbounds for (L, I) in enumerate(CartesianIndices(d))
            cur[L] /= sum(ntuple(m -> cs.λ[m][I[m]], Val(N)))
        end
        for k in 1:N
            n_mode_product!(nxt, cur, d, cs.E[k][1:d[k], 1:d[k]], k); cur, nxt = nxt, cur
        end
        copyto!(Y[1:len], cur[1:len])
        return Y
    end
    for k in 1:N
        T = cs.T[k][1:d[k], 1:d[k]]; Q = cs.Q[k][1:d[k], 1:d[k]]; QH = cs.QH[k][1:d[k], 1:d[k]]
        copyto!(T, H[k])
        zgees!(T, cs.w[1:d[k]], Q, cs.zwork, cs.rwork[1:d[k]])
        for j in 1:d[k], i in 1:d[k]
            QH[i, j] = conj(Q[j, i])
        end
    end
    @inbounds for i in 1:len
        cs.Z1[i] = C[i]
    end
    cur, nxt = cs.Z1, cs.Z2
    for k in 1:N
        n_mode_product!(nxt, cur, d, cs.QH[k][1:d[k], 1:d[k]], k); cur, nxt = nxt, cur
    end
    triangular_backsubstitution!(cur, cs.T, d)
    for k in 1:N
        n_mode_product!(nxt, cur, d, cs.Q[k][1:d[k], 1:d[k]], k); cur, nxt = nxt, cur
    end
    @inbounds for i in 1:len
        Y[i] = real(cur[i])
    end
    return Y
end

"""
In place: solve Σᵢ Y ×ᵢ Tᵢ = Y for upper-triangular Tᵢ (Y flat, dims d).  Recursive, as Casulli's
`laplace_tri`: split the largest mode i of the current block I into a lower part `lo` and an upper
part `hi`; the `hi` indices of mode i couple only among themselves, so solve that block first, then
remove its coupling from the `lo` block (Y_lo −= Y_hi ×ᵢ Tᵢ[lo, hi], matrix products), then solve
the `lo` block.  Blocks with all sides ≤ nmin (26 for N = 3, as in `laplace_tri`) are solved
entry by entry (`backsubstitution!`).
"""
function triangular_backsubstitution!(Y, T, d::NTuple{N,Int}; nmin::Int = N == 3 ? 26 : N == 4 ? 18 : 14) where {N}
    return triangular_solve!(Y, T, d, ntuple(m -> 1:d[m], Val(N)), nmin)
end

function triangular_solve!(Y, T, d, I, nmin)
    n = length.(I)
    maximum(n) ≤ nmin && return backsubstitution!(Y, T, d, I)
    i = argmax(n); k = first(I[i]) + n[i] ÷ 2
    lo, hi = first(I[i]):k-1, k:last(I[i])
    triangular_solve!(Y, T, d, Base.setindex(I, hi, i), nmin)
    couple!(Y, T[i], d, i, Base.setindex(I, lo, i), hi)
    triangular_solve!(Y, T, d, Base.setindex(I, lo, i), nmin)
    return Y
end

"""
Y[block I] −= Y[block I, mode-i indices hi] ×ᵢ Tᵢ[lo, hi] (lo = I[i]), one BLAS product per index
of the remaining modes.  Y is viewed as the 3-D array (I_pre, dᵢ, I_post) (as in `n_mode_product!`);
for fixed indices of the modes other than 1 and i the block is then the matrix
Y₃[rows, lo, c]: rows = the mode-1 range shifted by the fixed modes 2...i−1, c = the index of the
fixed modes after i.  For i = 1 the matrix is Y₃[lo, mode-2 range, c] with Y₃ = (d₁, d₂, d₃⋯d_N).
"""
couple!(Y, Ti, d, i, I, hi) = i == 1 ? couple_first!(Y, Ti, d, I, hi) : couple_mode!(Y, Ti, d, i, I, hi)

"""Linear strides of a flat tensor of dims d: mode m moves by d₁⋯dₘ₋₁."""
linear_strides(d::NTuple{N,Int}) where {N} = ntuple(m -> prod(q -> d[q], 1:m-1; init = 1), Val(N))

@views function couple_first!(Y, T1, d::NTuple{N,Int}, I, hi) where {N}     # i = 1
    lo = I[1]; st = linear_strides(d)
    Y3 = reshape(Y[1:prod(d)], d[1], d[2], prod(d) ÷ (d[1] * d[2]))
    w = ntuple(q -> st[q+2] ÷ st[3], Val(N - 2))                         # strides of modes 3...N in Y₃
    for J in CartesianIndices(ntuple(q -> I[q+2], Val(N - 2)))
        c = 1 + sum(ntuple(q -> (J[q] - 1) * w[q], Val(N - 2)); init = 0)
        mul!(Y3[lo, I[2], c], T1[lo, hi], Y3[hi, I[2], c], -1, 1)
    end
    return Y
end

@views function couple_mode!(Y, Ti, d::NTuple{N,Int}, i, I, hi) where {N}  # i ≥ 2
    lo = I[i]; st = linear_strides(d); pre, post = pre_post(d, i)
    Y3 = reshape(Y[1:prod(d)], pre, d[i], post)
    modes = ntuple(q -> q + 1 + (q + 1 ≥ i), Val(N - 2))                 # 2...N without i
    w = ntuple(q -> modes[q] < i ? st[modes[q]] : st[modes[q]] ÷ st[i+1], Val(N - 2))
    for J in CartesianIndices(ntuple(q -> I[modes[q]], Val(N - 2)))
        a, c = 0, 1                                   # row shift (modes before i), slice (modes after i)
        for q in 1:N-2
            modes[q] < i ? (a += (J[q] - 1) * w[q]) : (c += (J[q] - 1) * w[q])
        end
        rows = first(I[1]) + a : last(I[1]) + a
        mul!(Y3[rows, lo, c], Y3[rows, hi, c], transpose(Ti[lo, hi]), -1, 1)
    end
    return Y
end

"""Entry-by-entry back-substitution on the block I of Y (flat, dims d), last index first
(coupling to the rest of Y already removed)."""
function backsubstitution!(Y, T, d::NTuple{N,Int}, I) where {N}
    st = linear_strides(d)
    @inbounds for J in Iterators.reverse(CartesianIndices(I))
        L = 1 + sum(ntuple(m -> (J[m] - 1) * st[m], Val(N)))
        v = Y[L]; den = zero(eltype(Y))
        for m in 1:N
            Tm = T[m]; j = J[m]
            for p in j+1:last(I[m])
                v -= Tm[j, p] * Y[L + (p - j) * st[m]]
            end
            den += Tm[j, j]
        end
        Y[L] = v / den
    end
    return Y
end

# -- HOSVD of an N-D core -------------------------------------------------------

struct HOSVDWork
    W::Matrix{Float64}; R::Matrix{Float64}; U::Matrix{Float64}; VT::Matrix{Float64}
    σ::Vector{Float64}; tau::Vector{Float64}; qrwork::Vector{Float64}
    svdwork::Vector{Float64}; iwork::Vector{BlasInt}
end
HOSVDWork(Kc, N) = HOSVDWork(scratch(Kc^(N-1), Kc), zeros(Kc, Kc), zeros(Kc, Kc), zeros(Kc, Kc), zeros(Kc),
                             zeros(Kc), zeros(qr_lwork(Kc^(N-1), Kc)), zeros(svd_lwork(Kc, Kc)), zeros(BlasInt, 8Kc))

"""
    mode_svd!(hw, P, Y, d, k, θ) -> rank

Left singular vectors of the matricization Mat_k(Y) into P (dₖ × dₖ) and the truncation rank of
its singular values.  Computed from Mat_k(Y)ᵀ = Q R (thin QR) and the SVD of the small R = U Σ Vᵀ:
Mat_k(Y) = V Σ Uᵀ Qᵀ, so the left vectors are V.
"""
@views function mode_svd!(hw::HOSVDWork, P, Y, d::NTuple{N,Int}, k::Int, θ) where {N}
    K = d[k]
    W = hw.W[1:prod(d)÷K, 1:K]
    matricize!(W, Y, d, k; transposed = true)
    R = hw.R[1:K, 1:K]
    rfactor!(R, W, hw.tau[1:K], hw.qrwork)
    σ = hw.σ[1:K]; VT = hw.VT[1:K, 1:K]
    gesdd!(R, σ, hw.U[1:K, 1:K], VT, hw.svdwork, hw.iwork)
    for j in 1:K, i in 1:K
        P[i, j] = VT[j, i]
    end
    return truncation_rank(σ, θ)
end
