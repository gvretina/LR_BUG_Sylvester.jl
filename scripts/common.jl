# Shared by the scripts: the package, BLAS threads, the tolerance, the results layout, and ONE
# COMPLETE RUN of each method = inner-solver construction + solver workspace + solve.  Every
# timing in the paper is of such a complete run (setup is part of the method).
#
# Inner solvers (the same kind for every method on a given problem):
#   laplacian            : tridiagonal LDLᵀ                 (BUG K-steps, RKSM/TBRK shifted solves)
#   convection_diffusion : tridiagonal LU (LAPACK gtsv)     (the same)
#   random               : BUG: block Krylov K-step solver (one basis of K_m(A, C) per operator)
#                          RKSM/TBRK: GMRES for the shifted solves
#                          both at relative tolerance 0.01·tol
using LR_BUG_Sylvester, LinearAlgebra, Printf, Serialization, DelimitedFiles, HDF5

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "4")))
const TOL = 1e-10                                     # stopping: relative residual ≤ TOL
const THETA = 1e-14                                   # truncation: ℓ² tail of σ ≤ THETA·‖σ‖

"""results/<dim>/<case>/data: the data (and cached exact references) of one problem."""
datadir(dim, case) = mkpath(normpath(joinpath(@__DIR__, "..", "results", dim, case, "data")))

function matrix_problem(case, n)
    case == "laplacian" && return matrix_laplacian(n), nothing
    case == "convection_diffusion" && return matrix_convdiff(n), nothing
    return matrix_random(n)
end
function tensor_problem(case, n)
    case == "laplacian" && return tensor_laplacian(n), nothing
    case == "convection_diffusion" && return tensor_convdiff(n), nothing
    return tensor_random(n)
end

"""
Fingerprint of a problem's data, stored with each cached reference.  Julia hashes a large array
from a sample of its entries (3 ms for 16384²), so this detects any change to how a problem is
generated, not an edit of single entries.
"""
problem_key(p) = hash(ntuple(i -> getfield(p, i), nfields(p)))

"""
`compute()` (a NamedTuple of arrays and numbers), cached in the HDF5 file `file`: one
dataset per field, and the attributes `problem_key` (recomputed when it differs) and
`description`.  Arrays are column-major, so NumPy / h5py read them transposed.  The file is
written under a temporary name and renamed, so an interrupted run never leaves a partial one.
"""
function cached(compute, file, key, description)
    if isfile(file)
        k, value = h5open(file) do h
            read_attribute(h, "problem_key"), (; (Symbol(d) => read(h[d]) for d in keys(h))...)
        end
        k == key && return value
        println("  cache $file: problem changed, recomputing")
    end
    value = compute()
    h5open(file * ".tmp", "w") do h
        attributes(h)["problem_key"] = key
        attributes(h)["description"] = description
        for (d, v) in pairs(value)
            h[string(d)] = v
        end
    end
    mv(file * ".tmp", file; force = true)
    return value
end

"""The dense exact solution (src/reference.jl): eigenfactors `f` for the random problems."""
dense_reference(case, p, f) = case == "laplacian" ? reference_laplacian(p) :
                              case == "convection_diffusion" ? reference_convdiff(p) : reference_random(p, f)

"""
Exact solution of the matrix problem `p` (eigenfactors `f` for the random one) as low-rank
factors (U, sigma, V, tail), see `factor_reference`: ≈ 10 MB at n = 16384 instead of 2 GB dense.
Cached: computing it costs ≈ 30 min at n = 16384 (random), against seconds for the solves.
"""
exact_reference(case, n, p, f) =
    cached(joinpath(datadir("matrix", case), "reference_n$(n).h5"), problem_key(p),
           "Exact solution X of A X + X B' = Cu Cs Cv' (matrix $case, n = $n): X = U diag(sigma[1:k]) V' " *
           "+ O(1e-14 ||X||_F), k = size(U, 2); sigma = the leading singular values, tail = the 2-norm " *
           "of all the others.") do
        factor_reference(dense_reference(case, p, f))
    end

"""
The exact solution of the tensor problem `p`, truncated with the methods' rule (HOSVD):
(U1, ..., UN, G).  Only this is stored; the dense solution is formed only to compute it.
"""
exact_truncation(case, n, p, f) =
    cached(joinpath(datadir("tensor", case), "reference_n$(n).h5"), hash(THETA, problem_key(p)),
           "Exact solution X of the tensor Sylvester equation ($case, n = $n), HOSVD-truncated " *
           "with theta = $THETA: X_r = G x1 U1 x2 U2 ... xN UN.") do
        U, G, _ = truncate_reference(dense_reference(case, p, f), THETA)
        (; (Symbol("U$k") => U[k] for k in eachindex(U))..., G = G)
    end

"""BUG's K-step solver for the operator A (C = the RHS factor of its mode), workspace for rank ≤ rmax."""
kstep_solver(case, A, C, rmax) =
    case == "laplacian" ? BandedLDLt(A; rmax) :
    case == "convection_diffusion" ? BandedLU(A; width = rmax) : BlockKrylovKStep(A, C; rmax, rtol = 0.01TOL)

"""RKSM's / TBRK's shifted solver for the operator A, blocks of up to b columns."""
shifted_solver(case, A, b) =
    case == "laplacian" ? BandedLDLt(A) :
    case == "convection_diffusion" ? BandedLU(A; width = b) : GMRES(A; b, rtol = 0.01TOL)

"""The solver of a matrix method ("BUG" | "RKSM") with its inner solvers: the setup of a run."""
function build_matrix(method, case, p::MatrixProblem)
    method == "BUG" && return MatrixBUG(p, kstep_solver(case, p.A, p.Cu, 128), kstep_solver(case, p.B, p.Cv, 128); rmax = 128)
    b = size(p.Cs, 1)
    return RKSM(p, shifted_solver(case, p.A, b), shifted_solver(case, p.B, b))
end

"""The solver of a tensor method ("BUG" | "TBRK") with its inner solvers."""
function build_tensor(method, case, p::TuckerProblem)
    general = case != "laplacian"                     # non-symmetric operators
    method == "BUG" && return TuckerBUG(p, ntuple(k -> kstep_solver(case, p.As[k], p.Ub[k], 64), length(p.As)); rmax = 64, general)
    return TBRK(p, ntuple(k -> shifted_solver(case, p.As[k], size(p.Gb, k)), length(p.As)); general)
end

"""One complete run (setup + solve) stopped after at most `maxit` iterations; returns (stats, solver).
(At most 50 iterations for BUG, 100 for RKSM / TBRK: the solvers' default history length.)"""
function run_matrix(method, case, p::MatrixProblem; maxit = typemax(Int))
    s = build_matrix(method, case, p)
    return solve!(s, p; tol = TOL, theta = THETA, maxit), s
end
function run_tensor(method, case, p::TuckerProblem; maxit = typemax(Int))
    s = build_tensor(method, case, p)
    return solve!(s, p; tol = TOL, theta = THETA, maxit), s
end

"""
One row of the sweeps: time = minimum over 3 complete runs (GC excluded) after a warm-up run,
plus status, iterations, rank (max multilinear rank in 3-D), residual and workdim = the largest
basis the method worked with: BUG 2·max rank before augmentation (the augmented [K U]), RKSM
the dimension of its space for A, TBRK the largest of its three spaces.
"""
function measure(dim, case, method, p)
    run_once() = dim == "matrix" ? run_matrix(method, case, p) : run_tensor(method, case, p)
    run_once()                                        # warm-up (compilation)
    local st, s
    time = minimum(timed_nogc(() -> ((st, s) = run_once())) for _ in 1:3)
    if method == "BUG"
        hist = dim == "matrix" ? s.rank_hist[1:st.iters-1] : maximum.(s.rank_hist[1:st.iters-1])
        workdim = 2maximum([dim == "matrix" ? size(p.Cs, 1) : maximum(size(p.Gb)); hist])
    else
        workdim = hasproperty(st, :dims) ? maximum(st.dims) : st.dim
    end
    rank = dim == "matrix" ? st.rank : maximum(st.ranks)
    return (; st.status, st.iters, rank, workdim, st.residual, time)
end

"""Append one sweep row (x = n) to the CSV `file`, writing the header first."""
function append_row(file, xname, x, m)
    isfile(file) || open(io -> println(io, "$xname,status,iters,rank,workdim,residual,time"), file, "w")
    open(io -> println(io, join((x, m.status, m.iters, m.rank, m.workdim, m.residual, m.time), ",")), file, "a")
end

factors(s::MatrixBUG) = bug_factors(s)
factors(s::RKSM) = rksm_factors(s)
factors(s::TuckerBUG) = tucker_factors(s)
factors(s::TBRK) = tbrk_factors(s)
