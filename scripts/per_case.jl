# Data of the per-case figures: residual vs iterations of both methods, and (matrix case) the
# leading singular values of the exact solution X and of both approximations.
#   julia --project=. scripts/per_case.jl <matrix|tensor> <laplacian|random> <n>
# Writes results/<dim>/<case>/data/per_case_n<n>.jls.  The residuals are relative,
# ‖𝓛(Y) − B‖_F / ‖B‖_F (the stopping test), entry 1 = the zero initial guess.
# X_r = the exact solution truncated with the methods' rule; its residual is the best level
# reachable at that truncation.  The tensor reference is dense (n³), so only for n ≤ 512.
include(joinpath(@__DIR__, "common.jl"))

dim, case, n = ARGS[1], ARGS[2], parse(Int, ARGS[3])

if dim == "matrix"
    p, f = matrix_problem(case, n)
    F = exact_reference(case, n, p, f); f = nothing
    Ur, Vr, rX = truncate_reference(F, THETA)
    σX = F.sigma
    LXr = relative_residual(p, Ur, Diagonal(σX[1:rX]), Vr)
    out = Dict{String,Any}()
    for method in ("BUG", "RKSM")
        st, s = run_matrix(method, case, p)
        out[method] = (res = [1.0; s.res_hist[1:st.iters]], σ = diag(factors(s)[2]), status = st.status)
        @printf("%s: %s, %d its, rank %d, residual %.3e\n", method, st.status, st.iters, st.rank, st.residual)
    end
    data = (; dim, case, n, tol = TOL, theta = THETA, σX, θ = THETA * norm([σX; F.tail]), rX, LXr,
            BUG = out["BUG"], RKSM = out["RKSM"])
    @printf("exact: rank %d, truncated-exact residual %.3e\n", rX, LXr)
else
    p, f = tensor_problem(case, n)
    LXr = NaN; rX = (0, 0, 0)
    if n ≤ 512
        T = exact_truncation(case, n, p, f)
        rX = size(T.G)
        LXr = relative_residual(p, ntuple(k -> T[Symbol("U$k")], ndims(T.G)), T.G)
        @printf("exact: ranks %s, truncated-exact residual %.3e\n", rX, LXr)
    end
    f = nothing
    out = Dict{String,Any}()
    for method in ("BUG", "TBRK")
        st, s = run_tensor(method, case, p)
        out[method] = (res = [1.0; s.res_hist[1:st.iters]], ranks = st.ranks, status = st.status)
        @printf("%s: %s, %d its, ranks %s, residual %.3e\n", method, st.status, st.iters, st.ranks, st.residual)
    end
    data = (; dim, case, n, tol = TOL, theta = THETA, rX, LXr, BUG = out["BUG"], TBRK = out["TBRK"])
end
serialize(joinpath(datadir(dim, case), "per_case_n$(n).jls"), data)
