# Accuracy vs wall-clock time.  Point k of a method's curve = ONE COMPLETE RUN (inner solvers +
# workspace + solve) stopped after k iterations, timed by `min_time` (GC excluded), with
#   matrix: the error ‖X_r − Y‖_F / ‖X_r‖_F against the exact solution truncated with θ,
#   tensor: the relative residual ‖𝓛(Y) − B‖_F / ‖B‖_F (no exact solution at n = 4096).
#   julia --project=. scripts/error_vs_time.jl <matrix|tensor> <laplacian|random> <n>
# Writes results/<dim>/<case>/data/error_vs_time_n<n>.jls.
include(joinpath(@__DIR__, "common.jl"))

dim, case, n = ARGS[1], ARGS[2], parse(Int, ARGS[3])

if dim == "matrix"
    p, f = matrix_problem(case, n)
    F = exact_reference(case, n, p, f); f = nothing
    Ur, Vr, rX = truncate_reference(F, THETA)
    Xr = (Ur, F.sigma[1:rX], Vr)
    methods = ("BUG", "RKSM")
    run_method = (method, maxit) -> run_matrix(method, case, p; maxit)
    accuracy = s -> relative_error(Xr, factors(s)...)
else
    p, _ = tensor_problem(case, n)
    methods = ("BUG", "TBRK")
    run_method = (method, maxit) -> run_tensor(method, case, p; maxit)
    accuracy = s -> relative_residual(p, factors(s)...)
end

function curve(method)
    K = run_method(method, typemax(Int))[1].iters     # also compiles everything
    t = [0.0]; y = [1.0]                              # k = 0: Y = 0
    for k in 1:K
        local s
        push!(t, min_time(() -> (s = run_method(method, k)[2])))
        push!(y, accuracy(s))
    end
    @printf("%s: %d its, final %.2e, %.3f s\n", method, K, y[end], t[end]); flush(stdout)
    return (; t, y)
end

curves = Dict(method => curve(method) for method in methods)
serialize(joinpath(datadir(dim, case), "error_vs_time_n$(n).jls"), (; dim, case, n, tol = TOL, theta = THETA, curves))
