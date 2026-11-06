# Time, iterations and ranks vs n: one row per n (`measure` in common.jl).
#   julia --project=. scripts/time_vs_n.jl <matrix|tensor> <case> <BUG|RKSM|TBRK> n1,n2,...
# Appends to results/<dim>/<case>/data/time_vs_n_<method>.csv with columns
#   n, status, iters, rank, workdim, residual, time
include(joinpath(@__DIR__, "common.jl"))

dim, case, method = ARGS[1], ARGS[2], ARGS[3]
file = joinpath(datadir(dim, case), "time_vs_n_$(method).csv")

for n in parse.(Int, split(ARGS[4], ","))
    p, _ = dim == "matrix" ? matrix_problem(case, n) : tensor_problem(case, n)   # problem data: not timed
    m = measure(dim, case, method, p)
    append_row(file, "n", n, m)
    @printf("%s %s %s n=%d: %s, %d its, rank %d, residual %.2e, %.3f s\n", dim, case, method, n, m.status,
            m.iters, m.rank, m.residual, m.time)
    flush(stdout)
end
