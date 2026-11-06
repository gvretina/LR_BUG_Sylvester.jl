# Check that every solve allocates nothing: for each method and problem, build a fresh solver
# (setup allocates) and measure the bytes allocated by its first solve.
#   julia --project=. scripts/check_allocations.jl [n2d = 1024] [n3d = 128]
include(joinpath(@__DIR__, "common.jl"))

n2 = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 1024
n3 = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 128

function bytes_of_solve(s, p)
    local st
    b = @allocated(st = solve!(s, p; tol = TOL, theta = THETA))
    return st, b
end

for (dim, n, methods) in (("matrix", n2, ("BUG", "RKSM")), ("tensor", n3, ("BUG", "TBRK"))),
    case in ("laplacian", "convection_diffusion", "random")
    p, _ = dim == "matrix" ? matrix_problem(case, n) : tensor_problem(case, n)
    for method in methods
        build = dim == "matrix" ? build_matrix : build_tensor
        bytes_of_solve(build(method, case, p), p)                # compile
        st, b = bytes_of_solve(build(method, case, p), p)
        @printf("%-6s %-20s %-4s n=%-5d %-10s %2d its  residual %.3e   solve allocated %d bytes\n",
                dim, case, method, n, st.status, st.iters, st.residual, b)
    end
end
