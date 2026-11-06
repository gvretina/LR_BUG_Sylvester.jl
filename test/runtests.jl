# Every method on every test problem at a small size: it must converge (relative residual ≤ tol),
# and a second `solve!` with the same workspace must allocate nothing.
using LR_BUG_Sylvester, Test

const TOL, THETA = 1e-10, 1e-14

function check(s, p)
    st = solve!(s, p; tol = TOL, theta = THETA)
    @test st.status === :converged
    @test st.residual ≤ TOL
    @test (@allocated solve!(s, p; tol = TOL, theta = THETA)) == 0
end

@testset "LR_BUG_Sylvester.jl" begin
    @testset "matrix $name" for (name, p, kstep, shifted) in (
            ("laplacian", matrix_laplacian(256), A -> BandedLDLt(A; rmax = 64), A -> BandedLDLt(A)),
            ("convection-diffusion", matrix_convdiff(256), A -> BandedLU(A; width = 64), A -> BandedLU(A; width = 8)),
            ("random", first(matrix_random(256)), nothing, A -> GMRES(A; b = 7, rtol = 0.01TOL)))
        ks(A, C) = kstep === nothing ? BlockKrylovKStep(A, C; rmax = 64, rtol = 0.01TOL) : kstep(A)
        check(MatrixBUG(p, ks(p.A, p.Cu), ks(p.B, p.Cv); rmax = 64), p)
        check(RKSM(p, shifted(p.A), shifted(p.B)), p)
    end

    @testset "tensor $name" for (name, p, general, kstep, shifted) in (
            ("laplacian", tensor_laplacian(64), false, A -> BandedLDLt(A; rmax = 32), A -> BandedLDLt(A)),
            ("laplacian, 4-D", tensor_laplacian(64; N = 4), false, A -> BandedLDLt(A; rmax = 32), A -> BandedLDLt(A)),
            ("convection-diffusion", tensor_convdiff(64), true, A -> BandedLU(A; width = 32), A -> BandedLU(A; width = 16)),
            ("random", first(tensor_random(64)), true, nothing, A -> GMRES(A; b = 7, rtol = 0.01TOL)))
        N = length(p.As)
        ks(k) = kstep === nothing ? BlockKrylovKStep(p.As[k], p.Ub[k]; rmax = 32, rtol = 0.01TOL) : kstep(p.As[k])
        check(TuckerBUG(p, ntuple(ks, N); rmax = 32, general), p)
        check(TBRK(p, ntuple(k -> shifted(p.As[k]), N); general), p)
    end
end
