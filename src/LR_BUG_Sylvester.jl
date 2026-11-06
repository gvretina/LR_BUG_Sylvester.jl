"""
    LR_BUG_Sylvester

Code for the numerical experiments of "A Low-Rank BUG Method for Sylvester-Type Equations":
the rank-adaptive BUG method (matrix and Tucker) and the two rational Krylov methods it is
compared with (RKSM, Casulli-Robol; TBRK, Casulli), on the paper's test problems.

Every solver allocates all its memory when it is constructed; `solve!` allocates nothing.
"""
module LR_BUG_Sylvester

using LinearAlgebra, SparseArrays, Random, Printf
using Krylov: GmresWorkspace, gmres!, solution
import FFTW

# Large scratch arrays are left uninitialised (they are always written before being read);
# zeroing them dominated the setup time.
scratch(dims...) = Array{Float64}(undef, dims...)

include("lapack.jl")
include("problems.jl")
include("truncation.jl")
include("tensors.jl")
include("small_solvers.jl")
include("residual.jl")
include("krylov_basis.jl")
include("inner_solvers.jl")
include("matrix_bug.jl")
include("tucker_bug.jl")
include("rksm.jl")
include("tbrk.jl")
include("reference.jl")
include("timing.jl")

export MatrixProblem, TuckerProblem, matrix_laplacian, matrix_random, tensor_laplacian, tensor_random
export matrix_convdiff, tensor_convdiff
export BandedLDLt, BandedLU, GMRES, BlockKrylovKStep
export MatrixBUG, TuckerBUG, RKSM, TBRK, solve!, bug_factors, tucker_factors, rksm_factors, tbrk_factors
export truncation_rank, relative_residual, relative_error
export reference_laplacian, reference_random, reference_convdiff, factor_reference, truncate_reference
export timed_nogc, min_time

end
