# LR\_BUG\_Sylvester.jl

This is the code behind the numerical experiments of *A Low-Rank BUG Method for Sylvester-Type Equations* [1]. The low-rank BUG method computes a low-rank approximation of the solution of the Sylvester equation

```math
A X + X B^\top = C \quad \text{(matrix)}, \qquad \sum_{i=1}^{d} X \times_i A_i = B \quad \text{(Tucker tensor)},
```

by following the ideas of the BUG integrator for dynamical low-rank approximation [2, 3]. It is compared with two rational Krylov methods:

- **RKSM**, the block rational Krylov solver with adaptive pole selection of Casulli and Robol [4], for the matrix equation;
- **TBRK**, the tensorized block rational Krylov method of Casulli [5], for the tensor equation.

Both are ports of the authors' MATLAB codes ([rk\_adaptive\_sylvester](https://github.com/numpi/rk_adaptive_sylvester), [TBRK-Sylvester](https://github.com/numpi/TBRK-Sylvester)) with their default options. The 3-D convection-diffusion test problem is the one of [5, Section 6].

## Contents

- [Experiments](experiments.md): the methods, the test problems, and how to reproduce every figure.
- [Linear solves](linear_solves.md): every linear system the methods solve, how it is solved, and why.
- [API](api.md): the docstrings.

## Quick start

```julia
using LR_BUG_Sylvester
p = matrix_laplacian(4096)
bug = MatrixBUG(p, BandedLDLt(p.A; rmax = 128), BandedLDLt(p.B; rmax = 128))   # setup
stats = solve!(bug, p; tol = 1e-10, theta = 1e-14)                            # 0 bytes allocated
U, S, V = bug_factors(bug)                                                    # X ≈ U S Vᵀ
```

## References

1. G. Vretinaris, *A Low-Rank BUG Method for Sylvester-Type Equations*, arXiv:2511.01735, 2025. [doi:10.48550/arXiv.2511.01735](https://doi.org/10.48550/arXiv.2511.01735)
2. G. Ceruti and C. Lubich, *An unconventional robust integrator for dynamical low-rank approximation*, BIT Numer. Math. 62 (2022), pp. 23-44. [doi:10.1007/s10543-021-00873-0](https://doi.org/10.1007/s10543-021-00873-0)
3. G. Ceruti, J. Kusch and C. Lubich, *A rank-adaptive robust integrator for dynamical low-rank approximation*, BIT Numer. Math. 62 (2022), pp. 1149-1174. [doi:10.1007/s10543-021-00907-7](https://doi.org/10.1007/s10543-021-00907-7)
4. A. A. Casulli and L. Robol, *An efficient block rational Krylov solver for Sylvester equations with adaptive pole selection*, SIAM J. Sci. Comput. 46(2) (2024), pp. A798-A824. [doi:10.1137/23M1548463](https://doi.org/10.1137/23M1548463)
5. A. A. Casulli, *Tensorized block rational Krylov methods for tensor Sylvester equations*, IMA J. Numer. Anal. 46(1) (2026), pp. 431-453. [doi:10.1093/imanum/draf001](https://doi.org/10.1093/imanum/draf001)
