[![Build Status](https://github.com/gvretina/LR_BUG_Sylvester.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvretina/LR_BUG_Sylvester.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://gvretina.github.io/LR_BUG_Sylvester.jl/dev/)

# Replicating "A Low-Rank BUG Method for Sylvester-Type Equations"

This repository has the Julia code behind the numerical experiments of *A Low-Rank BUG Method for Sylvester-Type Equations* [1].

## Overview

The low-rank BUG method computes a low-rank approximation of the solution of a matrix or Tucker-tensor Sylvester equation by following the ideas of the BUG integrator for dynamical low-rank approximation [2, 3]. It is compared with two rational Krylov methods:

- **RKSM** [4] for the matrix equation, a port of [rk_adaptive_sylvester](https://github.com/numpi/rk_adaptive_sylvester);
- **TBRK** [5] for the tensor equation, a port of [TBRK-Sylvester](https://github.com/numpi/TBRK-Sylvester).

The test problems are Laplacians, convection-diffusion (the 3-D test of [5, Section 6]) and non-normal random operators, in 2-D (matrix) and 3-D (tensor). Every solver allocates all of its memory when it is constructed, and `solve!` allocates nothing.

The [documentation](https://gvretina.github.io/LR_BUG_Sylvester.jl/dev/) describes the methods, the test problems, every linear solve, and every script and figure.

## Environment setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs the packages recorded in `Project.toml` and `Manifest.toml`.

## Running the experiments

```bash
bash scripts/run_all.sh
```

This computes all the data and then the figures (about 1 h on an Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz; the first run also computes the exact reference solutions, about 45 min more, and caches them). The results go to `results/<matrix|tensor>/<laplacian|convection_diffusion|random>/{data,figures}`. Single scripts can be run on their own, for example

```bash
julia --project=. scripts/per_case.jl matrix laplacian 4096
julia --project=. scripts/plot.jl
```

To use a solver directly:

```julia
using LR_BUG_Sylvester
p = matrix_laplacian(4096)
bug = MatrixBUG(p, BandedLDLt(p.A; rmax = 128), BandedLDLt(p.B; rmax = 128))
stats = solve!(bug, p; tol = 1e-10, theta = 1e-14)
U, S, V = bug_factors(bug)                                  # X ≈ U S Vᵀ
```

## References

1. G. Vretinaris, *A Low-Rank BUG Method for Sylvester-Type Equations*, arXiv:2511.01735, 2025. [doi:10.48550/arXiv.2511.01735](https://doi.org/10.48550/arXiv.2511.01735)
2. G. Ceruti and C. Lubich, *An unconventional robust integrator for dynamical low-rank approximation*, BIT Numer. Math. 62 (2022), pp. 23-44. [doi:10.1007/s10543-021-00873-0](https://doi.org/10.1007/s10543-021-00873-0)
3. G. Ceruti, J. Kusch and C. Lubich, *A rank-adaptive robust integrator for dynamical low-rank approximation*, BIT Numer. Math. 62 (2022), pp. 1149-1174. [doi:10.1007/s10543-021-00907-7](https://doi.org/10.1007/s10543-021-00907-7)
4. A. A. Casulli and L. Robol, *An efficient block rational Krylov solver for Sylvester equations with adaptive pole selection*, SIAM J. Sci. Comput. 46(2) (2024), pp. A798-A824. [doi:10.1137/23M1548463](https://doi.org/10.1137/23M1548463)
5. A. A. Casulli, *Tensorized block rational Krylov methods for tensor Sylvester equations*, IMA J. Numer. Anal. 46(1) (2026), pp. 431-453. [doi:10.1093/imanum/draf001](https://doi.org/10.1093/imanum/draf001)
