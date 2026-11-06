# Experiments

The numbers in brackets refer to the [references](index.md#References) on the Home page.

## Equations and methods

| | equation | BUG | compared with |
|---|---|---|---|
| matrix | `A X + X Bᵀ = Cu Cs Cvᵀ` | `MatrixBUG` (`src/matrix_bug.jl`) | `RKSM` (`src/rksm.jl`): port of Casulli & Robol's `rk_adaptive_sylvester.m` [4] |
| tensor | `∑ᵢ₌₁ᵈ X ×ᵢ Aᵢ = B`, `B = Gb ×₁ Ub₁ ⋯ ×_d Ub_d` (d = 3) | `TuckerBUG` (`src/tucker_bug.jl`) | `TBRK` (`src/tbrk.jl`): port of Casulli's `Tuck_Sylvester_adaptive.m` [5] |

The tensor code (Tucker BUG, TBRK, the core solver, the residual) works for any number of modes N; the paper uses N = 3. Small tensors are flat buffers with explicit dims, handled by `n_mode_product!` and `matricize!` in `src/tensors.jl`. `tensor_laplacian(n; N = 4)` gives a 4-D test problem.

Each file opens with the algorithm in a few lines of comments. The code then follows those steps in the same order. The [Linear solves](linear_solves.md) page explains every linear system the methods solve, how it is solved, and why.

**Shared by all methods** (`src/truncation.jl`, `src/residual.jl`):
- **Truncation.** Keep the smallest rank whose ℓ² tail of singular values is ≤ θ·‖core‖_F. For Tucker this is applied per mode.
- **Stopping.** The relative residual ‖𝓛(Y) − B‖_F / ‖B‖_F of the approximation the method *returns* must be ≤ tol, or that residual must stagnate (|res_k − res_{k−1}| ≤ 0.01·res_k). For RKSM and TBRK, the stagnation must come from the truncation: their untruncated Galerkin residual must be ≤ 0.1·res.
- **Residual computation.** The residual is computed the same way for every method, from QRs of the low-rank factors, without forming Y.
- **Tolerances.** θ = 1e-14 (truncation) and tol = 1e-10 (stopping), set independently (`THETA`, `TOL` in `scripts/common.jl`).

**Inner solvers** (`src/inner_solvers.jl`). The same kind is used for every method on a given problem:
- **Laplacian:** LDLᵀ of the tridiagonal matrix (`BandedLDLt`, using LinearAlgebra's `ldlt!`), O(n) per right-hand side.
- **Convection-diffusion:** LU with partial pivoting of the non-symmetric tridiagonal matrix (`BandedLU`, using LAPACK's `gtsv` from LinearAlgebra), O(n) per right-hand side. BUG's K-step uses the complex Schur form of the small matrix M instead of its eigendecomposition.
- **Random, BUG:** `BlockKrylovKStep`. Every BUG K-step right-hand side lies in span(C), so one block Krylov basis of K_m(A, C) per operator serves all K-steps by Galerkin projection.
- **Random, RKSM/TBRK:** restarted GMRES from Krylov.jl (`GMRES`) for their shifted solves.
- **Tolerance:** 0.01·tol for both Krylov solvers.
- [Linear solves](linear_solves.md) derives every one of these solves.

**Small dense problems** (`src/small_solvers.jl`):
- `solve_sylvester!` (Bartels-Stewart plus one refinement step) for the matrix S-step, RKSM's projected equation and the Galerkin K-steps.
- `solve_core!` (eigendecompositions, or complex Schur forms for non-normal operators) for the 3-D cores.
- `mode_svd!` for the HOSVD truncation.

## Problems (`src/problems.jl`)

The random and Laplacian problems are generated from `MersenneTwister(1)`. The 3-D convection-diffusion problem is the test of the paper that introduced TBRK [5, Section 6], recreated in `src/problems.jl` and checked against its published data file (`Convection-Diffusion d=3 n=1024.mat`): A₂, A₃ and the right-hand side agree to rounding. A₁ differs on purpose: the paper states ν = 0.1, but the file has ν = 0.01 in the convective mode, because its code multiplies that mode by ν twice. We follow the paper. The 2-D problem is the same one with a dimension removed. Both are run at n = 4096 (the paper used n = 1024 in 3-D).

| | operators | right-hand side |
|---|---|---|
| `matrix_laplacian(n)` | `A = B` = 1-D Dirichlet FD Laplacian on [0, 4π) | rank-7 smooth cosine field |
| `matrix_convdiff(n)` | the 3-D problem below with one dimension fewer: `A` convective, `B` diffusive | f = 1/(1+x+y), rank 8 |
| `matrix_random(n)` | non-normal `P D P⁻¹`, spec(A) = [12, 13], spec(B) = [−2, −1] | `rand(n,7)·rand(7,n)` |
| `tensor_laplacian(n)` | the 1-D Laplacian in every mode | multilinear rank (7, 7, 7), trig basis |
| `tensor_convdiff(n)` | Casulli's TBRK test (Section 6, Figure 3): νΔu + w·∇u, ν = 0.1, w = (1 + (x+1)²/4, 0, 0), central differences: A₁ = −ν·tridiag(−1,2,−1)/h² + diag(w)·tridiag(−1,0,1)/(2h), A₂ = A₃ diffusion only | f = 1/(1+x+y+z), multilinear rank (10, 10, 10) at n = 4096 |
| `tensor_random(n)` | non-normal, spec = [−2, −1], [101, 102], [−202, −201] | `orth(rand(n,7))`, core `rand(7,7,7)` |

These are the intended spectra of the 3-D random problem. The original paper code wrote the arguments as `100(i−1) .+ (2, 1, (−1)^i)`; that broadcast also offsets the sign argument of `uniform_eigen` (101 and 199 instead of ±1), which scales the eigenvalues to [10201, 10302] and [39999, 40198] (see the docstring of `tensor_random`).

## Reproducing the figures

```bash
git clone https://github.com/gvretina/LR_BUG_Sylvester.jl
cd LR_BUG_Sylvester.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
bash scripts/run_all.sh          # all data, then the figures (≈ 1 h; the cached references save ≈ 45 min after the first time)
```

`run_all.sh` runs two lanes, each on 4 physical cores of one NUMA node with 4 BLAS threads (`numactl`). Nothing else should run on the machine while it does.

**Results layout.** `results/<matrix|tensor>/<laplacian|convection_diffusion|random>/` holds `data/` (the scripts' output and the cached exact references) and `figures/` (PDF, EPS, PNG). Figures that combine cases (accuracy vs time) are in `results/<matrix|tensor>/figures/`, and the one that combines both dimensions in `results/figures/`; the run logs are in `results/logs/`. The individual scripts are:

| script | produces (in `data/`) | figures |
|---|---|---|
| `per_case.jl <matrix\|tensor> <case> n` | `per_case_n<n>.jls`: residual vs iterations; matrix case also singular values of X and of both solutions | `fig_matrix_<case>_n{4096,16384}`, `fig_tucker_<case>_n{512,4096}`; convection-diffusion only `fig_matrix_convection_diffusion_n4096`, `fig_tucker_convection_diffusion_n4096` |
| `error_vs_time.jl <matrix\|tensor> <case> n` | `error_vs_time_n<n>.jls`: error (matrix) or residual (tensor) vs wall-clock time; point k = one complete run stopped after k iterations | `fig_matrix_error_vs_time`, `fig_tensor_error_vs_time` (Laplacian and random, both n); `fig_error_vs_time` in `results/figures/` (matrix n = 16384 and tensor n = 4096); one figure per case at its largest n: `fig_<dim>_<case>_error_vs_time_n<n>` |
| `time_vs_n.jl <matrix\|tensor> <case> <method> n1,n2,...` | `time_vs_n_<method>.csv`: time, iterations, ranks and residual vs n (convection-diffusion: one n, timings only) | `fig_comparison_2d`, `fig_comparison_3d`, `fig_comparison_3d_random` |
| `plot.jl` | - | all of the above |
| `check_allocations.jl` | bytes allocated by each solve (expected: 0) | - |

**Timing.** Every time is one complete run: inner solvers, plus solver workspace, plus solve (`build_*` + `solve!` in `scripts/common.jl`). The setup is part of the method. Each time is the minimum over at least 3 runs, after a warm-up run that compiles. Each timed run starts after a full garbage collection and runs with the collector disabled (`src/timing.jl`). Problem generation and the exact references are not timed.

**Exact references** (`src/reference.jl`), used only for the errors and the truncated-exact levels in the figures, never timed:
- Laplacians: the discrete sine transform. Random problems: the eigenfactors, plus iterative refinement against the stored operators. Convection-diffusion (2-D): one big K-step with the symmetric diffusive B = Q M Qᵀ, i.e. n tridiagonal LU solves (`reference_convdiff`).
- They are computed dense once, then stored small. A matrix solution is kept as low-rank factors, X = U diag(σ) Vᵀ to 1e-14·‖X‖_F (rounding level). The factors come from a randomised SVD, which takes seconds instead of a 20-minute dense SVD at n = 16384. The figures' errors are ‖X_r − Y‖_F/‖X_r‖_F, against X_r = the exact solution truncated with θ (the methods' rule), computed from the factors (`relative_error`). A tensor solution is kept only as its truncation, the only part the figures use. The dense n³ tensor is formed only for n ≤ 512; at n = 4096 the figures show the residual only.
- **Cache** (`results/<dim>/<case>/data/reference_n<n>.h5`, 27 MB in all, made by `exact_reference` and `exact_truncation` in `scripts/common.jl`):
  - The first run computes the references (≈ 45 min, mostly the dense n = 16384 random solution). Later runs reuse them.
  - Each file stores a fingerprint of the problem data it was made from (attribute `problem_key`). If a problem's generator changes, it is recomputed automatically. If you change `src/reference.jl` itself, delete those files by hand.
  - The files are plain HDF5, readable from any language. The datasets are `U`, `sigma`, `V`, `tail` (matrix) and `U1`, `U2`, `U3`, `G` (tensor), and the attribute `description` says what they are. Julia writes column-major, so h5py/NumPy shows each array transposed (e.g. `U` as k × n).
