[![Build Status](https://github.com/gvretina/LR_BUG_Sylvester.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvretina/LR_BUG_Sylvester.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Replicating “A Low-Rank BUG Method for Sylvester-Type Equations”

This repository provides a Julia implementation of the algorithms and numerical experiments presented in  
*A Low-Rank BUG Method for Sylvester-Type Equations*.  

---

## 📘 Overview
Sylvester-type equations appear in control theory, model reduction, and differential equations, often involving very large, sparse matrices.  
This project reproduces and illustrates the **Low-Rank BUG method**, which iteratively constructs a low-rank approximation to the solution by following the ideas of the BUG integrator as introduced by G.Ceruti and C.Lubich in [this work](https://doi.org/10.1007/s10543-021-00873-0).

The implementation follows the algorithmic descriptions in the paper as closely as possible and replicates the numerical experiments used to evaluate convergence.

---

## 🏗️ Implementation Notes
- Written entirely in **Julia**, using standard linear algebra routines and sparse matrix support.  
- Reproduces all numerical experiments and plots from the paper.  
- Unfortunately it is part of a larger project which is still WIP, so the generalizing concept of Tree Tensor Networks (TTNs) is used.

## Environment setup
To reproduce the exact environment, instantiate the Julia project:

```julia
julia --project=.
using Pkg
Pkg.instantiate()
```

This installs the packages recorded in `Project.toml` and `Manifest.toml`.

---

## Running the experiments

From the Julia REPL you can run the full experiment suites:

```julia
using LR_BUG_Sylvester

run_all_matrix()
run_all_tucker()
```

To run a specific example, call the example functions. The available problem types are:

- `"random"`
- `"laplacian_dirichlet"`

Example:

```julia
example_matrix(n=2^7,problem="laplacian",mode=:adaptive)
example_tucker(n=2^7,d=3,problem_name="laplacian_dirichlet")
```

Each example builds the specified test problem, runs the Low-Rank BUG method, and plots to the `results/` directory.
