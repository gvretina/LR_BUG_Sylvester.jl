# Replicating “A Low-Rank BUG Method for Sylvester-Type Equations”

This repository provides a Julia implementation of the algorithms and numerical experiments presented in  
*A Low-Rank BUG Method for Sylvester-Type Equations*.  

---

## 📘 Overview
Sylvester-type equations appear in control theory, model reduction, and differential equations, often involving very large, sparse matrices.  
This project reproduces and illustrates the **Low-Rank BUG method**, which iteratively constructs a low-rank approximation to the solution by following the ideas of the BUG integrator as introduced by G.Ceruti and C.Lubich in [this paper](https://doi.org/10.1007/s10543-021-00873-0).

The implementation follows the algorithmic descriptions in the paper as closely as possible and replicates the numerical experiments used to evaluate convergence.

---

## 🏗️ Implementation Notes
- Written entirely in **Julia**, using standard linear algebra routines and sparse matrix support.  
- Reproduces all numerical experiments and plots from the paper.  
- Unfortunately it is part of a WIP, and there are a lot of references towards TTNs, all implementations are included.  


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gvretina.github.io/LR_BUG_Sylvester.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gvretina.github.io/LR_BUG_Sylvester.jl/dev/)
[![Build Status](https://github.com/gvretina/LR_BUG_Sylvester.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvretina/LR_BUG_Sylvester.jl/actions/workflows/CI.yml?query=branch%3Amain)
