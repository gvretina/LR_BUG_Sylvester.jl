# How the linear systems are solved

Every method in this code (BUG, RKSM, Tucker BUG, TBRK) is, at heart, a way of replacing one *huge* linear system by many *small* or *structured* ones. This note derives which systems appear (Section 2), explains how each one is solved and why that solver fits (Section 3), and lists, case by case, which solver is used where (Section 4).

---

## 0. Notation

**Matrices.**
- ``I`` is the identity matrix and ``F^\top`` the transpose of ``F``.
- ``\|F\|_F = \big(\sum_{ij} F_{ij}^2\big)^{1/2}`` is the Frobenius norm. It comes from the inner product ``\langle F, G \rangle = \sum_{ij} F_{ij} G_{ij}``.
- A matrix ``U`` has **orthonormal columns** if ``U^\top U = I``. Then ``\|U F\|_F = \|F\|_F`` for every ``F``.
- ``\operatorname{orth}(W)`` is an orthonormal basis of the column space of ``W``, computed by QR: if ``W = QR``, then ``\operatorname{orth}(W) = Q``.
- ``[F\ \ G]`` is the matrix formed by putting ``F`` and ``G`` side by side.
- ``\operatorname{span}(F)`` is the column space of ``F``.
- ``F \oslash G`` is entrywise division.

**Kronecker product.** For ``F \in \mathbb{R}^{p \times q}`` and any ``G``, the block matrix ``F \otimes G = \begin{pmatrix} F_{11} G & \cdots & F_{1q} G \\ \vdots & & \vdots \\ F_{p1} G & \cdots & F_{pq} G \end{pmatrix}``. We use three rules:
```math
(F \otimes G)(H \otimes K) = (FH) \otimes (GK), \qquad (F \otimes G)^\top = F^\top \otimes G^\top, \qquad \operatorname{vec}(G\,X\,F^\top) = (F \otimes G)\operatorname{vec}(X),
```
where ``\operatorname{vec}(X)`` stacks the columns of ``X`` into one long vector.

**3-D arrays (tensors)**, for ``X \in \mathbb{R}^{n_1 \times n_2 \times n_3}``:
- A **mode-1 fibre** is a column ``X_{:,j,l}``. Mode-2 fibres ``X_{i,:,l}`` and mode-3 fibres ``X_{i,j,:}`` are defined the same way.
- The **mode-``k`` product** ``X \times_k F`` multiplies every mode-``k`` fibre by the matrix ``F``:
  ```math
  (X \times_1 F)_{ijl} = \sum_p F_{ip} X_{pjl}, \qquad (X \times_2 F)_{ijl} = \sum_p F_{jp} X_{ipl}, \qquad (X \times_3 F)_{ijl} = \sum_p F_{lp} X_{ijp}.
  ```
  Two rules: ``(X \times_k F) \times_k G = X \times_k (GF)``, and products in *different* modes can be done in any order.
- The **mode-``k`` unfolding** ``\operatorname{Mat}_k(X)`` is the matrix whose columns are the mode-``k`` fibres. For mode 1, column ``(j, l)`` is ``X_{:,j,l}``, with ``j`` running fastest. If ``a < b`` are the two modes other than ``k``, then
  ```math
  \operatorname{Mat}_k(X \times_k F) = F\,\operatorname{Mat}_k(X), \qquad \operatorname{Mat}_k(X \times_a F) = \operatorname{Mat}_k(X)\,(I \otimes F)^\top, \qquad \operatorname{Mat}_k(X \times_b F) = \operatorname{Mat}_k(X)\,(F \otimes I)^\top. \tag{0.1}
  ```
  For example, for ``k = 1`` and ``a = 2``: entry ``(i, (j,l))`` of ``\operatorname{Mat}_1(X)(I \otimes F)^\top`` is ``\sum_p X_{ipl} F_{jp} = (X \times_2 F)_{ijl}``.

---

## 1. The equations

### 1.1 Matrix Sylvester equation

Find ``X \in \mathbb{R}^{n \times m}`` with
```math
A X + X B^\top = C, \qquad C = C_u\, C_s\, C_v^\top. \tag{1.1}
```
- ``A`` is ``n \times n`` and ``B`` is ``m \times m``.
- The right-hand side is given in factored form. ``C_u`` (``n \times 7``) and ``C_v`` (``m \times 7``) have orthonormal columns, and ``C_s`` is a ``7 \times 7`` diagonal matrix. So ``C`` has rank 7, and we never form the ``n \times m`` matrix ``C`` itself.

**As one big linear system.** By the vec rule, ``\operatorname{vec}(AX) = (I \otimes A)\operatorname{vec}(X)`` and ``\operatorname{vec}(XB^\top) = (B \otimes I)\operatorname{vec}(X)``. So (1.1) is the linear system
```math
(I \otimes A + B \otimes I)\,\operatorname{vec}(X) = \operatorname{vec}(C)
```
with an ``nm \times nm`` matrix: ``2.7 \cdot 10^8`` rows for ``n = m = 16384``. We never form it.

**When is there a unique solution?** If ``Aa = \lambda a`` and ``Bb = \mu b``, then ``X = a b^\top`` gives
```math
A a b^\top + a b^\top B^\top = \lambda\, a b^\top + a (Bb)^\top = (\lambda + \mu)\, a b^\top.
```
So every sum ``\lambda_i(A) + \mu_j(B)`` is an eigenvalue of the operator ``X \mapsto AX + XB^\top``. (These are all of them when ``A`` and ``B`` are diagonalisable, and also in general.) The equation has a unique solution exactly when **no sum ``\lambda_i(A) + \mu_j(B)`` is zero**.

**A formula that explains all the methods.** Let ``w`` be an eigenvector of ``B^\top`` with eigenvalue ``\mu`` (so ``\mu`` is also an eigenvalue of ``B``). Multiply (1.1) from the right by ``w``:
```math
A X w + X B^\top w = C w \;\Longrightarrow\; (A + \mu I)\, Xw = Cw \;\Longrightarrow\; Xw = (A + \mu I)^{-1} C_u \big(C_s C_v^\top w\big). \tag{1.2}
```
So the solution ``X`` is built from **shifted solves** ``(A + \mu I)^{-1} C_u`` at the eigenvalues ``\mu`` of ``B``. Equivalently, these are solves with ``(A - \xi I)^{-1}`` at the points ``\xi = -\mu \in -\operatorname{spec}(B)``. All four methods approximate exactly these shifted solves, each in its own way.

### 1.2 Tensor Sylvester equation

Find ``X \in \mathbb{R}^{n \times \cdots \times n}`` (``d`` modes) with
```math
\sum_{i=1}^{d} X \times_i A_i = B, \qquad B = G_b \times_1 U_{b,1} \times_2 \cdots \times_d U_{b,d}. \tag{1.3}
```
The experiments use ``d = 3``, and so does the rest of this page.
- Each ``A_k`` is ``n \times n``.
- The right-hand side is again factored: ``G_b`` is a ``7 \times 7 \times 7`` array (the **core**), and each ``U_{b,k}`` (``n \times 7``) has orthonormal columns.
- The unknown has ``n^3`` entries, ``6.9 \cdot 10^{10}`` at ``n = 4096``, so it cannot even be stored.
- By the same argument as in 1.1, the equation is uniquely solvable when no sum ``\lambda_i(A_1) + \lambda_j(A_2) + \lambda_l(A_3)`` is zero.

### 1.3 Why this is possible: low rank

Every matrix has a singular value decomposition ``X = \sum_k \sigma_k u_k v_k^\top`` with ``\sigma_1 \ge \sigma_2 \ge \dots \ge 0``. Keeping only the first ``r`` terms gives the best rank-``r`` approximation, and its error is ``\big(\sum_{k > r} \sigma_k^2\big)^{1/2}``.

For these equations the ``\sigma_k`` decay extremely fast. Every method therefore keeps its solution as **factors**: ``X \approx U S V^\top`` with ``U`` (``n \times r``) and ``V`` (``m \times r``) having orthonormal columns and ``S`` ``r \times r``. In 3-D the analogue is ``X \approx G \times_1 U_1 \times_2 U_2 \times_3 U_3`` with a small core ``G``.

**Truncation rule** (`truncation_rank` in [src/truncation.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/truncation.jl)): keep the smallest ``r`` with ``\big(\sum_{k>r}\sigma_k^2\big)^{1/2} \le \theta \cdot \big(\sum_k \sigma_k^2\big)^{1/2}``, where ``\theta = 10^{-14}``. This is separate from the stopping tolerance ``\text{tol} = 10^{-10}`` on the relative residual.

### 1.4 The test problems

From [src/problems.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/problems.jl):

| problem | operators | properties |
|---|---|---|
| Laplacian, 2-D (``A = B``) and 3-D (``A_1 = A_2 = A_3``) | ``L = \frac{1}{h^2}\operatorname{tridiag}(1, -2, 1)``, ``h = 4\pi/n`` | **sparse** (tridiagonal), **symmetric**, **negative definite**, **badly conditioned**: eigenvalues from ``\approx -4/h^2`` to ``\approx -1/16``; the ratio is ``\approx 0.4\,n^2`` (``6.8 \cdot 10^6`` at ``n = 4096``) |
| convection-diffusion, 2-D and 3-D | mode 1: ``T = -\nu\frac{1}{h^2}\operatorname{tridiag}(-1, 2, -1) + \operatorname{diag}(w)\,\frac{1}{2h}\operatorname{tridiag}(-1, 0, 1)`` on ``[0, 1]``, ``\nu = 0.1``; other modes diffusion only, as in the TBRK paper (Section 3.7) | **sparse** (tridiagonal), **non-symmetric** (mode 1), **real negative spectrum**, mildly non-normal |
| random, 2-D | ``A = P_A D_A P_A^{-1}``, ``B = P_B D_B P_B^{-1}``, with ``P`` random dense and ``D`` diagonal | **dense**, **non-symmetric**, **well conditioned**: spec``(A)`` = [12, 13], spec``(B)`` = [−2, −1] |
| random, 3-D | the same construction | spec``(A_1)`` = [−2, −1], spec``(A_2)`` = [101, 102], spec``(A_3)`` = [−202, −201] |

These families need different solvers:
- **Laplacian and convection-diffusion:** the structure (3 nonzeros per row) allows a *direct* solve in ``O(n)`` operations. The bad conditioning would make iterative methods slow.
- **Random problems:** a direct solve of a dense system costs ``O(n^3)``. Instead we use *Krylov methods*, which only need products with ``A``. They converge in few steps because the relevant matrices are well conditioned.

---

## 2. Where the linear systems come from

### 2.1 The Galerkin principle

All four methods use the same idea. If the unknown is restricted to a small set of candidates, the equation can usually not be satisfied exactly. So we ask that the **residual** (right-hand side minus left-hand side) be orthogonal to the same small space. This gives as many conditions as unknowns, i.e. a small, square linear system.

### 2.2 Matrix BUG

[src/matrix_bug.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/matrix_bug.jl). The current approximation is ``X \approx U S V^\top`` (rank ``r``). It starts from ``U = C_u`` and ``V = C_v``. One iteration has four steps.

**K-step (new left basis).** Keep ``V`` fixed and look for ``X = K V^\top`` with an unknown ``K \in \mathbb{R}^{n \times r}``. The residual of (1.1) is ``C - A K V^\top - K V^\top B^\top`` (``n \times m``). The Galerkin condition asks that it vanish after multiplication by ``V`` from the right. With ``V^\top V = I``:
```math
C V - A K - K\,(V^\top B^\top V) = 0.
```
Define
```math
M := V^\top B V \quad (r \times r), \qquad R := C V = C_u\,\big(C_s\,(C_v^\top V)\big) \quad (n \times r).
```
Since ``(V^\top B V)^\top = V^\top B^\top V``, the K-step equation is
```math
\boxed{A K + K M^\top = R} \tag{2.1}
```
A few remarks:
- (2.1) is again a Sylvester equation, but very lopsided: ``n \times r`` unknowns, with a large ``A`` on the left and a small ``M`` on the right.
- ``R`` is computed from right to left (``C_v^\top V`` first), so the ``n \times m`` matrix ``C`` is never formed; the cost is ``O((n + m)\cdot 7r)``.
- The columns of ``R`` are linear combinations of the columns of ``C_u``. We will use this in Section 3.6.
- The equation does not involve the old ``S``: it determines ``K`` completely.

**L-step (new right basis).** Symmetrically, keep ``U`` fixed, look for ``X = U L^\top`` with ``L \in \mathbb{R}^{m \times r}``, and require ``U^\top(\text{residual}) = 0``. This gives ``U^\top C - (U^\top A U) L^\top - L^\top B^\top = 0``. Transposed:
```math
B L + L\,(U^\top A U)^\top = C^\top U = C_v\,\big(C_s^\top (C_u^\top U)\big).
```
This is (2.1) again, with ``B`` in place of ``A``, ``M = U^\top A U``, and right-hand side ``R = C^\top U``, whose columns lie in ``\operatorname{span}(C_v)``.

**Augmentation.** ``\hat U = \operatorname{orth}[K\ \ U]`` (``n \times 2r``) and ``\hat V = \operatorname{orth}[L\ \ V]`` (``m \times 2r``). The new bases contain both the new and the old directions.

**S-step (new small core).** Look for ``X = \hat U \hat S \hat V^\top`` and require ``\hat U^\top (\text{residual}) \hat V = 0``:
```math
\boxed{(\hat U^\top A \hat U)\,\hat S + \hat S\,(\hat V^\top B \hat V)^\top = (\hat U^\top C_u)\, C_s\, (\hat V^\top C_v)^\top} \tag{2.2}
```
This is a **small dense** Sylvester equation, ``2r \times 2r``.

**Truncation.** Take the SVD ``\hat S = \Phi \Sigma \Psi^\top`` and keep ``r'`` singular values by the rule of 1.3. Set
```math
U \leftarrow \hat U \Phi_{:,1:r'}, \qquad S \leftarrow \Sigma_{1:r',1:r'}, \qquad V \leftarrow \hat V \Psi_{:,1:r'}.
```

**Products with ``A``.** The method also needs ``AU`` and ``A\hat U`` (for ``U^\top A U`` and for the residual). For the Laplacian these products are cheap (sparse). For the random problems they are dense products, unless the block Krylov solver of 3.6 provides them almost for free.

### 2.3 RKSM: rational Krylov subspace method

[src/rksm.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/rksm.jl). RKSM builds one growing orthonormal basis per side: ``U`` for ``A``, started from ``C_u``, and ``V`` for ``B``, started from ``C_v``. In iteration ``k``:
1. Choose a **pole** ``\xi_k``, a complex number in the region ``-\operatorname{spec}(B)``, which is exactly where (1.2) says the shifted solves are needed. The choice is adaptive (the "sADM" rule in the code) and is not a linear solve, so we skip it here.
2. Let ``u_k`` be the first 7 columns of the newest block of ``U``. Solve the **shifted system**
   ```math
   \boxed{(A - \xi_k I)\, Y_k = u_k} \tag{2.3}
   ```
   and append ``\operatorname{orth}`` of ``Y_k``, orthogonalised against the existing basis, to ``U``.
   - If ``\xi_k`` is real, ``Y_k`` is ``n \times 7``.
   - If ``\xi_k`` is complex, ``Y_k`` is complex. Its conjugate pole ``\bar\xi_k`` gives ``\bar Y_k``, so both are covered by the 14 real columns ``[\operatorname{Re} Y_k\ \ \operatorname{Im} Y_k]``.
3. Do the same for ``B``, with a pole ``\eta_k \in -\operatorname{spec}(A)``.
4. **Galerkin.** Look for ``X = U Y V^\top`` and require ``U^\top(\text{residual})V = 0``:
   ```math
   \boxed{(U^\top A U)\, Y + Y\,(V^\top B V)^\top = (U^\top C_u)\, C_s\, (V^\top C_v)^\top}
   ```
   This is a small dense Sylvester equation, the same type as (2.2). The output is ``Y`` truncated by its SVD.

**Comparison with BUG.** BUG's K-step (2.1) solves, in effect, with *all* the shifts ``\mu_j \in \operatorname{spec}(M)`` at once (Section 3.2 makes this explicit). RKSM adds one well-chosen shift per iteration instead.

### 2.4 Tucker BUG (3-D)

[src/tucker_bug.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/tucker_bug.jl). The current approximation is ``X \approx G \times_1 U_1 \times_2 U_2 \times_3 U_3``:
- ``U_i`` is ``n \times r_i`` with orthonormal columns;
- the core ``G`` is ``r_1 \times r_2 \times r_3``;
- ``M_i := U_i^\top A_i U_i`` (``r_i \times r_i``).

It starts from ``U_i = U_{b,i}``.

**The equation unfolded along mode ``k``.** Let ``a < b`` be the other two modes. Applying ``\operatorname{Mat}_k`` to (1.3) and using (0.1):
```math
A_k\, \operatorname{Mat}_k(X) + \operatorname{Mat}_k(X)\, \mathcal{B}_k^\top = \operatorname{Mat}_k(B), \qquad \mathcal{B}_k := I \otimes A_a + A_b \otimes I. \tag{2.4}
```
This is a matrix Sylvester equation of type (1.1), with ``\mathcal{B}_k`` (``n^2 \times n^2``) in the role of ``B``.

**The current ``X`` in the same form.** By (0.1), ``\operatorname{Mat}_k(X) = U_k\, \operatorname{Mat}_k(G)\, (U_b \otimes U_a)^\top``. Let ``Q_k := \operatorname{orth}(\operatorname{Mat}_k(G)^\top)`` (``r_a r_b \times r_k``), so that ``\operatorname{Mat}_k(G)^\top = Q_k T_k`` with ``T_k`` square. Define
```math
W_k := (U_b \otimes U_a)\, Q_k \quad (n^2 \times r_k).
```
Then ``\operatorname{Mat}_k(X) = (U_k T_k^\top)\, W_k^\top``. The matrix ``W_k`` has orthonormal columns: ``W_k^\top W_k = Q_k^\top (U_b^\top U_b \otimes U_a^\top U_a) Q_k = Q_k^\top Q_k = I``. So the current ``X`` has exactly the 2-D form "``K V^\top``", with ``V := W_k``.

**K-step for mode ``k``.** Apply the 2-D K-step (2.1) to (2.4) with ``V = W_k``:
```math
\boxed{A_k K_k + K_k M_k^\top = R_k}, \qquad M_k := W_k^\top \mathcal{B}_k W_k, \qquad R_k := \operatorname{Mat}_k(B)\, W_k.
```
Both small quantities can be computed without anything of size ``n^2``. By the Kronecker rules,
```math
(U_b \otimes U_a)^\top (I \otimes A_a)(U_b \otimes U_a) = (U_b^\top U_b) \otimes (U_a^\top A_a U_a) = I \otimes M_a,
```
and the same holds for the other term. Therefore
```math
M_k = Q_k^\top\, (I \otimes M_a + M_b \otimes I)\, Q_k \qquad (r_k \times r_k).
```
For the right-hand side, set ``N_i := U_i^\top U_{b,i}`` (``r_i \times 7``). Since ``\operatorname{Mat}_k(B) = U_{b,k} \operatorname{Mat}_k(G_b)(U_{b,b} \otimes U_{b,a})^\top``,
```math
R_k = U_{b,k}\; \operatorname{Mat}_k\!\big(G_b \times_a N_a \times_b N_b\big)\; Q_k \qquad (n \times r_k).
```
The columns of ``R_k`` lie in ``\operatorname{span}(U_{b,k})``, just as in 2-D they lie in ``\operatorname{span}(C_u)``. All three K-steps use the *old* factors.

**Augmentation.** ``\hat U_k = \operatorname{orth}[K_k\ \ U_k]`` (``n \times 2r_k``).

**Core step.** Look for ``X = \hat G \times_1 \hat U_1 \times_2 \hat U_2 \times_3 \hat U_3`` and require the residual to vanish after ``\times_1 \hat U_1^\top \times_2 \hat U_2^\top \times_3 \hat U_3^\top``. For the first term,
```math
(\hat G \times_i \hat U_i) \times_1 A_1 \times_1 \hat U_1^\top \times_2 \hat U_2^\top \times_3 \hat U_3^\top = \hat G \times_1 (\hat U_1^\top A_1 \hat U_1) \times_2 (\hat U_2^\top \hat U_2) \times_3 (\hat U_3^\top \hat U_3) = \hat G \times_1 \hat M_1,
```
and similarly for the others. The result is
```math
\boxed{\hat G \times_1 \hat M_1 + \hat G \times_2 \hat M_2 + \hat G \times_3 \hat M_3 = G_b \times_1 (\hat U_1^\top U_{b,1}) \times_2 (\hat U_2^\top U_{b,2}) \times_3 (\hat U_3^\top U_{b,3})}, \qquad \hat M_i := \hat U_i^\top A_i \hat U_i. \tag{2.5}
```
This is a **small dense 3-D** equation, with at most ``(2r)^3 \approx 2.6 \cdot 10^5`` unknowns.

**Truncation (HOSVD).** For each mode, take the SVD of ``\operatorname{Mat}_k(\hat G)`` and keep its first ``r_k'`` left singular vectors ``P_k`` by the rule of 1.3. Then ``G \leftarrow \hat G \times_1 P_1^\top \times_2 P_2^\top \times_3 P_3^\top`` and ``U_k \leftarrow \hat U_k P_k``.

### 2.5 TBRK: tensorised block rational Krylov

[src/tbrk.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/tbrk.jl). This is the 3-D version of RKSM. It keeps one growing basis ``V_i`` per mode, started from ``U_{b,i}``. Every iteration extends each mode with a shifted solve
```math
\boxed{(A_i - \xi_i I)\, Y = u}
```
(``u`` = the first 7 columns of the newest block). The pole ``\xi_i`` lies in the region where (1.2), applied to the unfolding (2.4), says it is needed: the negated eigenvalues of ``\mathcal{B}_i``, i.e. ``-(\lambda(A_a) + \lambda(A_b))``. The Galerkin condition (as in 2.4) gives the small 3-D equation
```math
\boxed{Y \times_1 H_1 + Y \times_2 H_2 + Y \times_3 H_3 = G_b \times_1 (V_1^\top U_{b,1}) \times_2 (V_2^\top U_{b,2}) \times_3 (V_3^\top U_{b,3})}, \qquad H_i := V_i^\top A_i V_i,
```
and ``X \approx Y \times_1 V_1 \times_2 V_2 \times_3 V_3``, truncated by an HOSVD.

### 2.6 Summary: five kinds of linear system

| system | size | appears in |
|---|---|---|
| (a) **K-step** ``A K + K M^\top = R`` | ``n \times r``; ``M`` is ``r \times r`` | BUG (K- and L-steps), Tucker BUG (one per mode) |
| (b) **shifted solve** ``(A - \xi I)\,Y = u``, real or complex ``\xi`` | ``n \times 7`` | RKSM (two per iteration), TBRK (three per iteration) |
| (c) **small Sylvester** ``\hat A Y + Y \hat B^\top = \hat C`` | ``p \times q``, up to a few hundred | BUG S-step, RKSM Galerkin, inside (a) for the random problems |
| (d) **small 3-D core** ``Y \times_1 H_1 + Y \times_2 H_2 + Y \times_3 H_3 = \hat C`` | up to ``\approx 150^3`` | Tucker BUG core step, TBRK Galerkin |
| (e) **tridiagonal** ``(L + \sigma I)\, x = b`` | ``n`` | inside (a) and (b) for the Laplacian and for convection-diffusion |

---

## 3. How each system is solved

### 3.1 Tridiagonal systems: the Thomas algorithm (= tridiagonal LDLᵀ)

`factor_shifted!` in [src/inner_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/inner_solvers.jl), with `ldlt!` and `ldiv!` from Julia's LinearAlgebra

**Problem.** Solve ``T x = b``, where ``T = L + \sigma I`` is symmetric tridiagonal, with diagonal ``t_i = d_i + \sigma`` (``d_i = L_{ii}``) and off-diagonal ``e_i = L_{i,i+1} = L_{i+1,i}``. Row ``i`` reads
```math
e_{i-1} x_{i-1} + t_i\, x_i + e_i\, x_{i+1} = b_i
```
(without the first term in row 1 and the last in row ``n``).

**Forward sweep (elimination).** We turn every row into the form ``x_i + c_i\, x_{i+1} = y_i``, top to bottom.
- Row 1 divided by ``m_1 := t_1`` gives ``c_1 = e_1/m_1`` and ``y_1 = b_1/m_1``.
- If row ``i-1`` is already ``x_{i-1} = y_{i-1} - c_{i-1} x_i``, substitute this into row ``i``:
  ```math
  \underbrace{(t_i - e_{i-1} c_{i-1})}_{=:\,m_i}\, x_i + e_i\, x_{i+1} = b_i - e_{i-1} y_{i-1}.
  ```
  Divide by ``m_i``:
  ```math
  c_i = \frac{e_i}{m_i}, \qquad y_i = \frac{b_i - e_{i-1} y_{i-1}}{m_i}.
  ```

**Back substitution.** The last row has no ``x_{n+1}``, so ``x_n = y_n``. Then ``x_i = y_i - c_i\, x_{i+1}`` for ``i = n-1, \dots, 1``.

**Cost.** About ``9n`` arithmetic operations for one right-hand side. Compare ``\frac{2}{3}n^3`` for a dense LU.

**Why it is called LDLᵀ.** Write ``T = \mathcal{L} D \mathcal{L}^\top``, with ``\mathcal{L}`` unit lower bidiagonal (ones on the diagonal, ``\ell_i`` below it) and ``D = \operatorname{diag}(m_1, \dots, m_n)``. Multiplying out and comparing entries:
- ``(1,1)``: ``m_1 = t_1``;
- ``(i+1, i)``: ``\ell_i m_i = e_i``;
- ``(i+1, i+1)``: ``\ell_i^2 m_i + m_{i+1} = t_{i+1}``.

So ``\ell_i = e_i/m_i = c_i`` and ``m_{i+1} = t_{i+1} - e_i c_i``. These are exactly the numbers of the forward sweep. The forward sweep therefore computes the factorisation and at the same time solves ``\mathcal{L} z = b`` and ``D y = z`` (here ``z = D y``, and one checks ``y_i = (b_i - \ell_{i-1} z_{i-1})/m_i``). The back substitution solves ``\mathcal{L}^\top x = y``. "Thomas algorithm" is the traditional name, after L. H. Thomas (1949), for this elimination; numerical linear algebra books present it as LU or LDLᵀ of a banded matrix.

**In the code** the two halves are separate calls to Julia's LinearAlgebra:
- `ldlt!(T)` overwrites the `SymTridiagonal` matrix ``T`` with its factors (the ``m_i`` on the diagonal, the ``\ell_i`` off it), in about ``3n`` operations.
- `ldiv!(F, B)` then solves ``\mathcal{L} z = b``, ``D y = z`` and ``\mathcal{L}^\top x = y`` for every column ``b`` of ``B``, overwriting ``B`` with the solution, in about ``5n`` operations per column.

With several right-hand sides and the same shift (RKSM and TBRK solve 7 columns at once) the matrix is factored only once.

**Is it safe without pivoting?** Each pivot equals ``m_i = \det(T_{1:i,1:i}) / \det(T_{1:i-1,1:i-1})``, a ratio of leading principal minors. If ``T`` is **definite** (all eigenvalues of one sign), so is every leading principal submatrix. Then no minor is zero and all ``m_i \ne 0``. For definite matrices LDLᵀ without pivoting is also numerically stable. The Laplacian ``L`` is negative definite, and both kinds of shift used here keep ``T`` negative definite:
- **BUG K-step:** ``\sigma = \lambda_j(M)`` with ``M = V^\top L V`` (Section 3.2). Each eigenvalue of ``M`` is a value ``v^\top L v`` with ``\|v\| = 1``, hence lies between ``\lambda_{\min}(L)`` and ``\lambda_{\max}(L) < 0``. So ``L + \sigma I`` has only negative eigenvalues. In 3-D, ``M_k``'s eigenvalues are sums of two such values, still negative.
- **RKSM/TBRK:** ``\sigma = -\xi`` with ``\xi \in -\operatorname{spec}(\text{other operator}) \subset (0, \infty)``, so again ``\sigma < 0``.

The same code accepts a complex ``\sigma`` (a complex `SymTridiagonal`, `pair_solve!`). That case is only needed for complex poles, which do not occur for the symmetric Laplacian; it is there for completeness.

### 3.2 The K-step for the Laplacian: decouple, then Thomas

`kstep!(::BandedLDLt, ...)` in [src/inner_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/inner_solvers.jl)

**Problem.** System (2.1), ``A K + K M^\top = R``, with ``A`` tridiagonal. Thomas cannot be applied directly, because the term ``K M^\top`` mixes the columns of ``K``: column ``j`` of ``K M^\top`` is ``\sum_l M_{jl}\, k_l``.

**Decouple the columns.** For the Laplacian ``M`` is symmetric: ``M^\top = V^\top L^\top V = V^\top L V = M``. (In floating point it is symmetric only up to rounding, so the code uses ``\frac{1}{2}(M + M^\top)``.) A symmetric matrix has an orthonormal eigenbasis:
```math
M = E \Lambda E^\top, \qquad E^\top E = I, \qquad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_r)
```
(LAPACK `syev`, ``O(r^3)``). Substitute into (2.1) and multiply from the right by ``E``:
```math
A K E + K\,E \Lambda E^\top E = R E \quad\Longrightarrow\quad A\,(KE) + (KE)\,\Lambda = RE,
```
using ``E^\top E = I``. Introduce the **new unknown** ``\tilde K := K E`` and the transformed right-hand side ``\tilde R := R E``:
```math
A \tilde K + \tilde K \Lambda = \tilde R.
```
Now ``\Lambda`` is diagonal, so column ``j`` of ``\tilde K \Lambda`` is just ``\lambda_j \tilde k_j``. Column ``j`` of the equation reads
```math
(A + \lambda_j I)\,\tilde k_j = \tilde r_j, \qquad j = 1, \dots, r.
```
These are ``r`` **independent tridiagonal systems**, each solved by Thomas (3.1) with ``\sigma = \lambda_j``.

**Back to ``K``.** Since ``\tilde K = KE`` and ``E`` is orthogonal, ``K = \tilde K E^\top``.

**Cost.** ``O(r^3)`` for the eigendecomposition, ``O(nr)`` for the ``r`` Thomas solves, and ``O(nr^2)`` for the two products with ``E``. That is linear in ``n``, which is why BUG's time grows only linearly with ``n`` on the Laplacian.

**The connection to (1.2).** The ``\lambda_j`` are eigenvalues of ``V^\top L V``, i.e. approximations to eigenvalues ``\mu`` of the other operator. So the K-step is exactly a set of shifted solves ``(A + \mu I)^{-1}`` at all current approximations of ``\operatorname{spec}(B)`` at once.

The same applies to the tensor K-steps of 2.4: ``M_k = Q_k^\top (I \otimes M_a + M_b \otimes I) Q_k`` is symmetric when ``M_a`` and ``M_b`` are.

### 3.3 Small dense Sylvester equations: Bartels-Stewart

`solve_sylvester!` in [src/small_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/small_solvers.jl)

**Problem.** ``\hat A Y + Y \hat B^\top = \hat C``, with small dense ``\hat A`` (``p \times p``), ``\hat B`` (``q \times q``), ``\hat C`` (``p \times q``) and unknown ``Y`` (``p \times q``). Typical sizes are ``p, q \lesssim 300``. It is used for:
- BUG's S-step (2.2), with ``p = q = 2r``;
- RKSM's Galerkin equation;
- the Galerkin solve inside the block Krylov K-step (3.6).

**Step 1: Schur forms.** Every real square matrix has a **real Schur form** ``F = Z T Z^\top``:
- ``Z`` is orthogonal;
- ``T`` is block upper triangular, with ``1 \times 1`` diagonal blocks for real eigenvalues and ``2 \times 2`` blocks for complex-conjugate pairs.

It is computed by the QR algorithm (LAPACK `gees`) in ``O(p^3)``. Compute ``\hat A = Z_A T_A Z_A^\top`` and ``\hat B = Z_B T_B Z_B^\top``.

**Step 2: transform.** Write the unknown as ``Y = Z_A \tilde Y Z_B^\top``, which defines ``\tilde Y := Z_A^\top Y Z_B``. Substituting, and using ``Z_A^\top Z_A = I`` and ``Z_B^\top Z_B = I``:
```math
Z_A T_A \tilde Y Z_B^\top + Z_A \tilde Y\, T_B^\top Z_B^\top = \hat C.
```
Multiply from the left by ``Z_A^\top`` and from the right by ``Z_B``:
```math
T_A \tilde Y + \tilde Y T_B^\top = \tilde C, \qquad \tilde C := Z_A^\top \hat C Z_B.
```

**Step 3: solve column by column.** Column ``j`` of ``\tilde Y T_B^\top`` is ``\sum_k \tilde y_k (T_B)_{jk}``. Since ``T_B`` is upper triangular, ``(T_B)_{jk} = 0`` for ``k < j``. Column ``j`` of the equation is therefore
```math
T_A \tilde y_j + (T_B)_{jj}\, \tilde y_j + \sum_{k > j} (T_B)_{jk}\, \tilde y_k = \tilde c_j
\quad\Longrightarrow\quad
\big(T_A + (T_B)_{jj} I\big)\, \tilde y_j = \tilde c_j - \sum_{k > j} (T_B)_{jk}\, \tilde y_k.
```
Solve for the **last column first** (``j = q, q-1, \dots, 1``). Each right-hand side then only contains columns already computed. Each step is a **triangular** system, solved by back substitution in ``O(p^2)``.
- Its matrix is singular only if ``\lambda_i(\hat A) + (T_B)_{jj} = 0``: the solvability condition of 1.1 again.
- A ``2 \times 2`` block of ``T_B`` couples two neighbouring columns, which are then solved together. A ``2 \times 2`` block of ``T_A`` is handled inside the triangular solve.

This is LAPACK's `trsyl`.

**Step 4: transform back.** ``Y = Z_A \tilde Y Z_B^\top``.

**Iterative refinement.** The small matrices are themselves rounded (for example ``\hat U^\top A \hat U``), and the transformations add rounding errors, so ``Y`` may have lost a few digits. We compute the residual ``\hat F := \hat C - \hat A Y - Y \hat B^\top`` and solve ``\hat A\,\delta Y + \delta Y\,\hat B^\top = \hat F`` with the *same* Schur forms (steps 2-4). Then we set ``Y \leftarrow Y + \delta Y``. Because the equation is linear, an exact ``\delta Y`` would make ``Y + \delta Y`` exact; in practice one such step restores full accuracy.

**Cost.** ``O(p^3 + q^3)`` for the Schur forms and ``O(p^2 q + p q^2)`` for the rest.

### 3.4 Small dense 3-D cores

`solve_core!` in [src/small_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/small_solvers.jl)

**Problem.** ``Y \times_1 H_1 + Y \times_2 H_2 + Y \times_3 H_3 = \hat C``, with small ``H_i`` (``d_i \times d_i``), right-hand side ``\hat C`` and unknown ``Y`` (``d_1 \times d_2 \times d_3``). This is (2.5) and TBRK's Galerkin equation.

**Symmetric ``H_i`` (Laplacian): diagonalise.** Write ``H_i = E_i \Lambda_i E_i^\top``, with ``E_i`` orthogonal and ``\Lambda_i = \operatorname{diag}(\lambda^{(i)}_1, \lambda^{(i)}_2, \dots)``. Express the unknown in the eigenbases, ``Y = \tilde Y \times_1 E_1 \times_2 E_2 \times_3 E_3``. Using the mode-product rules, the first term becomes
```math
Y \times_1 H_1 = \tilde Y \times_1 (H_1 E_1) \times_2 E_2 \times_3 E_3 = \tilde Y \times_1 (E_1 \Lambda_1) \times_2 E_2 \times_3 E_3 = (\tilde Y \times_1 \Lambda_1) \times_1 E_1 \times_2 E_2 \times_3 E_3,
```
and likewise for modes 2 and 3. Apply ``\times_1 E_1^\top \times_2 E_2^\top \times_3 E_3^\top`` to the whole equation. This removes the ``E``'s, because ``E_i^\top E_i = I``:
```math
\tilde Y \times_1 \Lambda_1 + \tilde Y \times_2 \Lambda_2 + \tilde Y \times_3 \Lambda_3 = \tilde C, \qquad \tilde C := \hat C \times_1 E_1^\top \times_2 E_2^\top \times_3 E_3^\top.
```
A mode product with a diagonal matrix just scales: ``(\tilde Y \times_1 \Lambda_1)_{abc} = \lambda^{(1)}_a \tilde Y_{abc}``. So every entry decouples:
```math
\big(\lambda^{(1)}_a + \lambda^{(2)}_b + \lambda^{(3)}_c\big)\,\tilde Y_{abc} = \tilde C_{abc}.
```
Divide, then transform back with ``Y = \tilde Y \times_1 E_1 \times_2 E_2 \times_3 E_3``. The cost is dominated by the six mode products: ``O(d^4)`` for ``d_i = d``.

**Non-symmetric ``H_i`` (random): triangularise.** One could diagonalise a non-symmetric matrix as ``H = E \Lambda E^{-1}``. But for non-normal matrices ``E`` can be badly conditioned, and all rounding errors would be amplified by ``\|E\|\,\|E^{-1}\|``. So instead we use the **complex Schur form** ``H_i = Q_i T_i Q_i^{H}`` (LAPACK `zgees`), where:
- ``Q_i`` is unitary (``Q_i^H`` is its conjugate transpose and ``Q_i^H Q_i = I``);
- ``T_i`` is upper triangular, with the eigenvalues of ``H_i`` on its diagonal.

The complex form is used because a real Schur form would have ``2 \times 2`` blocks, which are awkward in three dimensions. Exactly as above (with ``Q_i`` for ``E_i`` and ``Q_i^H`` for ``E_i^\top``) the equation becomes
```math
\tilde Y \times_1 T_1 + \tilde Y \times_2 T_2 + \tilde Y \times_3 T_3 = \tilde C, \qquad \tilde C := \hat C \times_1 Q_1^H \times_2 Q_2^H \times_3 Q_3^H.
```
Entry ``(a, b, c)`` reads
```math
\sum_{p \ge a} (T_1)_{ap} \tilde Y_{pbc} + \sum_{q \ge b} (T_2)_{bq} \tilde Y_{aqc} + \sum_{s \ge c} (T_3)_{cs} \tilde Y_{abs} = \tilde C_{abc},
```
because the ``T_i`` are upper triangular. Every term except the three diagonal ones involves an entry with a *larger* index in one position. So
```math
\tilde Y_{abc} = \frac{\tilde C_{abc} - \big(\text{the terms with } p > a,\ q > b,\ s > c\big)}{(T_1)_{aa} + (T_2)_{bb} + (T_3)_{cc}}
```
can be computed going through ``c, b, a`` from the last index down to the first. This is the 3-D version of Bartels-Stewart.

Done entry by entry, this costs ``O(d^4)`` operations in scalar loops. The code (`triangular_backsubstitution!`) does the same work in matrix products instead, as Casulli's MATLAB `laplace_tri` does. It splits the largest mode ``i`` of ``\tilde Y`` into a lower part `lo` and an upper part `hi`. Because ``T_i`` is upper triangular, the `hi` rows involve only `hi` rows, so that block is solved first (recursively). Its contribution to the `lo` rows, ``\tilde Y_{\text{hi}} \times_i (T_i)_{\text{lo},\text{hi}}``, is then subtracted by one matrix product per slice, and the `lo` block is solved. Blocks with all sides ``\le 26`` are solved entry by entry. For a ``189 \times 256 \times 256`` core this is 10 times faster than the scalar loops. Finally ``Y = \tilde Y \times_1 Q_1 \times_2 Q_2 \times_3 Q_3``. The exact ``Y`` is real, so its imaginary part is rounding error and we keep the real part.

### 3.5 GMRES for dense matrices (random problems: RKSM and TBRK)

`GMRES`, `SylvesterOperator`, `shifted_solve!` and `pair_solve!` in [src/inner_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/inner_solvers.jl), using `gmres!` from the package [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)

**Problem.** Here we treat a general equation
```math
\mathcal{A}(K) := A K + K M^\top = R, \qquad K, R \in \mathbb{R}^{n \times w}, \quad M \in \mathbb{R}^{w \times w} \text{ small}, \tag{3.1}
```
with a **dense** ``A``. A direct solve would cost ``O(n^3)``. GMRES only needs products with ``A``.

``\mathcal{A}`` is a linear map on the vector space of ``n \times w`` matrices, and we use the Frobenius inner product on that space. "Krylov space", "orthonormal" and "norm" below are meant in this space; each "vector" is an ``n \times w`` matrix.

**Krylov space.** ``\mathcal{K}_j = \operatorname{span}\{R,\ \mathcal{A}(R),\ \mathcal{A}^2(R),\ \dots,\ \mathcal{A}^{j-1}(R)\}``: all combinations ``\sum_i y_i\, \mathcal{A}^{i-1}(R)`` with scalar coefficients ``y_i``. GMRES picks the ``K \in \mathcal{K}_j`` with the **smallest residual norm** ``\|R - \mathcal{A}(K)\|_F``, for ``j = 1, 2, \dots``

**Arnoldi process (an orthonormal basis of ``\mathcal{K}_j``).** Set ``\beta := \|R\|_F`` and ``V_1 := R/\beta``. For ``j = 1, 2, \dots``:
1. ``W := \mathcal{A}(V_j)`` (one product ``A V_j``, plus ``V_j M^\top``).
2. Orthogonalise: for ``i = 1, \dots, j``, set ``h_{ij} := \langle V_i, W \rangle`` and ``W \leftarrow W - h_{ij} V_i``. (Modified Gram-Schmidt. We ask Krylov.jl to do it a second time, `reorthogonalization = true`, which keeps the basis orthonormal to working accuracy.)
3. ``h_{j+1,j} := \|W\|_F`` and ``V_{j+1} := W / h_{j+1,j}``.

By construction ``\mathcal{A}(V_j) = \sum_{i=1}^{j+1} h_{ij} V_i``. Collect the ``h_{ij}`` in the ``(j+1) \times j`` matrix ``\bar H_j``, which is upper Hessenberg (zero below the first subdiagonal).

**The small least-squares problem.** Write ``K = \sum_{i=1}^{j} y_i V_i`` with coefficients ``y \in \mathbb{R}^j``. Then
```math
R - \mathcal{A}(K) = \beta V_1 - \sum_{l=1}^{j} y_l \sum_{i=1}^{l+1} h_{il} V_i = \sum_{i=1}^{j+1} \big(\beta e_1 - \bar H_j y\big)_i\, V_i,
```
where ``e_1 = (1, 0, \dots, 0)^\top``. Because the ``V_i`` are orthonormal,
```math
\|R - \mathcal{A}(K)\|_F = \|\beta e_1 - \bar H_j y\|_2.
```
So the large minimisation becomes a tiny least-squares problem of size ``(j+1) \times j``. It is solved incrementally: each new column of ``\bar H_j`` is reduced to upper triangular form by one **Givens rotation**, a ``2 \times 2`` rotation that zeroes the subdiagonal entry. The same rotations applied to ``\beta e_1`` leave the current residual norm as the absolute value of its last entry, so it is available at every step at no cost. When we stop, the triangular system for ``y`` is solved by back substitution and ``K = \sum_i y_i V_i``.

**Restart and stopping.**
- After ``m = 40`` steps we form ``K`` and start a new Arnoldi process from the residual ``R - \mathcal{A}(K)`` (`restart = true`, `memory = 40`). This keeps the memory at about 40 basis vectors.
- We stop when the residual norm from the rotations satisfies ``\|R - \mathcal{A}(K)\|_F \le 0.01 \cdot \text{tol} \cdot \|R\|_F = 10^{-12} \|R\|_F`` (`rtol`).

**How the operator is handed to Krylov.jl.** Krylov.jl works with vectors. `SylvesterOperator` therefore stores ``A`` and ``M`` and defines the product for a vector ``x`` of length ``n w``: copy ``x`` into an ``n \times w`` matrix ``X``, compute ``A X + X M^\top``, and copy the result back into a vector. The two copies cost ``O(nw)``, negligible next to the ``2n^2 w`` of the product.

**Cost per step.** One dense product ``A V_j``, i.e. ``2n^2 w`` operations, plus ``O(j\, n w)`` for the orthogonalisation.

**How fast does it converge?** By the argument of 1.1, the eigenvalues of ``\mathcal{A}`` are the sums ``\lambda_i(A) + \mu_l(M)``. When they lie in a region far from 0 compared with its size, the residual shrinks by a roughly constant factor per step, and GMRES needs few steps compared with ``n``. (For non-normal ``A`` the eigenvalues alone do not tell the whole story, but they are a good guide here.) For the random problems all relevant regions are far from 0:
- RKSM, ``A``-space: ``\lambda(A) - \xi \in [12, 13] - [1, 2] = [10, 12]``.
- RKSM, ``B``-space: ``\lambda(B) - \eta \in [-2, -1] + [12, 13] = [10, 12]``.
- TBRK, mode 1: ``\lambda(A_1) - \xi_1 \in [-2, -1] - [99, 101] = [-103, -100]``. Modes 2 and 3 are similar.

**Shifted solves (system (b)).** ``(A - \xi I)\, Y = u`` with real ``\xi`` is (3.1) with ``w = 7`` and ``M = -\xi I``, since ``A Y + Y(-\xi I)^\top = (A - \xi I) Y``.

**Complex shifts in real arithmetic.** For complex ``\xi``, write the unknown as ``y = y_r + i\, y_i`` and set ``a + ib := -\xi`` (``a``, ``b`` real). Then ``(A - \xi I) y = u`` is ``(A + aI + ibI)(y_r + i\, y_i) = u``. Its real and imaginary parts are
```math
A y_r + a\, y_r - b\, y_i = u, \qquad A y_i + b\, y_r + a\, y_i = 0.
```
Put ``Y := [Y_r\ \ Y_i]`` (``n \times 14``), with the real and imaginary parts of all 7 columns side by side. With
```math
M := \begin{pmatrix} aI & -bI \\ bI & aI \end{pmatrix}, \qquad Y M^\top = [Y_r\ \ Y_i] \begin{pmatrix} aI & bI \\ -bI & aI \end{pmatrix} = [\,a Y_r - b Y_i\ \ \ b Y_r + a Y_i\,].
```
So the two equations are exactly ``A Y + Y M^\top = [u\ \ 0]``: an equation of type (3.1) in real numbers.

### 3.6 One Krylov space for all K-steps (random problems: BUG)

`BlockKrylovKStep`, `KrylovBasis` in [src/inner_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/inner_solvers.jl) and [src/krylov_basis.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/krylov_basis.jl)

**Problem.** The K-step (2.1), ``A K + K M^\top = R``, with a dense ``A``. One could use GMRES (3.5), but BUG solves many K-steps, and they share structure:
- In every K-step the columns of ``R`` lie in ``\operatorname{span}(C_u)`` (Section 2.2): ``R = C_u \times (\text{small matrix})``. For the L-step they lie in ``\operatorname{span}(C_v)``, and in 3-D in ``\operatorname{span}(U_{b,k})``.
- From one K-step to the next only ``M`` and the small factor change. The large operator ``A`` and the starting block ``C_u`` stay the same.

**The block Krylov space.** Build an orthonormal basis ``Q = [Q_1\ Q_2\ \cdots\ Q_m]``, ``b = 7`` columns per block, of
```math
\mathcal{K}_m(A, C_u) = \operatorname{span}\{C_u,\ A C_u,\ A^2 C_u,\ \dots,\ A^{m-1} C_u\}.
```
Start with ``Q_1 := \operatorname{orth}(C_u)``. Each new block is ``Q_{j+1} := \operatorname{orth}\big((I - \sum_{i \le j} Q_i Q_i^\top)\, A Q_j\big)``: multiply the newest block by ``A`` and orthogonalise against everything so far. (Directions that are numerically dependent are dropped; this is called deflation.) By construction:

> **Krylov property:** ``A Q_j`` lies in ``\operatorname{span}(Q_1, \dots, Q_{j+1})``. So ``A`` maps every block into the space, except the newest one, ``Q_m``, which "sticks out".

The products ``AQ`` (computed once per column) and the small matrix ``\hat A := Q^\top A Q`` are stored.

**Galerkin solve.** Look for ``K = Q Y``, with ``Y`` of size ``(\dim) \times r``. The residual is ``\mathcal{R} := R - A Q Y - Q Y M^\top``. Require ``Q^\top \mathcal{R} = 0``. With ``Q^\top Q = I``:
```math
\hat A\, Y + Y M^\top = Q^\top R.
```
This is a small dense Sylvester equation, solved by Bartels-Stewart (3.3), here with ``p \approx 63`` and ``q = r``.

**How large is the residual?** Let ``P := QQ^\top``, the orthogonal projector onto the space. Then:
- the Galerkin condition says ``P\mathcal{R} = Q Q^\top \mathcal{R} = 0``, so ``\mathcal{R} = (I - P)\mathcal{R}``;
- ``R`` lies in the space, so ``(I - P) R = 0``;
- ``Q Y M^\top`` lies in the space, so ``(I - P)\, Q Y M^\top = 0``.

What remains is
```math
\mathcal{R} = -(I - P)\, A Q\, Y.
```
By the Krylov property, ``(I - P) A Q_j = 0`` for every block except the newest. Let ``Z := \operatorname{orth}\big((I - P) A Q_m\big)`` (``n \times b``) and ``\Gamma := Z^\top A Q`` (``b \times \dim``, nonzero only in the columns of the newest block). Then ``(I - P) A Q = Z \Gamma``, and since ``Z`` has orthonormal columns,
```math
\|\mathcal{R}\|_F = \|Z\, \Gamma Y\|_F = \|\Gamma Y\|_F,
```
a product of two *small* matrices. If ``\|\Gamma Y\|_F > 0.01 \cdot \text{tol} \cdot \|R\|_F``, one more block is added and the Galerkin solve is repeated.

**Why this is efficient.**
1. **Built once.** The space depends only on ``A`` and ``C_u``, not on ``M``, so it serves every later K-step. In the 2-D random problem at ``n = 4096`` it reaches 9 blocks of 7 columns (dimension 63) during the first K-step and never grows afterwards. Building it costs 9 products ``A \cdot (n \times 7)`` for each of ``A`` and ``B``. That is most of the run time: the first BUG iteration takes 78% of the total at ``n = 4096`` and 87% at ``n = 16384`` (see the error-vs-time figures).
2. **No more dense products afterwards.** All BUG bases stay inside ``\operatorname{span}(Q)``. (The code keeps the augmented bases inside it: `restrict_to_basis!`.) So any product ``AU`` is computed from the stored ``AQ`` as ``AU = (AQ)(Q^\top U)``, which costs ``O(n \cdot \dim \cdot r)`` instead of ``O(n^2 r)`` (`apply!`).
3. **Few blocks suffice.** An element of ``\mathcal{K}_m(A, C_u)`` is ``p(A)\, C_u`` for a matrix polynomial ``p`` of degree ``< m``. By (1.2), what must be approximated is ``(A + \mu I)^{-1}`` for the shifts ``\mu`` in ``\operatorname{spec}(M)``, i.e. on the set ``\operatorname{spec}(A + \mu I) \approx [12, 13] + [-2, -1] = [10, 12]``. For a *symmetric* matrix with spectrum in ``[\alpha, \beta]``, ``0 < \alpha``, the best polynomial (a Chebyshev polynomial) reduces the error by a factor of about
   ```math
   \rho = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}, \qquad \kappa = \beta/\alpha,
   ```
   per degree. Here ``\kappa = 12/10 = 1.2``, so ``\rho \approx 0.046`` and ``\rho^9 \approx 10^{-12}``: nine blocks, as observed. (``A`` is not symmetric, so this is a guide only; the actual stopping test is the residual above.)
   - **For the Laplacian this fails.** There ``\kappa \approx 0.4\,n^2``, so ``\rho \approx 1 - 3/n``, and reaching ``10^{-12}`` would need about ``9n`` steps, more than the dimension ``n`` itself. That is why the Laplacian uses the direct K-step of 3.2 instead.

**3-D random problem.** The three tensor K-steps use the same solver, with ``A_k`` and ``U_{b,k}``. The relevant spectra are again far from 0, e.g. for mode 1: ``\lambda(A_1) + \lambda(M_1) \in [-2, -1] + ([101, 102] + [-202, -201]) = [-103, -100]``.

**Why RKSM and TBRK cannot use this trick.** Their right-hand sides ``u`` are new basis vectors every iteration, not in ``\operatorname{span}(C_u)``, and every pole ``\xi`` is different. So there is no single space to reuse, and each iteration pays a fresh GMRES solve, each step of which is a pass over the dense ``A``.

### 3.7 Convection-diffusion: non-symmetric tridiagonal systems

`BandedLU` in [src/inner_solvers.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/inner_solvers.jl), using `LAPACK.gtsv!` from Julia's LinearAlgebra; the problems are `convdiff_1d`, `matrix_convdiff`, `tensor_convdiff` in [src/problems.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/problems.jl)

**The operator.** The 3-D test problem of the paper that introduced TBRK (Casulli, Section 6, Figure 3), and the same problem one dimension lower: ``\nu\Delta u + w\cdot\nabla u = f`` on ``[0, 1]^d`` with zero boundary values, in the sign convention of that paper's code (the paper writes ``-\nu\Delta u + w\cdot\nabla u = f``, which differs by the direction of ``w``), with
```math
\nu = 0.1, \qquad w = \Big(1 + \frac{(x_1 + 1)^2}{4},\ 0,\ \dots\Big), \qquad f = \frac{1}{1 + x_1 + \dots + x_d},
```
discretised with central differences on the grid ``0 = t_1 < \dots < t_n = 1``, ``h = 1/(n-1)``. In one dimension, with a variable coefficient ``w(t)``:
```math
T = -\nu\,\frac{1}{h^2}\operatorname{tridiag}(-1, 2, -1) + \operatorname{diag}\big(w(t_i)\big)\,\frac{1}{2h}\operatorname{tridiag}(-1, 0, 1),
```
so the diagonal entries are ``-2\nu/h^2``, ``T_{i,i+1} = \nu/h^2 + w_i/(2h)`` and ``T_{i+1,i} = \nu/h^2 - w_{i+1}/(2h)``. Mode 1 is **convective**; the other modes have ``w = 0`` and are pure **diffusion** (symmetric).
- **3-D** (``n = 4096``; the paper used ``n = 1024``): ``A_1`` convective, ``A_2 = A_3`` diffusive; ``f`` has multilinear rank (10, 10, 10).
- **2-D** (``n = 4096``): ``A`` convective, ``B`` diffusive; ``f`` has rank 8.

The right-hand sides keep the singular values of ``f`` that are ``\ge 10^{-10}``, as both reference codes do. (The paper's data file has ``\nu = 0.01`` in its convective mode, because its code multiplies that mode by ``\nu`` twice. We follow the paper's stated ``\nu = 0.1``; everything else agrees with the file to rounding.)

**Real spectrum, mildly non-normal.** Both off-diagonals are positive here (``|w| h < 2\nu``). Then the diagonal matrix ``D = \operatorname{diag}(\rho_1, \dots, \rho_n)`` with ``\rho_1 = 1``, ``\rho_{i+1} = \rho_i \sqrt{T_{i+1,i}/T_{i,i+1}}`` makes ``S := D^{-1} T D`` symmetric: ``S_{i+1,i} = \frac{\rho_i}{\rho_{i+1}} T_{i+1,i}`` and ``S_{i,i+1} = \frac{\rho_{i+1}}{\rho_i} T_{i,i+1}`` both equal ``\sqrt{T_{i,i+1} T_{i+1,i}}``. So ``T = D S D^{-1}`` has the real eigenvalues of the symmetric tridiagonal ``S``, which the code computes (to rounding) and passes to RKSM and TBRK. The eigenvectors of ``T`` are ``D`` times those of ``S``, with
```math
\operatorname{cond}(D) = \prod_i \sqrt{\frac{T_{i,i+1}}{T_{i+1,i}}} \approx \exp\Big(\int_0^1 \frac{w}{2\nu}\Big) = e^{7.9} \approx 3 \cdot 10^3
```
for the convective operator. That is non-normal but not extremely so: dense eigenvalue computations still return the real spectrum. (With ``\nu = 0.01`` it would be ``e^{79}``, and computed eigenvalues of ``T`` and of its projections ``V^\top T V`` become complex.)

**Solving ``(T + \sigma I)\, x = b``.** Gaussian elimination on a tridiagonal matrix, as in 3.1, but without symmetry: an LU factorisation ``T + \sigma I = \mathcal{L}\,\mathcal{U}`` with ``\mathcal{L}`` unit lower bidiagonal and ``\mathcal{U}`` upper triangular with (at most) two superdiagonals. LAPACK's `gtsv` does it with **partial pivoting**: at each step it swaps rows ``i`` and ``i+1`` if the entry below the pivot is larger. That makes it stable for every shift ``\sigma``, real or complex, at the cost of one extra superdiagonal (fill-in from the swaps). The cost is ``O(n)`` per right-hand side, and `gtsv` handles all columns of ``B`` with one factorisation. It overwrites the three diagonals with the factors, so the code reloads them (`load_shifted!`, ``3n`` copies) before every solve.

**The K-step ``A K + K M^\top = R``.** In 3.2 the symmetric ``M`` was diagonalised. Here ``M = V^\top B V`` is not symmetric, and diagonalising it is unsafe for the reason in 3.4 (its eigenvectors can be ill-conditioned). Use instead the **complex Schur form** ``M = Q T_M Q^H`` (LAPACK `zgees`, ``Q`` unitary, ``T_M`` upper triangular). Then ``M^\top = \bar Q\, T_M^\top Q^\top``, where ``\bar Q`` is the entrywise complex conjugate. Multiply the K-step from the right by ``\bar Q`` and use ``Q^\top \bar Q = \overline{Q^H Q} = I``:
```math
A\,(K \bar Q) + (K \bar Q)\, T_M^\top = R \bar Q.
```
With the new unknown ``\tilde K := K \bar Q`` and ``\tilde R := R \bar Q``, column ``j`` of ``\tilde K T_M^\top`` is ``\sum_i \tilde k_i (T_M)_{ji}``, which only involves ``i \ge j`` because ``T_M`` is upper triangular. Column ``j`` of the equation is therefore
```math
\big(A + (T_M)_{jj} I\big)\, \tilde k_j = \tilde r_j - \sum_{i > j} (T_M)_{ji}\, \tilde k_i, \qquad j = r, r-1, \dots, 1:
```
the columns are no longer independent as in 3.2, but they are coupled *triangularly*, exactly as in step 3 of Bartels-Stewart (3.3), so solving the last column first leaves one tridiagonal system per column. Each has a complex shift ``(T_M)_{jj}`` (an eigenvalue of ``M``) and is solved by `gtsv` in complex arithmetic. Finally ``K = \tilde K \bar Q^{-1} = \tilde K\, Q^\top``, whose imaginary part is rounding error and is dropped.

**Cost.** ``O(r^3)`` for the Schur form, ``O(nr)`` for the ``r`` tridiagonal solves, and ``O(nr^2)`` for the products with ``\bar Q``, ``Q^\top`` and the coupling sums: still linear in ``n``, as for the Laplacian, but in complex arithmetic (about 4 times the operations of 3.2).

**Shifted solves (RKSM, TBRK).** ``(A - \xi I) Y = u`` is one call to `gtsv` with ``\sigma = -\xi``: in real arithmetic for a real pole, in complex arithmetic for a complex one, whose real and imaginary parts then give the real block ``[\operatorname{Re} y\ \ \operatorname{Im} y]`` (as in 2.3).

---

## 4. Case by case

| case | method | systems solved, and how |
|---|---|---|
| **2-D Laplacian** | BUG | K- and L-steps: eigendecomposition of ``M``, then ``r`` Thomas solves (3.2). S-step: Bartels-Stewart (3.3). |
| | RKSM | shifted solves ``(L - \xi I)Y = u``: Thomas, 7 columns, real poles only (3.1). Galerkin: Bartels-Stewart (3.3). |
| **2-D convection-diffusion** | BUG | K- and L-steps: complex Schur form of ``M``, then ``r`` coupled tridiagonal LU solves, last column first (3.7). S-step: Bartels-Stewart (3.3). |
| | RKSM | shifted solves: tridiagonal LU, real and complex poles (3.7). Galerkin: Bartels-Stewart (3.3). |
| **2-D random** | BUG | K- and L-steps: one block Krylov space per operator, built in the first iteration, then Galerkin with Bartels-Stewart (3.6). S-step: Bartels-Stewart (3.3). |
| | RKSM | shifted solves: GMRES, real poles with ``M = -\xi I``, complex poles with the ``2 \times 2``-block ``M`` (3.5). Galerkin: Bartels-Stewart (3.3). |
| **3-D Laplacian** | Tucker BUG | three K-steps per iteration: eigendecomposition of ``M_k``, then Thomas (3.2). Core: diagonalisation (3.4). |
| | TBRK | three shifted solves per iteration: Thomas (3.1). Core: diagonalisation (3.4). |
| **3-D convection-diffusion** | Tucker BUG | three K-steps: complex Schur of ``M_k``, then tridiagonal LU (3.7). Core: complex Schur + 3-D back substitution (3.4). |
| | TBRK | three shifted solves: tridiagonal LU (3.7). Core: complex Schur + 3-D back substitution (3.4). |
| **3-D random** | Tucker BUG | three K-steps: block Krylov space per mode, then Galerkin (3.6). Core: complex Schur + 3-D back substitution (3.4). |
| | TBRK | three shifted solves: GMRES (3.5). Core: complex Schur + 3-D back substitution (3.4). |

**Rule of thumb.**
- A **sparse, structured** operator is solved **directly**, using its structure (Thomas, or tridiagonal LU when it is not symmetric).
- A **dense, well-conditioned** operator is solved **iteratively**, multiplying by it as few times as possible (Krylov).
- Every **small** projected problem is solved **densely**, by reducing it to diagonal or triangular form (eigendecomposition or Schur).

Both iterative inner solvers (GMRES and the block Krylov K-step) stop at the same relative residual, ``0.01 \cdot \text{tol} = 10^{-12}``, so neither kind of method gets a more accurate inner solve than the other. Thomas, eigendecompositions and Schur forms are direct and accurate to rounding error.

---

## 5. Checking the answers: the exact reference solutions

[src/reference.jl](https://github.com/gvretina/LR_BUG_Sylvester.jl/blob/main/src/reference.jl). These are not part of any method and are never timed. They are only used to measure errors in the figures.

**Laplacian: the sine transform.** The matrix ``\mathcal{S}`` with entries ``\mathcal{S}_{ij} = \sin\!\big(\frac{\pi i j}{n+1}\big)`` has the eigenvectors of ``L`` as its columns:
```math
L\, \mathcal{S} = \mathcal{S} \Lambda, \qquad \lambda_j = -\frac{4}{h^2} \sin^2\!\Big(\frac{\pi j}{2(n+1)}\Big).
```
Moreover ``\mathcal{S}^2 = \frac{n+1}{2} I``, so ``\mathcal{S}^{-1} = \frac{2}{n+1}\mathcal{S}``. Substitute ``X = \mathcal{S} \tilde X \mathcal{S}`` into ``L X + X L = C`` (here ``A = B = L``, which is symmetric):
```math
\mathcal{S}\big(\Lambda \tilde X + \tilde X \Lambda\big)\mathcal{S} = C \quad\Longrightarrow\quad (\lambda_i + \lambda_j)\,\tilde X_{ij} = \Big(\frac{2}{n+1}\Big)^2 (\mathcal{S}\, C\, \mathcal{S})_{ij}.
```
Products with ``\mathcal{S}`` are computed by the fast sine transform (FFTW), in ``O(n^2 \log n)`` for the whole matrix. In 3-D one divides by ``\lambda_i + \lambda_j + \lambda_l`` instead.

**Random: the known eigenvectors.** By construction ``A = P_A D_A P_A^{-1}`` and ``B = P_B D_B P_B^{-1}``, so ``B^\top = P_B^{-\top} D_B P_B^\top``. Substitute ``X = P_A \tilde X P_B^\top``:
```math
P_A D_A \tilde X P_B^\top + P_A \tilde X P_B^\top P_B^{-\top} D_B P_B^\top = P_A\,(D_A \tilde X + \tilde X D_B)\,P_B^\top = C.
```
Both ``D``'s are diagonal, so entry by entry ``(d_{A,i} + d_{B,j})\,\tilde X_{ij} = (P_A^{-1} C P_B^{-\top})_{ij}``. Hence
```math
X = P_A \Big( \big(P_A^{-1} C P_B^{-\top}\big) \oslash \big(d_{A,i} + d_{B,j}\big)_{ij} \Big) P_B^\top.
```
The stored ``A`` was formed as ``P_A D_A P_A^{-1}`` in floating point, so it differs slightly from the exact product. We therefore add a few steps of **iterative refinement**, as in 3.3:
1. compute the residual ``C - AX - XB^\top`` with the *stored* ``A`` and ``B``;
2. solve for a correction with the formula above;
3. add it to ``X``.

This takes 3 steps in 2-D and 2 in 3-D. All of it needs dense ``n \times n`` products, about half an hour at ``n = 16384``. That is why the references are computed once and cached in `results/<dim>/<case>/data/reference_n<n>.h5`.

**Convection-diffusion: one big K-step.** In 2-D, ``B`` is the diffusive operator, which is symmetric tridiagonal, so the whole equation ``A X + X B^\top = C`` is the K-step of 3.2 with ``M = B`` itself. LAPACK's tridiagonal eigensolver gives ``B = Q M_B Q^\top`` with ``Q`` orthogonal. With ``\tilde X := X Q``, column ``j`` reads ``(A + \mu_j I)\, \tilde x_j = (CQ)_j``, where the ``\mu_j`` are the eigenvalues of ``B``: ``n`` independent tridiagonal systems. ``A`` is not symmetric, so they are solved by tridiagonal LU with partial pivoting (3.7), ``O(n)`` each and ``O(n^2)`` in all. Finally ``X = \tilde X Q^\top``. Only orthogonal transformations and backward-stable solves are involved: no inverse and no refinement. The 3-D problem at ``n = 4096`` would need a dense ``n^3`` array (512 GB), so its figures show residuals only, like the other 3-D figures at ``n = 4096``.
