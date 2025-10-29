using LinearSolve
using Random
using CairoMakie, MakiePublication, Colors
using LaTeXStrings
using SparseArrays
include("tensor.jl")
include("TTN.jl")

function remove_at(a::AbstractVector{T}, idx::Int) where T
    n = length(a)
    result = Vector{T}(undef, n-1)
    copyto!(result, 1, a, 1, idx-1)
    copyto!(result, idx, a, idx+1, n-idx)
    return result
end

¬(A::AbstractArray, skip_dim::Int) = complementary_size(A,skip_dim)

function complementary_size(A::AbstractArray, skip_dim::Int)
    s = size(A)
    total = 1
    @inbounds for i in 1:length(s)
        if i != skip_dim
            total *= s[i]
        end
    end
    return total
end
#### RETHINK OF ORDER OF BASIS MATMUL AND TRANSPOSES
function tucker_sylv(op,rhs,rs=nothing,tol=1e-6)
    if isnothing(rs)
        r_min = 1
        r_max = 100
    else
        r_min, r_max = rs
    end

    t = copy_structure(rhs)
    n_leaves = count_leaves(t)
    for i in eachindex(t.leaves)
        n,r = size(t.leaves[i].X)
        @reset t.leaves[i].X = Matrix(qr(rand(n,r)).Q)#[:,r]
    end
    t.X .= rand(size(t.X)...)
    ts = ntuple(n_leaves) do k
        @set t.leaves[k].X = op[k]*t.leaves[k].X
    end

    lhs = add_TTNs(ts)
    err = add_TTNs((lhs, (@set rhs.X = -rhs.X)))
    F_norm_err = contract_TTNs(err,err) |> only |> sqrt
    errs = []
    append!(errs,F_norm_err)

    # t = orthonormalize_ttn!(t)
    T = eltype(t)
    max_iter = 10
    flag = Array{T}(undef,1,1)
    M = [sparse(t.leaves[i].X'*op[i]'*t.leaves[i].X) for i in 1:n_leaves]
    N = [sparse(rhs.leaves[i].X'*t.leaves[i].X) for i in 1:n_leaves]
    Uhats = Vector{Matrix{T}}(undef,n_leaves)
    Q = Vector{Matrix{T}}(undef,n_leaves)
    iter = 0
    prev_err = errs[1]
    while iter < max_iter
        iter += 1
        for k in 1:n_leaves
            r = size(t.X,k)
            if r > ¬(t.X,k)
                throw("Nope")
                Q[k] = flag
            else
                Q[k] = Matrix(qr(matricize(t.X,k)').Q)#[:,1:r]
            end
            Uhats[k] = solve_basis(t,rhs,op[k],Q[k],M,N,k,rs)
        end

        leaves = get_tuple(Val(n_leaves),(i,Uhats)->TTN(Uhats[i]),TTN{T},Uhats)
        t_aug = TTN(t.X,leaves)

        M .= [sparse(Uhats[i]'*op[i]'*Uhats[i]) for i in 1:n_leaves]
        N .= [sparse(rhs.leaves[i].X'*Uhats[i]) for i in 1:n_leaves]
        t = solve_core(t_aug,rhs,M,N)
        t = rank_truncation(t,1e-10)
        M .= [sparse(t.leaves[i].X'*op[i]'*t.leaves[i].X) for i in 1:n_leaves]
        N .= [sparse(rhs.leaves[i].X'*t.leaves[i].X) for i in 1:n_leaves]

        ts = ntuple(n_leaves) do k
            @set t.leaves[k].X = op[k]*t.leaves[k].X
        end

        lhs = add_TTNs(ts)
        err = add_TTNs((lhs, (@set rhs.X = -rhs.X)))
        F_norm_err = contract_TTNs(err,err) |> only |> sqrt
        # #@myshow F_norm_err
        append!(errs,F_norm_err)
        # r_tol = tol*norm(t.X)
        if F_norm_err <= tol
            println("first")
            # println(r_tol)
            break
        elseif abs2(prev_err-F_norm_err) < 1e-10
            println("second")
            break
        end
        prev_err = F_norm_err
    end
    return t, errs
end

function solve_basis(t,rhs,Ak,Qk,M,N,k,rs=nothing)

    if isnothing(rs)
        r_min = 1
        r_max = 100
    else
        r_min, r_max = rs
    end
    n_leaves = count_leaves(t)
    Uk = t.leaves[k].X
    Pk = nothing
    for i in 1:n_leaves
        if i == k
            continue
        end
        P_coef = [(j == i ? M[j] : I(t.leaves[j].r)) for j in n_leaves:-1:1 if j != k]
        Pi = reduce(kron,P_coef)
        isnothing(Pk) ? Pk = Pi : Pk += Pi
    end

    R_coef = [N[j] for j in n_leaves:-1:1 if j != k]

    if length(Qk) != 1
        Pk = Qk' * Pk * Qk
        Rk = reduce(kron, R_coef) * Qk
        r2 = ¬(t.X,k)
    else
        Rk = reduce(kron, R_coef)
        r2 = size(Rk,2)
    end

    #Ax=b formulation
    A = kron(I(size(Pk,2)),Ak) + kron(Pk',I(size(Ak,2))) |> sparse
    b = rhs.leaves[k].X * matricize(rhs.X,k) * Rk |> vec

    # Yk = Array{eltype(t)}(undef, size(Uk,1), r2)
    # x = vec(Yk)
    #do this elsewhere too, simpleGMRES might be better
    u = solve(LinearProblem(A,b, UMFPACKFactorization())).u
    Yk = reshape(u,size(Uk,1),:)
    # #@myshow size(u)
    # #@myshow size(x)
    # x .= u
    #    x .= cg(A,b) ### probably should be iterative, might want to look into LinearSolve
    Uk_new, Rn = qr([Yk Uk])

    r = rank(Rn)

    if r < r_min
        r = r_max
    elseif r > r_max
        r = r_max
    end

    return (@views @inbounds Uk_new[:,1:r])
end

function solve_core(t,rhs,M,N)

    n_leaves = count_leaves(t)
    P = nothing
    for i in 1:n_leaves
        P_coef = [(j == i ? M[j] : I(t.leaves[j].r)) for j in n_leaves:-1:1]
        Pi = reduce(kron,P_coef)
        isnothing(P) ? P = Pi' : P += Pi'
    end
    R_coef = [N[j] for j in n_leaves:-1:1]
    R = reduce(kron,R_coef)'

    #Ax=b formulation
    b = R*vec(rhs.X)
    A = P

    rs = ntuple(i->t.leaves[i].r,n_leaves)
    C = Array{eltype(t)}(undef,rs...)
    x = vec(C)

    x .= solve(LinearProblem(A,b, UMFPACKFactorization())).u
    # x .= A \ b

    return TTN(C, t.leaves)
end

function smooth_periodic_random_array(sizes::NTuple{N,Int}, x;
                                       max_freq::Int=3,
                                       rng=Random.GLOBAL_RNG) where {N}
    @assert all(>=(2), sizes) "All dimensions must be at least size 2"
    @assert max_freq >= 1 "max_freq must be at least 1"

    # Initialize the array
    A = zeros(Float64, sizes)

    # Precompute coordinate arrays for each dimension
    coords = [x for _ in sizes]

    # Iterate over frequency vectors
    freq_ranges = (-max_freq:max_freq for _ in 1:N)

    for kvec in Iterators.product(freq_ranges...)
        # Random amplitude decays with total frequency magnitude
        freq_norm = sum(abs, kvec)
        amp = randn(rng) / (1 + freq_norm)
        phase = 2π * rand(rng)

        # Precompute kvec[i] * coords[i] for each dimension
        kx_arrays = ntuple(i -> kvec[i] .* coords[i], N)

        # Use CartesianIndices for efficient iteration
        for idx in CartesianIndices(sizes)
            # Compute dot product: sum of kvec[i] * x[i] for all dimensions
            dotval = sum(kx_arrays[i][idx[i]] for i in 1:N)
            A[idx] += amp * cos(dotval + phase)
        end
    end

    return A
end

uniform_eigen(n,smax,smin,s=1) = begin
    samples = (rand(n-2) * (smax - smin)) .+ (smin)
    samples = [smax; samples; smin]
#     samples = (rand(n) * (smax - smin)) .+ (smin)

    samples = s == 1 ? sort(samples) : sort(samples,rev=true)
    d = sign.(samples)
    return s .* samples .* d
end
function generate_matrix(n,args,func)
    p = my_normalize(rand(n,n))
    inv_p = pinv(p)
    eigenvals = func(n,args...)
    return p*Diagonal(eigenvals)*inv_p
end

my_normalize(p) =
    reduce(hcat,map(x->x/norm(x),eachcol(p)))

function example(;n=2^7,d=3,problem_name)
    Random.seed!(1234)
    if problem_name == "laplacian_periodic"
        xmin = 0
        xmax = 4π
        x = LinRange(xmin, xmax, n+1)[begin:end-1]
        dx = x[2] - x[1]
        B = smooth_periodic_random_array(ntuple(i->n, d), x)
        mean = sum(B) / prod(size(B))
        B .= B .- mean

        Is = [1:n; 1:n-1; 2:n]
        Js = [1:n; 2:n; 1:n-1]
        Vs = [fill(-2, n); fill(1, 2n-2)]
        A1 = sparse(Is, Js, Vs)
        A1[1, end] = 1
        A1[end, 1] = 1
        D = A1 / dx^2
        Ds = ntuple(i->D, d)
        Cb, Ub = tucker_hosvd(B; tol=1e-12)
        B_TTN = TTN(Cb, ntuple(i->TTN(Ub[i]), d))

    elseif problem_name == "laplacian_dirichlet"
        xmin = 0
        xmax = 4π
        x = LinRange(xmin, xmax, n+1)[begin:end-1]
        dx = x[2] - x[1]
        B = smooth_periodic_random_array(ntuple(i->n, d), x)
        mean = sum(B) / prod(size(B))
        B .= B .- mean

        Is = [1:n; 1:n-1; 2:n]
        Js = [1:n; 2:n; 1:n-1]
        Vs = [fill(-2, n); fill(1, 2n-2)]
        A1 = sparse(Is, Js, Vs)
#         A1[1, end] = 1
#         A1[end, 1] = 1
        D = A1 / dx^2
        Ds = ntuple(i->D, d)
        Cb, Ub = tucker_hosvd(B; tol=1e-12)
        B_TTN = TTN(Cb, ntuple(i->TTN(Ub[i]), d))

    elseif problem_name == "random"
        spec_range = 1
        dist = 100
        Ds = ntuple(i->generate_matrix(n, dist*(i-1) .+ (1+spec_range, 1, (-1)^i), uniform_eigen), d)
        r = 7
        B_TTN = TTN(rand(fill(r, d)...), ntuple(i->TTN(Matrix(qr(rand(n, r)).Q)), d))
        B = half_reconstruct(B_TTN)

    else
        error("Unknown problem_type: $problem_name. Choose from \"laplacian_periodic\", \"laplacian_dirichlet', or \"random\"")
    end

    n_leaves = count_leaves(B_TTN)

    X = solve_multilinear_sylvester(Ds,B)
    Cx,Ux = tucker_hosvd(X; tol=1e-16)
    X_TTN = TTN(Cx,ntuple(i->TTN(Ux[i]),d))
    X_TTN_trunc = rank_truncation(X_TTN,1e-10)
    X_TTN_rest = add_TTNs((X_TTN,(@set X_TTN_trunc.X = -X_TTN_trunc.X)),1e-16)

    Xs = ntuple(n_leaves) do k
        @set X_TTN_rest.leaves[k].X = Ds[k]*X_TTN_rest.leaves[k].X
    end
    residual = add_TTNs(Xs)
    residual_err_norm = contract_TTNs(residual,residual) |> only |> sqrt

    rs = (2,34)
    Y_TTN,errs = tucker_sylv(Ds,B_TTN,rs,residual_err_norm)

    R = add_TTNs((Y_TTN,(@set X_TTN.X = -X_TTN.X)))
    norm_R = contract_TTNs(R,R) |> only |> sqrt
    Rs = ntuple(n_leaves) do k
        @set R.leaves[k].X = Ds[k]*R.leaves[k].X
    end
    residual = add_TTNs(Rs)
    residual_norm = contract_TTNs(residual,residual) |> only |> sqrt

    Xs = ntuple(n_leaves) do k
        @set X_TTN_trunc.leaves[k].X = Ds[k]*X_TTN_trunc.leaves[k].X
    end
    lhs_X = add_TTNs(Xs)
    trunc_err = add_TTNs((lhs_X,(@set B_TTN.X = -B_TTN.X)))
    trunc_err_norm = contract_TTNs(trunc_err,trunc_err) |> only |> sqrt

    colors = RGB.(MakiePublication.seaborn_colorblind())
    set_theme!(theme_latexfonts(), palette=(color = colors,))
    fig = Figure(fontsize=24,dpi=144)

    ns = length(errs)
    N = ns > 10 ? div(ns,10) : 1
    v = 0:N:ns |> collect

    n_elems = n^d |> sqrt

    ax1 = Axis(fig[1,1],title="Multilinear rank of Y = $(size(Y_TTN.X))",ylabel="Error",xlabel="Iterations",
                yscale=log10,xticks = (v, string.(v)),titlefont=:regular)#,xlabel="Iterations")
    scatterlines!(ax1,0:length(errs)-1,errs/n_elems,label=L"\Vert\mathcal{L}(Y\,)-B\Vert_{s{F}}")#label=L"\Vert A_1 Y + Y A_2^T - B \Vert_F")
    hlines!(ax1,trunc_err_norm/n_elems,label=L"\Vert \mathcal{L}(X_r\,) -B\Vert_{s{F}}", color=colors[4])
    hlines!(ax1,residual_norm/n_elems,label=L"\Vert \mathcal{L}(R\,) \Vert_{s{F}}", color=colors[5])
    axislegend(ax1)#,position = :lb)
#     name = "random"
    save("results/tucker_$problem_name.eps",fig,px_per_unit = 3)#,px_per_unit=dpi/96)
    save("results/tucker_$problem_name.pdf",fig,px_per_unit = 3)#,px_per_unit=dpi/96)
#     display(fig)
    # errs
    Y_TTN, errs
end

function solve_multilinear_sylvester(Ais::NTuple{N,AbstractMatrix}, B::AbstractArray) where {N}
    sizes = size(B)
    eigs = [eigen(Matrix(A)) for A in Ais]
    P = [E.vectors for E in eigs]
    Λ = [E.values for E in eigs]

    # Transform B into eigenbasis
    B̂ = B
    for i in 1:N
        B̂ = n_mode_product(B̂, inv(P[i]), i)
    end

    # Elementwise divide in spectral space
    X̂ = similar(B̂)
    for I in CartesianIndices(sizes)
        denom = sum(Λ[i][I[i]] for i in 1:N)
        X̂[I] = B̂[I] / denom
    end

    # Transform back
    X = X̂
    for i in 1:N
        X = n_mode_product(X, P[i], i)
    end

    return X
end

