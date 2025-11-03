
include("tensor.jl")
include("TTN.jl")
include("TTNO.jl")

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

function Tucker_BUG_Sylvester(operator_tuple,rhs,rs=nothing,abs_tol=1e-6, trunc_tol=1e-10)
    op, ttno = operator_tuple
    if isnothing(rs)
        r_min = 1
        r_max = 100
    else
        r_min, r_max = rs
    end
    flag = 0
    t = copy_structure(rhs)
    n_leaves = count_leaves(t)
    for i in eachindex(t.leaves)
        n,r = size(t.leaves[i].X)
        @reset t.leaves[i].X = Matrix(qr(rand(n,r)).Q)#[:,r]
    end
    t.X .= rand(size(t.X)...)

    lhs = apply_TTNO(ttno,t)
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
        t = rank_truncation(t,trunc_tol)
        M .= [sparse(t.leaves[i].X'*op[i]'*t.leaves[i].X) for i in 1:n_leaves]
        N .= [sparse(rhs.leaves[i].X'*t.leaves[i].X) for i in 1:n_leaves]

        lhs = apply_TTNO(ttno,t)
        err = add_TTNs((lhs, (@set rhs.X = -rhs.X)),trunc_tol)
        F_norm_err = norm(err.X)
        append!(errs,F_norm_err)
        if F_norm_err <= abs_tol
            println("first")
            break
        elseif abs2(prev_err-F_norm_err) <= abs_tol
            if flag == 1
                println("second")
                break
            else
                flag = 1
            end
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

    B = rhs.leaves[k].X * matricize(rhs.X,k) * Rk

    if issparse(Ak)
        Yk = sylvester_sparse_dense(Ak,Pk,B)
    else
        #Ax=b formulation
        A = kron(I(size(Pk,2)),Ak) + kron(Pk',I(size(Ak,2))) |> sparse
        b = B |> vec

        u = solve(LinearProblem(A,b)).u
        Yk = reshape(u,size(Uk,1),:)
    end
        
    Uk_new, Rn = qr([Yk Uk])

    r = rank(Rn)

    if r < r_min
        r = r_max
    elseif r > r_max
        r = r_max
    end

    return (@views @inbounds Uk_new[:,1:r])
end

# function solve_core(t,rhs,M,N)

#     n_leaves = count_leaves(t)
#     P = nothing
#     for i in 1:n_leaves
#         P_coef = [(j == i ? M[j] : I(t.leaves[j].r)) for j in n_leaves:-1:1]
#         Pi = reduce(kron,P_coef)
#         isnothing(P) ? P = Pi' : P += Pi'
#     end
#     R_coef = [N[j] for j in n_leaves:-1:1]
#     R = reduce(kron,R_coef)'

#     #Ax=b formulation
#     b = R*vec(rhs.X)
#     A = P
#     rs = ntuple(i->t.leaves[i].r,n_leaves)
#     C = Array{eltype(t)}(undef,rs...)
#     x = vec(C)
#     x .= solve(LinearProblem(A,b)).u
#     # F = lu(A)
#     # x .= F \ b
#     # x .= A \ b

#     return TTN(C, t.leaves)
# end

function solve_core(t,rhs,M,N)

    n_leaves = count_leaves(t)
    # C_X = t.X
    C_B = copy(rhs.X)
    for i in 1:n_leaves
        C_B = n_mode_product(C_B,Matrix(N[i]'),i)        
    end

    C_X = solve_multilinear_sylvester(map(transpose,M),C_B)

    return TTN(C_X, t.leaves)
end

function example_tucker(;n=2^7,d=3,problem_name="laplacian_periodic")
    if problem_name == "laplacian_periodic"
        xmin = 0
        xmax = 4π
        x = LinRange(xmin, xmax, n+1)[begin:end-1]
        dx = x[2] - x[1]
        B = smooth_periodic_random_array(ntuple(i->n, d), x)
        mean = sum(B) / length(B)
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

        Sylv_TTNO = Sylvester_TTNO(B_TTN,Ds)

    elseif problem_name == "laplacian_dirichlet"
        xmin = 0
        xmax = 4π
        x = LinRange(xmin, xmax, n+1)[begin:end-1]
        dx = x[2] - x[1]
        B = smooth_periodic_random_array(ntuple(i->n, d), x)
        mean = sum(B) / length(B)
        B .= B .- mean

        Is = [1:n; 1:n-1; 2:n]
        Js = [1:n; 2:n; 1:n-1]
        Vs = [fill(-2, n); fill(1, 2n-2)]
        A1 = sparse(Is, Js, Vs)
        D = A1 / dx^2
        Ds = ntuple(i->D, d)
        Cb, Ub = tucker_hosvd(B; tol=1e-12)
        B_TTN = TTN(Cb, ntuple(i->TTN(Ub[i]), d))
        Sylv_TTNO = Sylvester_TTNO(B_TTN,Ds)

    elseif problem_name == "random"
        spec_range = 1
        dist = 100
        Ds = ntuple(i->generate_matrix(n, dist*(i-1) .+ (1+spec_range, 1, (-1)^i), uniform_eigen), d)
        r = 7
        B_TTN = TTN(rand(fill(r, d)...), ntuple(i->TTN(Matrix(qr(rand(n, r)).Q)), d))
        B = half_reconstruct(B_TTN)
        Sylv_TTNO = Sylvester_TTNO(B_TTN,Ds)

    else
        error("Unknown problem_type: $problem_name. Choose from \"laplacian_periodic\", \"laplacian_dirichlet', or \"random\"")
    end

    n_leaves = count_leaves(B_TTN)

    X = solve_multilinear_sylvester(Ds,B)
    Cx,Ux = tucker_hosvd(X; tol=1e-16)
    X_TTN = TTN(Cx,ntuple(i->TTN(Ux[i]),d))    
    
    trunc_tol = 1e-10

    X_TTN_trunc = rank_truncation(X_TTN,trunc_tol)

    lhs_X = apply_TTNO(Sylv_TTNO,X_TTN_trunc)
    trunc_err = add_TTNs((lhs_X,(@set B_TTN.X = -B_TTN.X)))
    trunc_err_norm = norm(trunc_err.X)

    rs = (2,div(n,4))
    Y_TTN,errs = Tucker_BUG_Sylvester((Ds,Sylv_TTNO),B_TTN,rs,trunc_err_norm,trunc_tol)

    R = add_TTNs((Y_TTN,(@set X_TTN.X = -X_TTN.X)),1e-11)
    residual = apply_TTNO(Sylv_TTNO, R) |> orthonormalize_ttn!
    residual_norm = norm(residual.X) 

    colors = RGB.(MakiePublication.seaborn_colorblind())
    set_theme!(theme_latexfonts(), palette=(color = colors,))
    
    inch = 96

    fig = Figure(fontsize=24,size=1.5 .* (4inch,3inch))

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
    save("results/tucker_$(problem_name)_$n.eps",fig)#,px_per_unit=dpi/inch)
    save("results/tucker_$(problem_name)_$n.pdf",fig)#,px_per_unit=dpi/inch)

    Y_TTN, errs
end

function solve_multilinear_sylvester(Ais, B::AbstractArray{T}) where {T}
    N = length(Ais)
    sizes = size(B)
    eigs = [eigen(Matrix(A)) for A in Ais]
    P = [E.vectors for E in eigs]
    Λ = [E.values for E in eigs]
    if any(!isreal(Λ))
        P = map(StructArray,P)
        invP = map(StructArray ∘ inv, P)
        B̂ = StructArray(ComplexF64.(B))
    else
        invP = map(inv, P)
        B̂ = copy(B)
    end

    for i in 1:N
        B̂ = n_mode_product(B̂, invP[i], i)
    end

    # Elementwise divide in spectral space
    X = similar(B̂)
    for I in CartesianIndices(sizes)
        denom = sum(Λ[i][I[i]] for i in 1:N)
        X[I] = B̂[I] / denom
    end

    # Transform back
    for i in 1:N
        X = n_mode_product(X, P[i], i)
    end

    if eltype(X) <: Complex && !(T <: Complex)
        if any(abs.(imag(X)) .> 1e-15) 
            throw("Nope")
        else
            X = real(X)
        end
    end

    return X
end

function run_all_tucker()

    Random.seed!(1)
    problem_set = ["random", "laplacian_dirichlet"]

    for problem in problem_set
        println(problem)
        example_tucker(n=2^7,d=3,problem_name=problem);
    end
    nothing
end