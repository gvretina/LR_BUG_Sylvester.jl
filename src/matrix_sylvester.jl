
@inline function kronecker_sum_sylv(A,B)
    na = size(A,1)
    nb = size(B,1)
    P = kron(I(nb),A) + kron(transpose(B),I(na))
    return P
end

function Matrix_BUG_Sylvester(;n=2^7,
                               problem="laplacian_dirichlet",
                               mode=:adaptive,
                               abs_err=nothing,trunc_tol=1e-10)
    m = n
    x1 = LinRange(0,4pi,n+1)[begin:end-1]
    x2 = LinRange(0,4pi,m+1)[begin:end-1]
    ttol = trunc_tol

    if occursin("laplacian",problem)
        B = smooth_periodic_random_matrix(n,m,max_freq=3)
        rb = rank(B)

        Is = [1:n; 1:n-1; 2:n]; Js = [1:n; 2:n; 1:n-1]; Vs = [fill(-2,n); fill(1, 2n-2)];
        A1 = sparse(Is,Js,Vs)
        if occursin("periodic",problem)
            #WARNING THE PERIODIC LAPLACIAN HAS A ZERO EIGENVALUE
            #THIS CASE ISN'T EXPECTED TO WORK WELL (yet does in some way?)

            #This is needed for uniqueness of solution
            B = B .- mean(B)
            

            A1[1,end] = 1
            A1[end,1] = 1
        elseif occursin("dirichlet",problem)
            nothing
        else
            error("Unknown problem_type: $problem_name. Choose from \"laplacian_periodic\", \"laplacian_dirichlet', or \"random\"")
        end
        A2 = copy(A1) ./ x2[2]^2
        A1 = A1 ./ x1[2]^2

    elseif problem == "random"
        rb = 7
        B = rand(n,rb)*rand(rb,m) #B*B'
        smin = 1
        smax = 2
        spec_range = 1
        dist = 10
        A1 = generate_matrix(n,(smax+dist+spec_range,smax+dist),uniform_eigen)
        A2 = generate_matrix(m,(smax,smin,-1),uniform_eigen)
    else
        error("Unknown problem_type: $problem_name. Choose from \"laplacian_periodic\", \"laplacian_dirichlet', or \"random\"")
    end
    println("rb = $(rb)")

    if occursin("laplacian",problem)
        P = kronecker_sum_sylv(A1,A2)
        x = P \ vec(B)
        X = reshape(x,n,m)
    else
        X = sylvester(A1,A2',-B)
    end

    Ux,Sx,Vx = svd(X)
    tol = (ttol*norm(Sx))^2
    r = findfirst(x -> x < tol, reverse(cumsum(reverse(Sx.^2))))# - 1
    r = isnothing(r) ? length(Sx) : r - 1
    if r < n ÷ 2
        @show r
    end

    Xtr = Ux[:,1:r]*Diagonal(Sx[1:r])*Vx[:,1:r]'
    isnothing(abs_err) ? abs_err = norm(A1*Xtr + Xtr*A2' - B) : nothing

    r = rb
    U = qr(rand(n,r)).Q[:,1:r]
    V = qr(rand(m,r)).Q[:,1:r]
    ru = rv = r

    Q1 = U' * A1 * U
    Q2 = V' * A2 * V
    BV = B*V
    BtU = B'*U

    errs = Float64[]

    ##### Compute error of random initial guess ######
    coef_K = kronecker_sum_sylv(A1,Q2')
    K = reshape(coef_K \ vec(BV),n,ru)
    coef_L = kronecker_sum_sylv(A2,Q1')
    L = reshape(coef_L \ vec(BtU),m,rv)
    U = qr(K) |> Matrix
    V = qr(L) |> Matrix

    Q1 = U' * A1 * U
    BtU = B'*U
    Q2 = V' * A2 * V
    BV = B*V
    UtBV = U' * BV

    if issparse(A1)
        coef_S = kronecker_sum_sylv(Q1,Q2')
        vecS = coef_S \ vec(UtBV)
        S = reshape(vecS,ru,rv)
    else
        S = sylvester(Q1,Q2',-UtBV)
    end
    Y = U*S*V'
    err = norm(A1*Y + Y*A2'- B)
    push!(errs,err)

    max_iter = 10
    p = Progress(max_iter)

    for i in 1:max_iter
        Q1 = U' * A1 * U
        Q2 = V' * A2 * V
        BV = B*V
        BtU = B'*U

        if issparse(A1)
            K = sylvester_sparse_dense(A1,Q2',BV)
        else
            coef_K = kronecker_sum_sylv(A1,Q2')
            K = reshape(coef_K \ vec(BV),n,ru)
        end

        if issparse(A2)
            L = sylvester_sparse_dense(A2,Q1',BtU)
        else
            coef_L = kronecker_sum_sylv(A2,Q1')
            L = reshape(coef_L \ vec(BtU),m,rv)
        end

        if mode == :fixed
            U, Ru = qr(K)#[K U])
            V, Rv = qr(L)#[L V])
        elseif mode == :adaptive
            U, Ru = qr([K U])
            V, Rv = qr([L V])
            ru = rank(Ru)
            rv = rank(Rv)
        end
        U = U[:,1:ru]
        V = V[:,1:rv]

        Q1 = U' * A1 * U
        BtU = B'*U
        Q2 = V' * A2 * V
        BV = B*V
        UtBV = U' * BV

        coef_S = kronecker_sum_sylv(Q1,Q2')
        # S = sylvester(Q1t,Q2t',-UtBV)
        S = reshape((coef_S) \ vec(UtBV),ru,rv)
        P,Σ,Q = svd(S)

        if mode == :fixed
            r = length(Σ)
         elseif mode == :adaptive
            s = norm(Σ)
            tol = ttol*s
            tolsq = (tol)^2
            r = findfirst(x -> x < tolsq, reverse(cumsum(reverse(Σ.^2))))# - 1
            r = isnothing(r) ? length(Σ) : r - 1
        end
        U = U*P[:,1:r]
        V = V*Q[:,1:r]
        S = Diagonal(Σ[1:r])
        ru = rv = r
        Y = U*S*V'
        err = norm(A1*Y + Y*A2'- B)
        push!(errs,err)

        if err < abs_err || (i>1 && abs(err-errs[i]) < abs_err)
            break
        end

        next!(p, showvalues=[(:err,err),(:r,r)],valuecolor=:yellow)
    end
    finish!(p)
    return (U,S,V), errs, B, (A1,A2), X
end

example_matrix(;n=2^7,problem="laplacian",mode=:adaptive) = begin
    n_elems_sq = n
    trunc_tol = 1e-10
    sol,errs,B,As,X = Matrix_BUG_Sylvester(n=n,problem=problem,mode=mode,trunc_tol=trunc_tol);

    Y = sol[1]*sol[2]*sol[3]'

    U,sX,V = svd(X)
    sY = diag(sol[2])
    r = length(sY)
    Xtr = U[:,1:r]*Diagonal(sX[1:r])*V[:,1:r]'

    R = X - Y
    residual_error = norm(As[1]*R + R*As[2]')
    trunc_err = norm(As[1]*Xtr + Xtr*As[2]' - B)

    n = length(errs)
    N = n > 10 ? div(n,10) : 1
    v = 0:N:n |> collect

    inch = 96
    colors = RGB.(MakiePublication.seaborn_colorblind())
    set_theme!(theme_latexfonts(), palette=(color = colors,))
    fig = Figure(fontsize=24,size=1.5 .* (7inch,3inch))
    fig1 = fig[1,1:2]

    Label(fig[2,1:2],text="Iterations",padding=(0,0,0,0))
    ax1 = Axis(fig1,ylabel="Error",yscale=log10,xticks = (v, string.(v)))
    scatterlines!(ax1,0:n-1,errs/n_elems_sq,label=L"\Vert \mathcal{L}(Y\,)-B \Vert_{sF}")
    hlines!(ax1,trunc_err/n_elems_sq,label=L"\Vert \mathcal{L}(X_r\,)-B \Vert_{sF}", color=colors[4])
    hlines!(ax1,residual_error/n_elems_sq,label=L"\Vert \mathcal{L}(R\,) \Vert_{sF}", color=colors[5])
    axislegend(ax1)

    n = minimum(size(Y))
    get_exp_base_10(x) = round(Int,log10(x))

    if get_exp_base_10(n)-get_exp_base_10(r) >= 2
        xticks =  [min(r,div(n,2)),max(r,div(n,2)),n]
    elseif r < 10
        xticks =  [r,div(n,2),n]
    elseif abs(r-div(n,2)) <= 5
        xticks =  [1,r,n]
    else
        xticks = [1,min(r,div(n,2)),max(r,div(n,2)),n]
    end
    fig2 = fig[1,3:5]
    Label(fig[2,3:5],text="Index of singular value", padding=(0,0,0,0))
    ax21 = Axis(fig2[1,1], ylabel=L"\sigma_k\, /\sigma_1",yscale=log10, xticks=xticks)
    scatterlines!(ax21,sX ./ sX[1],label=L"\sigma_X",markersize=14)
    scatterlines!(ax21,sY ./ sY[1],label=L"\sigma_Y",markersize=14,marker='⨯')
    tol=trunc_tol*norm(sX)
    # exponent = get_exp_base_10(trunc_tol)
    hlines!(ax21,tol/sX[1],label=L"\vartheta", color=colors[3])
    xticks = [1, div(r,2), r]
    ax22 = Axis(fig2[1,2], ylabel=L"|\sigma_X - \sigma_Y\,|",yscale=log10, xticks=xticks)
    scatterlines!(ax22,abs.(sX[1:r].-sY),markersize=14)
    axislegend(ax21)
    rowgap!(fig1.layout, 1, 5)
    rowgap!(fig2.layout, 1, 5)
    display(fig)
    save("results/matrix_$(problem)_$n.pdf",fig)#,px_per_unit = 3)#,px_per_unit=dpi/96)
    save("results/matrix_$(problem)_$n.eps",fig)#,px_per_unit = 3)#,px_per_unit=dpi/96)

    sol, errs
end

function run_all_matrix()

    Random.seed!(1)
    problem_set = ["random", "laplacian_dirichlet"]

    for problem in problem_set
        println(problem)
        example_matrix(n=2^7,problem=problem)
    end
    example_matrix(n=2^11,problem="laplacian_dirichlet")
    nothing
end
