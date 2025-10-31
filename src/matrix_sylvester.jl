using LinearAlgebra
using ProgressMeter
using FStrings
using Random
using SparseArrays
using LaTeXStrings
using CairoMakie, Colors
using MakiePublication

function mean(A::AbstractArray)
    return sum(A) / length(A)
end

function get_makie_figure_size(desired_width_px::Int, desired_height_px::Int; px_per_unit=1.0, dpi=96.0)
    # Calculate the size in points based on desired pixels, px_per_unit, and DPI
    points_per_pixel = 96.0 / dpi  # Standard points per pixel at given DPI
    width_points = desired_width_px * points_per_pixel / px_per_unit
    height_points = desired_height_px * points_per_pixel / px_per_unit

    # Return the size tuple rounded to integers, as Makie expects integer sizes
    return (round(Int, width_points), round(Int, height_points))
end

uniform_eigen(n,smax,smin,s=1) = begin
    samples = (rand(n-2) * (smax - smin)) .+ (smin)
    samples = [smax; samples; smin]
#     samples = (rand(n) * (smax - smin)) .+ (smin)

    samples = s == 1 ? sort(samples) : sort(samples,rev=true)
    d = sign.(samples)
    return s .* samples .* d
end

my_normalize(p) =
    reduce(hcat,map(x->x/norm(x),eachcol(p)))

function generate_matrix(n,args,func)
    p = my_normalize(rand(n,n))
    inv_p = pinv(p)
    eigenvals = func(n,args...)
    return p*Diagonal(eigenvals)*inv_p
end

function smooth_periodic_random_matrix(m::Int, n::Int; max_freq::Int=3, rng=Random.GLOBAL_RNG)
    # Ensure m, n >= 2 and max_freq >= 1
    @assert m >= 2 && n >= 2 "Matrix dimensions must be at least 2x2"
    @assert max_freq >= 1 "Maximum frequency must be at least 1"

    # Initialize the matrix
    A = zeros(Float64, m, n)

    # Grid points (excluding the last point to avoid redundancy due to periodicity)
    x = LinRange(0, 4π, m+1)[1:m]
    y = LinRange(0, 4π, n+1)[1:n]

    # Generate random Fourier coefficients
    # For frequencies k, l from -max_freq to max_freq
    for k in -max_freq:max_freq, l in -max_freq:max_freq
        # Random amplitude (scaled to reduce high-frequency contributions)
        amp = randn(rng) / (1 + abs(k) + abs(l))  # Decay to ensure smoothness
        phase = 2π * rand(rng)  # Random phase
        # Add contribution of cos(kx + ly + phase) to ensure real output
        for i in 1:m, j in 1:n
            A[i, j] += amp * cos(k * x[i] + l * y[j] + phase)
        end
    end

    return A
end

function LR_BUG_Sylvester(;n=2^7,problem="laplacian_periodic",mode=:adaptive,abs_err=nothing,trunc_err=1e-10)
    m = n
    x1 = LinRange(0,4pi,n+1)[begin:end-1]
    x2 = LinRange(0,4pi,m+1)[begin:end-1]
    ttol = trunc_err

    if occursin("laplacian",problem)
        B = smooth_periodic_random_matrix(n,m,max_freq=3)
        B = B .- mean(B)
        rb = rank(B)

        Is = [1:n; 1:n-1; 2:n]; Js = [1:n; 2:n; 1:n-1]; Vs = [fill(-2,n); fill(1, 2n-2)];
        A1 = sparse(Is,Js,Vs)
        if occursin("periodic",problem)
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

    X = sylvester(Matrix(A1),Matrix(A2)',-B)
    Ux,Sx,Vx = svd(X)
    tol = (trunc_err*norm(Sx))^2
    r = findfirst(x -> x < tol, reverse(cumsum(reverse(Sx.^2))))# - 1
    r = isnothing(r) ? length(Sx) : r - 1
    if r < n ÷ 2
        @show r
    end

    Xtr = Ux[:,1:r]*Diagonal(Sx[1:r])*Vx[:,1:r]'
    abs_err = norm(A1*Xtr + Xtr*A2' - B)

    r = rb
    U = qr(rand(n,r)).Q[:,1:r]
    V = qr(rand(m,r)).Q[:,1:r]
    ru = rv = r

    Q1t = U' * A1 * U
    Q2t = V' * A2 * V
    BV = B*V
    BtU = B'*U

    errs = Float64[]

    ##### Compute error of random initial guess ######
    coef_K = kron(I(rv),A1) + kron(Q2t, I(n))
    K = reshape(coef_K \ vec(BV),n,ru)
    coef_L = kron(I(ru),A2) + kron(Q1t, I(m))
    L = reshape(coef_L \ vec(BtU),m,rv)
    U = qr(K) |> Matrix
    V = qr(L) |> Matrix

    Q1t = U' * A1 * U
    BtU = B'*U
    Q2t = V' * A2 * V
    BV = B*V
    UtBV = U' * BV

    coef_S = kron(I(rv),Q1t) + kron(Q2t,I(ru))
    S = reshape(pinv(coef_S) * vec(UtBV),ru,rv)
    Y = U*S*V'
    err = norm(A1*Y + Y*A2'- B)
    push!(errs,err)

    max_iter = 10
    p = Progress(max_iter)

    for i in 1:max_iter
        Q1t = U' * A1 * U
        Q2t = V' * A2 * V
        BV = B*V
        BtU = B'*U

        coef_K = kron(I(rv),A1) + kron(Q2t, I(n))
        K = reshape(coef_K \ vec(BV),n,ru)
        coef_L = kron(I(ru),A2) + kron(Q1t, I(m))
        L = reshape(coef_L \ vec(BtU),m,rv)

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

        Q1t = U' * A1 * U
        BtU = B'*U
        Q2t = V' * A2 * V
        BV = B*V
        UtBV = U' * BV

        coef_S = kron(I(rv),Q1t) + kron(Q2t,I(ru))
        S = reshape(pinv(coef_S) * vec(UtBV),ru,rv)
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

run_example(;n=2^7,problem="laplacian_periodic",mode=:adaptive) = begin
    Random.seed!(42)
    n_elems_sq = n
    sol,errs,B,As,X = LR_BUG_Sylvester(n=n,problem=problem,mode=mode);

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

    dpi = 92
    colors = RGB.(MakiePublication.seaborn_colorblind())
    set_theme!(theme_latexfonts(), palette=(color = colors,))
    fig = Figure(fontsize=24,size=get_makie_figure_size(1024,420;dpi=dpi))#,px_per_unit=1)
    fig1 = fig[1,1:2]

    Label(fig[2,1:2],text="Iterations",padding=(0,0,0,0))
    ax1 = Axis(fig1,ylabel="Error",yscale=log10,xticks = (v, string.(v)))
    scatterlines!(ax1,0:n-1,errs/n_elems_sq,label=L"\Vert \mathcal{L}(Y\,)-B \Vert_{sF}")
    hlines!(ax1,trunc_err/n_elems_sq,label=L"\Vert \mathcal{L}(X_r\,)-B \Vert_F", color=colors[4])
    hlines!(ax1,residual_error/n_elems_sq,label=L"\Vert \mathcal{L}(R\,) \Vert_F", color=colors[5])
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
    scatterlines!(ax21,sX ./ sX[1],label="Exact Sol.",markersize=14)
    scatterlines!(ax21,sY ./ sY[1],label="Num. Sol.",markersize=14,marker='⨯')
    ttol=1e-10*norm(sX)
    hlines!(ax21,ttol/sX[1],label=L"\vartheta", color=colors[3])
    xticks = [1, div(r,2), r]
    ax22 = Axis(fig2[1,2], ylabel=L"|\sigma_X - \sigma_Y\,|",yscale=log10, xticks=xticks)
    scatterlines!(ax22,abs.(sX[1:r].-sY),markersize=14)
    axislegend(ax21)
    rowgap!(fig1.layout, 1, 5)
    rowgap!(fig2.layout, 1, 5)
    display(fig)
    save("results/matrix_$problem.pdf",fig,px_per_unit = 3)#,px_per_unit=dpi/96)

end

function run_all_matrix()
    problem_set = ["random", "laplacian_dirichlet", "laplacian_periodic"]

    for problem in problem_set
        println(problem)
        run_example(n=2^7,problem=problem)
    end
end