using LinearAlgebra
using ProgressMeter
using FStrings
using Random
using SparseArrays
using LaTeXStrings
using Distributions
using CairoMakie , Colors
using MakiePublication

macro myshow(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :(print($(string(ex)), " = ")))
        push!(blk.args, :(display($(esc(ex)))))
    end
    return blk
end

function get_makie_figure_size(desired_width_px::Int, desired_height_px::Int; px_per_unit=1.0, dpi=96.0)
    # Calculate the size in points based on desired pixels, px_per_unit, and DPI
    points_per_pixel = 96.0 / dpi  # Standard points per pixel at given DPI
    width_points = desired_width_px * points_per_pixel / px_per_unit
    height_points = desired_height_px * points_per_pixel / px_per_unit

    # Return the size tuple rounded to integers, as Makie expects integer sizes
    return (round(Int, width_points), round(Int, height_points))
end

function rand_mixture(n::Int=1, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, w=0.5)
    # Input validation for weight
    if !(0 ≤ w ≤ 1)
        throw(ArgumentError("Weight w must be between 0 and 1"))
    end

    # Define the two Gaussian components, truncated to [0, ∞)
    gaussian1 = Truncated(Normal(μ1, σ1), 0, Inf)
    gaussian2 = Truncated(Normal(μ2, σ2), 0, Inf)

    # Create a mixture model with weights [w, 1-w]
    mixture = MixtureModel([gaussian1, gaussian2], [w, 1-w])

    # Sample n values from the mixture
    return rand(mixture, n)
end

uniform_eigen(n,smax,smin,s=1) = begin
    samples = (rand(n-2) * (smax - smin)) .+ (smin)
    samples = [smax; samples; smin]
#     samples = (rand(n) * (smax - smin)) .+ (smin)

    samples = s == 1 ? sort(samples) : sort(samples,rev=true)
    d = sign.(samples)
    return s .* samples .* d
end

normal_eigen(n,mean,std,s=1) = begin
    samples = sort(randn(n) * std) .+ (mean)
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

#works perfectly for symmetric and antisymmetric cases without constants
function iterative_sol(n,args=nothing)
    m = n
    x1 = LinRange(0,4pi,n+1)[begin:end-1]
    x2 = LinRange(0,4pi,m+1)[begin:end-1]
    k1 = fftfreq(n)*(n)*2pi/(x1[end]+x1[2]-2x1[1])
    k2 = fftfreq(m)*(m)*2pi/(x2[end]+x2[2]-2x2[1])
    ttol = 1e-10

    a = 1e-2
    k = 1/2
    y1 = @. a*cos(k*x1)
    y2 = @. a*cos(k*x2)
    (@isdefined args) ? nothing : args = nothing
    if isnothing(args)
        pert = 100
        y1_coef = rand()*2*pert - pert
        y2_coef = rand()*2*pert - pert
        arb(v) = rand()*v + 1
        a1 = arb(3)
        a2 = arb(3)
    else
        y1_coef, a1, y2_coef, a2 = args
    end

    # B = @. 1 + y1_coef*Complex(y1)^a1 + y2_coef*Complex(y2)'^a2 + a*cos(k*x1 + k*x2') |> real
    # B = @. exp( -(x1-2pi)^2 / 1e-1 - (x2' - 2pi)^2 / 1e-1)
    B = smooth_periodic_random_matrix(n,m,max_freq=3)
    # Ub, Sb, Vbt = svd(B)
    # rb = rank(B)
    # B = Ub[:,1:rb]*Diagonal(Sb[1:rb])*Vbt[:,1:rb]'
    # B = @. 1 + y1 + y2'
    B = B .- mean(B)
    # Ub, Sb, Vbt = svd(B)
    # rb = Diagonal(Sb) |> rank
    # Uhat = reduce(hcat,fft.(eachcol(Ub[:,1:rb])));
    # Vhat = reduce(hcat,fft.(eachcol(Vbt[:,1:rb])));
    # B = Uhat*Diagonal(Sb[1:rb])*Vhat'

    rb = rank(B)
#     @myshow rb
    # @show (rb,y1_coef,a1,y2_coef,a2)

#     rb = 5
    # B = rand(n,rb)
#     B = rand(n,rb)*randn(rb,m) #B*B'

    # U = qr(rand(n,rb)).Q[:,1:rb]#Ub[:,rb+1:2rb] #qr(rand(n,rb)).Q[:,1:rb]#rand(n,rb)#
    # V = qr(rand(m,rb)).Q[:,1:rb]#Vbt[:,rb+1:2rb] #qr(rand(2n,rb)).Q[:,1:rb]#Vbt[:,rb+1:2rb] #rand(2n,rb)#

    # rb = 5
    # rb = 13

    D(x,k) = begin
        reduce(hcat,map(eachcol(x)) do y
            yk = fft(y)
            lmul!(Diagonal(-k.^2),yk)
            return real.(ifft(yk))
        end)
    end

    # A1 = AstroVlasov.central_differences(n,x1[2]-x1[1];order=2);
    # A2 = AstroVlasov.central_differences(m,x2[2]-x2[1];order=2);

    # B = rand(n,m)

    Is = [1:n; 1:n-1; 2:n]; Js = [1:n; 2:n; 1:n-1]; Vs = [fill(-2,n); fill(1, 2n-2)];
    A1 = sparse(Is,Js,Vs)
#     A1[1,end] = 1
#     A1[end,1] = 1
    A2 = copy(A1) ./ x2[2]^2
    A1 = A1 ./ x1[2]^2


    # A1 = ifft(I(n),1) * Diagonal(-k1.^2) * fft(I(n),1) #|> real
    # A2 = ifft(I(m),1) * Diagonal(-k2.^2) * fft(I(m),1) #|> real
    # A1 = Diagonal(-k1.^2)
    # A2 = Diagonal(-k2.^2)
    # X = reshape( (kron(I(m),A1) + kron(A2', I(n)) ) \ vec(B), n,m)
    # r = rank(X)
    # @show r
#     l1 = max(n,m) #max(n,m)
#     l2 = -div(l1,2)
#     s1 = 1
#     s2 = 1#0.5
#     r = 1
#     w = 0.5
    # eigenval_gen(n,l,s=-1) = s*LinRange(1,n,n) .+ (s*l)
    # eigenval_gen(n,l,s=-1) = sort(s * rand(n) * l) .+ (s*l)
#     eigenval_gen(n,l,s=-1) = begin
#         l == 0 ? l = 0.1 : nothing
#         samples = sort(randn(n) / sqrt(abs(l))) .+ (s*l)
#         d = sign.(samples)
#         return s .* samples .* d
#     end
#
    # eigenval_gen(n,l) = -rand_mixture(n,l,sqrt(l),1,1,w)
#     iter = 0
#     my_normalize(p) =
#              reduce(hcat,map(x->x/norm(x),eachcol(p)))
#     while true
#         iter += 1

        # d1 = collect(1:n)./l .- l
        # d2 = collect(1:m)./l .- l

#         d1 = eigenval_gen(n,l1,s1)#sort(-rand(n) * l)# .- l)#,rev=true)
#         d2 = eigenval_gen(m,l2,s2)#sort(-rand(n) * l)# .- l)#,rev=true)

        # d1 = (@. -100 / (1 + exp( 0.5*(x1 - x1[n÷2]) )) - 1)
        # d2 = (@. -100 / (1 + exp( 0.5*(x2 - x2[n÷2]) )) - 1)

#     indices = [(i, j) for (i, a) in pairs(d1), (j, b) in pairs(d2) if abs(a + b) < 1e-14]

#     while length(indices) > 1
#         d1 = eigenval_gen(n,l1,s1)#sort(-rand(n)*4,rev=true)
#         d2 = eigenval_gen(m,l2,s2)#sort(-rand(n)*4,rev=true)
#         indices = [(i, j) for (i, a) in pairs(d1), (j, b) in pairs(d2) if abs(a + b) < 1e-14]
#     end
#     p1 = my_normalize(rand(n,n))
#     p1inv = pinv(p1)
#     p2 = my_normalize(rand(m,m))
#     p2inv = pinv(p2)
#     A1 = p1inv * Diagonal(d1) * p1
#     A2 = p2inv * Diagonal(d2) * p2
smin = 1
smax = 2
spec_range = 1
dist = 10
#     A1 = generate_matrix(n,(smax,smin,-1),uniform_eigen)
#     A2 = generate_matrix(m,(smax+dist,smin+dist),uniform_eigen)
#     A1 = generate_matrix(n,(smax+dist+spec_range,smax+dist),uniform_eigen)
#     A2 = generate_matrix(m,(smax,smin,-1),uniform_eigen)

#     sA = svd(Matrix(A1)).S
#     sB = svd(Matrix(A2)).S
#     val = [x+y for x in eigen(Matrix(A1)).values, y in eigen(Matrix(A2)).values] |> vec
#     eig_range = (round(minimum(val),digits=4),round(maximum(val),digits=4))
#     condD = (sA[1]+sA[2])/(sA[end] + sB[end])

    P =  kron(I(n),A1) + kron(A2, I(n))
    smax = svdsolve(P,5,:LR)[1][1]
    smin = svdsolve(P,5,:SR)[1][1]
    cond = smax/smin
    @show cond
    throw("Stop")
#     X = sylvester(Matrix(A1),Matrix(A2)',-B)
    X = reshape( P  \ vec(B), n,n)
        s = svd(X).S# .< 1e-10)
        tol = (ttol*norm(s))^2
        r = findfirst(x -> x < tol, reverse(cumsum(reverse(s.^2))))# - 1
        r = isnothing(r) ? length(s) : r - 1
        if r < n ÷ 2
            @show r
        end

#         @show (dist, eig_range,condD,r)

#     X = sylvester(Matrix(A1),Matrix(A2)',-B)
    # A = kron(I(m),A1) + kron(A2,I(n))
    # X = similar(B)
    # x = vec(X)
    # x .= pinv(A) * vec(B)
    Ux,Sx,Vx = svd(X)
    tol = (ttol*norm(Sx))^2
    r = findfirst(x -> x < tol, reverse(cumsum(reverse(Sx.^2))))# - 1
    r = isnothing(r) ? length(Sx) : r - 1

    R = Ux[:,r+1:end]*Diagonal(Sx[r+1:end])*Vx[:,r+1:end]'
    trunc_err = norm(A1*R + R*A2')

    # U = reduce(hcat,fft.(eachcol(Ub[:,rb+1:2rb])));
    # V = reduce(hcat,fft.(eachcol(Vbt[:,rb+1:2rb])));
#     r = 7
    U = qr(rand(n,r)).Q[:,1:r]#Ub[:,rb+1:2rb] #qr(rand(n,rb)).Q[:,1:rb]#rand(n,rb)#
    V = qr(rand(m,r)).Q[:,1:r]#Vbt[:,rb+1:2rb] #qr(rand(2n,rb)).Q[:,1:rb]#Vbt[:,rb+1:2rb] #rand(2n,rb)#

    # P = kron(I(m),A1) + kron(A2,I(n)) |> Matrix
    # @myshow sqrt.(eigen(P'*P).values)
    # small_lambda = eigs(P, which=:SM, ritzvec=false)[1]
    # large_lambda = eigs(P, which=:LM, ritzvec=false)[1]
    # @myshow small_lambda
    # @myshow large_lambda
    # try

    # if l >= 2
    #     P = kron(I(m),A1) + kron(A2,I(n))
    #     sigma_min = eigs(P'*P; which=:SM,ritzvec=false, nev=1)[1][1] |> sqrt
    #     @show sigma_min
    # end
    # catch e
    #     nothing
    # end

    # ΔV = Vx[:,1:r] - V[:,1:r]
    # @show norm(ΔV)
    # @myshow svd(ΔV).S[1]
    K = similar(U)
    L = similar(V)
    Uold = similar(U)
    Vold = similar(V)

    vecK = vec(K)
    vecL = vec(L)
    ru = size(K,2)
    rv = size(L,2)
    # coef = kron(I(ru),A)
    S = Array{eltype(U)}(undef,ru,rv)
    vecS = vec(S)
    Q1t = U' * A1 * U
    Q2t = V' * A2 * V
    BV = B*V
    BtU = B'*U

    errs = []
    errsU = []
    errsV = []

    max_iter = 10
    p = Progress(max_iter)

    for i in 1:max_iter
        Q1t = U' * A1 * U
        Q2t = V' * A2 * V
        BV = B*V
        BtU = B'*U

        coef_K = kron(I(rv),A1) + kron(Q2t, I(n))
        # vecK .= coef_K \ vec(BV)
#         K = reshape(pinv(coef_K) * vec(BV),n,ru)
        K = reshape(coef_K \ vec(BV),n,ru)
        #         K = sylvester(A1,Q2t',-BV)
#         @show size(K)
        coef_L = kron(I(ru),A2) + kron(Q1t, I(m))
#         L = reshape(pinv(coef_L) * vec(BtU),m,rv)
        L = reshape(coef_L \ vec(BtU),m,rv)
#         L = sylvester(A2,Q1t',-BtU)
        # copyto!(Uold,U)
        Uold = copy(U)
        # U,Ru = qr([K U])
        # ru = min(2ru,size(U,2))
        # U = U[:,1:ru]
        U,Ru = qr(K)#[K U])
        U = U[:,1:ru]#2ru]
#         @show size(U)


        # vecL .= coef_L \ vec(BtU)
        # copyto!(Vold,V)

        Vold = copy(V)
        # V, Rv = qr([L V])
        # rv = min(2rv,size(V,2))
        # V = V[:,1:rv]
        V, Rv = qr(L)#[L V])
        V = V[:,1:rv]#2rv]


        errU = norm(Ru)#U'*Uold)
        errV = norm(Rv)#V'*Vold)
        err1 = norm(B'*U)
        err2 = norm(B*V)
        err3 = norm(U'*B*V)
        push!(errsU,(errU,err1))
        push!(errsV,(errV,err2))
        # if abs(err1 - errsU[i][2]) < 1e-10 && abs(err2 - errsV[i][2]) < 1e-10
        #     break
        # end

    #     next!(p)
    #     if errU < 1e-8 && errV < 1e-8
    #         break
    #     end

    # end
        # M = U1' * U
        # N = V1' * V

        Q1t = U' * A1 * U
        BtU = B'*U
        Q2t = V' * A2 * V
        BV = B*V
        UtBV = U' * BV

        # coef_S = kron(I(rv),Q1t) + kron(Q2t,I(ru))
        # S = reshape(coef_S \ vec(UtBV),ru,rv)
        # r = ru
        coef_S = kron(I(rv),Q1t) + kron(Q2t,I(ru))
        S = reshape(pinv(coef_S) * vec(UtBV),ru,rv)
        P,Σ,Q = svd(S)
        # s = norm(Σ)
        # tol = ttol*s
        # tolsq = (tol)^2
        # r = findfirst(x -> x < tolsq, reverse(cumsum(reverse(Σ.^2))))# - 1
        # r = isnothing(r) ? length(Σ) : r - 1
        r = length(Σ)
        U = U*P[:,1:r]
        V = V*Q[:,1:r]
        S = Diagonal(Σ[1:r])
        ru = rv = r
        Y = U*S*V'
        # Y .-= mean(Y)
        err = norm(A1*Y + Y*A2'- B)
        # err2 = norm(Y-f)
        push!(errs,(err,err3))

        if err < trunc_err || (i>1 && abs(err-errs[i-1][1]) < tol)
            break
        end
        # sleep(0.1)
        next!(p, showvalues=[(:err,err),(:r,r),(:mean,mean(Y))],valuecolor=:yellow)
    end
    finish!(p)
    return (U,S,V), (errs,errsU,errsV), B, (A1,A2), X
end

test_sol(n) = begin
    Random.seed!(44)

    pert = 100
    y1_coef = rand()*2*pert - pert
    y2_coef = rand()*2*pert - pert
    arb(v) = rand()*v + 1
    a1 = arb(4)
    a2 = arb(4)

    args = (y1_coef,a1,y2_coef,a2)
    a = y1_coef
    b = y2_coef
    e1 = nothing
    e2 = nothing
    err1 = nothing
    err2 = nothing
    B = nothing
    As = nothing
    sol1 =nothing
    X = nothing
    try
        sol1,err1,B,As,X = iterative_sol(n);
    catch e1
        println("Iterative failed")
    end
    try
        error("No")
        sol2,err2 = gradient_flow(n,args)
    catch e2
        println("Gradient flow failed")
    end
    # (isnothing(err1) || isnothing(err2)) && throw("One of the methods failed.")
    # errU = err[2]
    # errV = err[3]
    # err = err[1]
    # display(err);# err[err .> normB] .= NaN;


    dpi = 92
    colors = RGB.(MakiePublication.wong())
    set_theme!(theme_latexfonts(), palette=(color = colors,))
    fig = Figure(fontsize=24,size=get_makie_figure_size(1024,420;dpi=dpi))#,px_per_unit=1)
    fig1 = fig[1,1]
    s = ~isnothing(err1) + ~isnothing(err2)
    if s == 0
        throw("Neither method worked.")
    end
    errU = err1[2]
    errV = err1[3]
    err1 = err1[1]

    sol = sol1
    Y = sol[1]*sol[2]*sol[3]'
    @show rank(Y)

    U,sX,V = svd(X)
    sY = diag(sol[2])
    r = length(sY)
#     R = U[:,r+1:end] * Diagonal(sX[r+1:end]) * V[:,r+1:end]'
    @show sX[1:r]
    @show abs.(sX[1:r] .- sY) #./ sX[1:r]
    # R = fft(U[:,r+1:end],1) * Diagonal(sX[r+1:end]) * fft(V[:,r+1:end],1)'

    # err_n = norm(X-Y)
    # err_R = norm(R)
    R = X - Y
    trunc_error = norm(As[1]*R + R*As[2]')
    abs_error = first.(err1)[end]
    @show trunc_error
    println("abs. err. = ",norm(As[1]*Y + Y*As[2]' - B))
    @show abs(trunc_error-abs_error)
    @show norm(X-Y)
    @show norm(R)

    # ax0 = Label(fig[0,1:2],text=f"B is hermitian",#f"B = 1 + {y1_coef:.3f}*a*cos(kx1)^{a1:.3f} + {y2_coef:.3f}*a*cos(kx2')^{a2:.3f} + a*sin(x1+x2')",
    #  font= (:bold))
    n = length(errU)
    N = n > 10 ? div(n,10) : 1
    v = 0:N:n |> collect
    v[1] = 1
#     Label(fig[0,1:2],text="BUG Sylvester Solver",font=:bold)
    if ~isnothing(err1)
        ax1 = Axis(fig1[1,1],xlabel="Iterations",ylabel="Error",yscale=log10,xticks = (v, string.(v)))
        scatterlines!(ax1,1:length(err1),first.(err1),label=L"\Vert A_1 Y + Y A_2^T - B \Vert_F")
        hlines!(ax1,trunc_error,label=L"\Vert A_1 R + R A_2^T \Vert_F", color=colors[3])
        axislegend(ax1)
    end
    if ~isnothing(err2)
        ax2 = Axis(fig1[1,2],title="Gradient flow",xlabel="iterations",ylabel="error",yscale=log10)
        scatterlines!(ax2,1:length(err2),err2,label="Abs.Error")
        axislegend(ax2)
    end
    #save("results/test_bed/error.png",fig,dpi=144)
    # Y = Y .- mean(Y)

    # @myshow sum(abs2.(imag(Y)))
    # Y = real(Y)
    # global wtf = sol

#     eigen1 = eigen(As[1]).values |> real
#     eigen2 = eigen(As[2]).values |> real

#     fig4 = fig[2,2]
#     ax41 = Axis(fig4[1,1], title=L"\mathrm{Eigenvalues\,\, of\,\, }A_1")
#     ax42 = Axis(fig4[1,2], title=L"\mathrm{Eigenvalues\,\, of\,\, }A_2")
#     hist!(ax41,eigen1,label=L"\lambda(A_1)")
#     hist!(ax42,eigen2,label=L"\lambda(A_2)")

#     fig2 = fig[1,2]#Figure()
#     ax21 = Axis(fig2[1,1],title="Num. Sol.")
#     hm1 = heatmap!(ax21,Y)
#     Colorbar(fig2[1,1][1,2],hm1)
#     ax22 = Axis(fig2[1,2],title="Exact Sol.")
#     hm2 = heatmap!(ax22,X)
#     Colorbar(fig2[1,2][1,2],hm2)
    n = minimum(size(Y))
    get_exp_base_10(x) = round(Int,log10(x))

    if get_exp_base_10(n)-get_exp_base_10(r) >= 2
        xticks =  [min(r,div(n,2)),max(r,div(n,2)),n]
    elseif r < 10
        xticks =  [0,r,div(n,2),n]
    elseif abs(r-div(n,2)) <= 5
        xticks =  [0,r,n]
    else
        xticks = [0,min(r,div(n,2)),max(r,div(n,2)),n]
    end
    ax21 = Axis(fig[1,2], ylabel=L"\sigma_k\, /\sigma_1",yscale=log10,xlabel="Index of singular value", xticks=xticks)
    scatterlines!(ax21,sX ./ sX[1],label="Exact Sol.",markersize=14)
    scatterlines!(ax21,sY ./ sY[1],label="Num. Sol.",markersize=6)#,marker=:xcross)
    ttol=1e-10*norm(sX)
    hlines!(ax21,ttol/sX[1],label=L"\vartheta", color=colors[3])
    xticks = [1, div(r,2), r]
    ax22 = Axis(fig[1,3], ylabel=L"|\sigma_X - \sigma_Y\,|",yscale=log10,xlabel="Index of singular value", xticks=xticks)
    scatterlines!(ax22,abs.(sX[1:r].-sY),markersize=14)
    hlines!(ax22,ttol,label=L"\vartheta", color=colors[3])
    #     vlines!(ax2,r, color=:gray,label=latexstring("r=$(r)"))

    axislegend(ax22)
    axislegend(ax21)
    display(fig)
    save("/home/vretinaris/AstroVlasov.jl/results/low_rank_sylvester_laplacian.eps",fig,px_per_unit = 3)#,px_per_unit=dpi/96)


    # display(GLMakie.Screen(),fig1)
    # display(GLMakie.Screen(),fig2)
    # println()
    # println()
    # println()
    # println()
    # return (errU,errV)
    # return fig
end
