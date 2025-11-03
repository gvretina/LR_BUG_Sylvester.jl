using LinearSolve
using Random
using CairoMakie, MakiePublication, Colors
using SparseArrays
using StructArrays
using LinearAlgebra
using ProgressMeter
using FStrings
using LaTeXStrings

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


function mean(A::AbstractArray)
    return sum(A) / length(A)
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

function sylvester_sparse_dense(A::SparseMatrixCSC, B::AbstractMatrix, C::AbstractMatrix)
    n, m = size(C)
    @assert size(A,1) == size(A,2) == n "A must be square and match C's row dimension"
    @assert size(B,1) == size(B,2) == m "B must be square and match C's column dimension"

    # Schur decomposition of small dense B
    sch = schur(B)        # B = Q*T*Q'
    T = sch.T
    Q = sch.Z

    # Transform right-hand side
    Ctilde = C * Q

    Y = zeros(eltype(A), n, m)

    # Solve column by column
    for j in 1:m
        rhs = Ctilde[:, j]
        # subtract contributions from previous columns (since T is upper triangular)
        for k in 1:j-1
            rhs -= Y[:, k] * T[k, j]
        end
        # Solve (A + T[j,j]*I) * y_j = rhs
        M = A + T[j,j] * I
        y = M \ rhs   # sparse direct or iterative solve
        Y[:, j] = y
    end

    # Transform back
    X = Y * Q'
    return X
end
