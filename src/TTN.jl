using Accessors
using LinearAlgebra
using ConstructionBase

abstract type AbstractTTN{T,M,N,A,L} end

struct TTN{T,M,N,A<:AbstractArray{T,M},Leaves<:NTuple{N}} <: AbstractTTN{T,M,N,A,Leaves}
    X::A                    # Connection tensor C_τ or basis matrix
    size::NTuple{M,Int}
    leaves::Leaves          # Child nodes τ₁, ..., τₘ
    r::Int
    tmp::A
end

@inline Base.eltype(t::AbstractTTN{T,M,N,A,L}) where {T,M,N,A,L} = T
@inline Base.size(t::AbstractTTN{T,M,N,A,L}) where {T,M,N,A,L} = t.size

function TTN(C::AbstractArray{T}, leaves::Leaves) where {T,Leaves <: Tuple}
    s = size(C)
    TTN{T,ndims(C),length(leaves),typeof(C),Leaves}(C, s, leaves, s[end], similar(C))
end

function TTN(C::AbstractArray{T}, leaves::Leaves, tmp::AbstractArray{T}) where {T,Leaves <: Tuple}
    s = size(C)
    TTN{T,ndims(C),length(leaves),typeof(C),Leaves}(C, s, leaves, s[end], tmp)
end

function TTN(U::AbstractMatrix{T}) where {T}
    s = size(U)
    TTN{T,ndims(U),0,typeof(U),Tuple{}}(U, s, (), s[end], similar(U))
end

function TTN(U::AbstractMatrix{T},tmp::AbstractArray{T}) where {T}
    s = size(U)
    TTN{T,ndims(U),0,typeof(U),Tuple{}}(U, s, (), s[end], tmp)
end

@inline function isleaf(t::AbstractTTN)
    return isempty(t.leaves)
end

@inline function isroot(t::AbstractTTN)
    return length(t.leaves) == ndims(t.X)# - 1 && t.r == 1
end

@inline function tensorize_root(t::AbstractTTN)
    return reshape(t.X,(size(t.X)...,1))
end

# Function to print the tree structure
function print_tree(t::AbstractTTN, level::Int = 0)
    if isroot(t)
        s = "Root"
    elseif isleaf(t)
        s = "Leaf"
    else
        s = "Subtree"
    end
    println("\t" ^ level, "$s Dim: ", size(t))
    for leaf in t.leaves
        print_tree(leaf, level + 1)
    end
end

function print_tree(ttns::AbstractTTN...)
    for (i,ttn) in enumerate(ttns)
        print("$(i): ")
        print_tree(ttn)
    end
end

Base.show(io::IO, ::MIME"text/plain", x::AbstractTTN{T,M,N,A,L}) where {T,M,N,A,L} = begin
    name = typeof(x).name.name |> string
    println("Structure of $T $name:")
    print_tree(x)
end

function Base.copy(t::AbstractTTN{T,M,N,A,Leaves}) where {T,M,N,A<:AbstractArray{T,M},Leaves<:NTuple{N}}
    # Copy the connection tensor or basis matrix
    X_copy = copy(t.X)

    # Recursively copy the leaves (child nodes)
    leaves_copy = map(copy, t.leaves)

    # Copy the temporary array
    tmp_copy = copy(t.tmp)

    # Construct a new TTN with the copied fields
    return TTN{T,M,N,A,typeof(leaves_copy)}(
        X_copy, t.size, leaves_copy, t.r, tmp_copy
    )
end

function Base.copy(t::AbstractTTN{T,M,N,A,Leaves},type) where {T,M,N,A<:AbstractArray{T,M},Leaves<:NTuple{N}}
    # Copy the connection tensor or basis matrix
    X_copy = type.(t.X)

    # Recursively copy the leaves (child nodes)
    leaves_copy = map(l->copy(l,type), t.leaves)

    # Copy the temporary array
    tmp_copy = type.(t.tmp)

    # Construct a new TTN with the copied fields
    return TTN{type,M,N,typeof(X_copy),typeof(leaves_copy)}(
        X_copy, t.size, leaves_copy, t.r, tmp_copy
    )
end

function copy_structure(t::AbstractTTN{T,M,N,A,Leaves},type=nothing) where {T,M,N,A<:AbstractArray{T,M},Leaves<:NTuple{N}}
    isnothing(type) ? type = eltype(t) : nothing
    # Copy the connection tensor or basis matrix
    X_copy = Array{type}(undef,size(t.X))

    # Recursively copy the leaves (child nodes)
    leaves_copy = map(l->copy_structure(l,type), t.leaves)

    # Copy the temporary array
    tmp_copy = similar(X_copy)

    # Construct a new TTN with the copied fields
    return TTN{type,M,N,typeof(X_copy),typeof(leaves_copy)}(
        X_copy, t.size, leaves_copy, t.r, tmp_copy
    )
end

@inline function contract_TTNs(t1::AbstractTTN,t2::AbstractTTN)
    if isleaf(t1) == isleaf(t2) == true
        return contract_leaves(t1,t2)
    elseif isleaf(t1) == isleaf(t2) == false
        return _contract_TTNs(t1,t2)
    else
        throw("Cannot contract TTNs with incompatible structures.")
    end
end

### optimize this to become non-allocating, MOST IMPORTANT FUNCTION TO OPTIMIZE
function _contract_TTNs(t1::AbstractTTN,t2::AbstractTTN)
    @assert isleaf(t1) == isleaf(t2) == false
    m = length(t1.leaves)
    UW = @inbounds ntuple(j -> contract_TTNs(t2.leaves[j],t1.leaves[j]), m)
    ten = t1.tmp
    copyto!(ten,t1.X)
#     ten = copy(t1.X)

    @inbounds for j in 1:m
        ten = n_mode_product(ten,(UW[j]),j)
    end

    CUW = isroot(t1) ? matricize(reshape(ten,(size(ten)...,1)),0) : matricize(ten,0)
    G = isroot(t1) ? matricize(reshape(t2.X,(size(t2.X)...,1)),0) : matricize(t2.X,0)

    res = CUW * transpose(G)
    return res
end

@inline function contract_leaves(t1::AbstractTTN,t2::AbstractTTN)
    @assert isleaf(t1) == isleaf(t2) == true "Trees of different shape can't be contracted."
    return t1.X' * t2.X
end

function count_leaves(t::AbstractTTN)
    if isleaf(t)
        return 1
    else
        return sum(count_leaves, t.leaves)
    end
end

function orthonormalize_ttn!(root::AbstractTTN)
    # Process each child of the root (root is not orthonormalized)
    for (i, child) in enumerate(root.leaves)
        root = _orthonormalize_recursive!(child, root, i)
    end
    return root
end

function _orthonormalize_recursive!(node::AbstractTTN, parent::AbstractTTN, mode::Int)
    # Recursively process all children of the current node
    for (i, child) in enumerate(node.leaves)
        node = _orthonormalize_recursive!(child, node, i)
    end

    # Perform QR decomposition on node's X
    if isleaf(node)
        Q, R = qr(node.X)
        @views @inbounds copyto!(node.X, Q[:,1:size(node.X,2)])
    else
        Q, R = qr(transpose(matricize(node.X,0)))
        new_core = tensorize(transpose(Q[:,1:last(size(node.X))]),0,size(node.X))
        @assert size(node.X) == size(new_core)

        @views @inbounds copyto!(node.X, new_core)
    end
    # Contract R with parent's X along the specified mode
    @reset parent.leaves[mode] = node
    @reset parent.X = n_mode_product(parent.X, R, mode)
    @reset parent.size = size(parent.X) #not needed as R ∈ R^{r × r}
    @reset parent.r = last(parent.size)
    return parent
end

function orth_n_trunc_ttn!(root::AbstractTTN,tol::Real =1e-14)
     # Process each child of the root (root is not orthonormalized)
    for (i, child) in enumerate(root.leaves)
        root = _orth_n_trunc_recursive!(child, root, i, tol)
    end
    return root

end

function _orth_n_trunc_recursive!(node::AbstractTTN, parent::AbstractTTN, mode::Int, tol::Real =1e-14)
    # Recursively process all children of the current node
    for (i, child) in enumerate(node.leaves)
        node = _orth_n_trunc_recursive!(child, node, i)
    end
    # Perform QR decomposition on node's X
    if isleaf(node)
        Q, S, Vt = svd(node.X)
        #R = Diagonal(S) * Vt'
        r = findfirst(x->x<tol,S) #rank(R)
        r = isnothing(r) ? length(S) : r - 1
        R = @views @inbounds Diagonal(S[1:r]) * Vt'[1:r,:]
        @reset node.X = Q[:,1:r]
        @reset node.size = size(node.X)
        @reset node.r = r#last(node.size)
        @reset node.tmp = similar(node.X)
    else
        Q, S, Vt = svd(transpose(matricize(node.X,0)))
        r = findfirst(x->x<tol,S) #rank(R)
        r = isnothing(r) ? length(S) : r - 1
        R = @views @inbounds Diagonal(S[1:r]) * Vt'[1:r,:]
        s = size(node.X)
        @reset s[end] = r
        @reset node.X = @views tensorize(transpose(Q[:,1:r]),0,s)
        @reset node.size = s
        @reset node.r = r#last(node.size)
        @reset node.tmp = similar(node.X)
    end
    # Contract R with parent's X along the specified mode
#     mode = isroot(parent) ? mode - 1 : mode
    @reset parent.leaves[mode] = node
    @reset parent.X = n_mode_product(parent.X, R, mode)
    @reset parent.size = size(parent.X)
    @reset parent.r = last(parent.size)
    @reset parent.tmp = similar(parent.X)
    return parent
end

# Structural compatibility check using tuples
@inline function check_compatible(ttns::NTuple{N, AbstractTTN}) where {N}
    isempty(ttns) && return true
    first_ttn = first(ttns)

    # Check local node compatibility using all
    all(t -> (length(t.leaves) == length(first_ttn.leaves) &&
             size(t.X) == size(first_ttn.X)), ttns) || return false

    # Recursive check using tuple operations
    return all(1:length(first_ttn.leaves)) do i
        check_compatible(map(t -> t.leaves[i], ttns))
    end
end

function add_TTNs(ttns::NTuple{N,AbstractTTN},tol=1e-14) where {N}
    sum = _add_TTNs(ttns)
#     display(sum)
    return orth_n_trunc_ttn!(sum,tol)#sum#orthonormalize_ttn!(sum)#
end

function _add_TTNs(ttns::NTuple{N,AbstractTTN}) where {N}
    first_ttn = first(ttns)
    constructor = constructorof(typeof(first_ttn))
    if isleaf(first_ttn)
        # Base case: leaf nodes
        @assert all(t -> isleaf(t), ttns) "All TTNs must be leaves or non-leaves together"
        # Horizontal concatenation is already efficient for column-major matrices
        new_X = hcat(ntuple(i -> ttns[i].X, N)...)
        return constructor(new_X)
    else
        # Recursive case: non-leaf nodes
        @assert all(t -> length(t.leaves) == length(first_ttn.leaves), ttns) "Structure mismatch"

        # Process children recursively
        child_groups = ntuple(length(first_ttn.leaves)) do i
            _add_TTNs(ntuple(n -> ttns[n].leaves[i], N))
        end

        # Calculate dimensions for the new core tensor
        nd = ndims(first_ttn.X)
        core_dims = ntuple(nd) do d
            sum(ntuple(n -> size(ttns[n].X, d), N))
        end

        # Determine common element type
        T = eltype(first_ttn.X)#promote_type(ntuple(n -> eltype(ttns[n].X), N)...)
        new_core = zeros(T, core_dims)

        # Calculate starting indices for each tensor in each dimension
        offsets = ntuple(nd) do d
            ntuple(N) do n
                if n == 1
                    0
                else
                    sum(ntuple(i -> size(ttns[i].X, d), n-1))
                end
            end
        end

        # Fill the core tensor using direct block copying
        for n in 1:N
            # Calculate ranges for this tensor's position in the output
            ranges = ntuple(d -> (offsets[d][n] + 1):(offsets[d][n] + size(ttns[n].X, d)), nd)

            # Get the view of the destination block
            block = view(new_core, CartesianIndices(ranges))

            # Direct copy that preserves column-major order
            block .= ttns[n].X
        end

        return constructor(new_core, core_dims, child_groups, first(core_dims), similar(new_core))
    end
end

function get_leaf_x(t::AbstractTTN, idx::Int)
    idx < 1 && throw(BoundsError(t, idx))
    result = _find_leaf(t, idx, 1)
    result === nothing && throw(BoundsError(t, idx))
    return result
end

function half_reconstruct(t::AbstractTTN)
    contracted = t.X
    for (i,leaf) in enumerate(t.leaves)
        leafX = isleaf(leaf) ? leaf.X : transpose(matricize(leaf.X,0))
        contracted = n_mode_product(contracted,leafX,i)
    end
    return contracted
end

function rank_truncation(t::AbstractTTN, tol=1e-8, rs = nothing)
    constructor = constructorof(typeof(t))

    m = length(t.leaves)

    if isnothing(rs)
        r_min = 1
        r_max = 100
    else
        r_min, r_max = rs
    end
    tol_sq = tol*tol
    Ps = Vector{Matrix{Float64}}(undef,m)
    for i in 1:m
        leaf = t.leaves[i]

        P, Σ, _ = svd(matricize(t.X,i))
        ttol = tol_sq * sum(abs2,Σ)
        r = something(
            findfirst(x -> x <= ttol, rev_cumsum_sq(Σ)),
            length(Σ)+1) - 1
        r = clamp(r, r_min, r_max)
        Pr = @views @inbounds P[:,1:r]
        Ps[i] = Pr
        if isleaf(leaf)
            new_leaf = constructor(leaf.X*Pr)
            @reset t.leaves[i] = new_leaf
        else
            C = n_mode_product(leaf.X,transpose(Pr),0)
            leaf = constructor(C, leaf.leaves)
            @reset t.leaves[i] = rank_truncation(leaf,tol,rs)
        end
    end
    C = t.X
    for i in 1:m
        C = n_mode_product(C,transpose(Ps[i]),i)
    end

    return constructor(C,t.leaves)
end

function rev_cumsum_sq!(c,x)
    cumsum = zero(eltype(x))
    @inbounds for i in lastindex(x):-1:firstindex(x)
        cumsum += x[i]^2
        c[i] = cumsum
    end
    return c
end

function rev_cumsum_sq(x)
    c = similar(x)
    return rev_cumsum_sq!(c,x)
end