struct TTNO{T,M,N,A<:AbstractArray{T,M},Leaves<:NTuple{N}} <: AbstractTTN{T,M,N,A,Leaves}
    X::A                    # Connection tensor C_τ or basis matrix
    size::NTuple{M,Int}
    leaves::Leaves          # Child nodes τ₁, ..., τₘ
    r::Int
    tmp::A
end

function TTNO(C::AbstractArray{T}, leaves::Leaves) where {T,Leaves <: Tuple}
    s = size(C)
    TTNO{T,ndims(C),length(leaves),typeof(C),Leaves}(C, s, leaves, s[end], similar(C))
end

function TTNO(C::AbstractArray{T}, leaves::Leaves, tmp::AbstractArray{T}) where {T,Leaves <: Tuple}
    s = size(C)
    TTNO{T,ndims(C),length(leaves),typeof(C),Leaves}(C, s, leaves, s[end], tmp)
end

function TTNO(U::AbstractMatrix{T}) where {T}
    s = size(U)
    TTNO{T,ndims(U),0,typeof(U),Tuple{}}(U, s, (), s[end], similar(U))
end

function TTNO(U::AbstractMatrix{T},tmp::AbstractArray{T}) where {T}
    s = size(U)
    TTNO{T,ndims(U),0,typeof(U),Tuple{}}(U, s, (), s[end], tmp)
end

function set_TTNO_cores_Sylv(t::AbstractTTN)
        
    # If not a leaf, recursively traverse children
    if !isleaf(t)
        n = length(t.leaves)
        n = isroot(t) ? n : n + 1

        leaves = map(t.leaves) do leaf
            set_TTNO_cores(leaf)
        end

        C = zeros(eltype(t),ntuple(i->2,n))
        idxs = reverse(ntuple(i->CartesianIndex(ntuple(j-> j==i ? 2 : 1 ,n)),n))
        for idx in idxs
            C[idx] = one(eltype(t))
        end
        t = TTNO(C,leaves)
    end

    return t
end

function set_TTNO_cores(t::AbstractTTN)
        
    # If not a leaf, recursively traverse children
    if !isleaf(t)
        n = length(t.leaves)
        n = isroot(t) ? n : n + 1

        leaves = map(t.leaves) do leaf
            set_TTNO_cores(leaf)
        end

        t = TTNO(ones(ntuple(i->1,n)),leaves)
    end

    return t
end

function replace_leaves(ttno::AbstractTTN, replacements)
    counter = Ref(1)
    new_tree = _replace_rec(ttno, replacements, counter)

    # sanity check – all replacements were used
    if counter[] - 1 != length(replacements)
        error("Expected $(length(replacements)) leaves, found $(counter[]-1)")
    end

    return new_tree
end

function _replace_rec(t::AbstractTTN, repls, counter::Ref{Int})
    if isleaf(t)                     
        # ---- deepest leaf ----
        idx = counter[]
        repl = repls[idx]
        counter[] += 1
        t = TTNO(repl,())
    else
        # ---- internal node – recurse into leaf child ----
        new_leaves = map(t.leaves) do leaf
            _replace_rec(leaf, repls, counter)
        end
        t = TTNO(t.X, new_leaves)
    end
    return t
end

function Sylvester_TTNO(t, Ds)

    Sylv_TTNO = set_TTNO_cores_Sylv(t)
    d = length(Ds)
    Us = ntuple(d) do i
        D = Ds[i]
        n = size(D,1)
        len = n*n
        U = Matrix(hcat(reshape(one(eltype(D))*I(n),len,1),reshape((Ds[i]),len,1)))
        U
    end

    ttno = replace_leaves(Sylv_TTNO,Us)
    return ttno
end

# function Sylvester_TTNO(t, Ds)

#     Sylv_TTNO = set_TTNO_cores(t)
#     d = length(Ds)
#     ttnos = ntuple(d) do i
#         D = Ds[i]
#         n = size(D,1)
#         len = n*n
#         Us = ntuple(j-> j == i ? reshape((Ds[i]),len,1) : reshape(one(eltype(D))*I(n),len,1), d)
#         # U = reduce(hcat ∘ Matrix, Us)
#         Us = map(Matrix, Us)
#         replace_leaves(Sylv_TTNO,Us)
#     end
#     ttno = add_TTNs(ttnos,1e-14)
#     # ttno = replace_leaves(Sylv_TTNO,Us)
#     return ttno
# end

function apply_TTNO(op::TTNO,t::TTN)
    result = _apply_TTNO(op,t)
    # result = rank_truncation(result,1e-14) 
    return result #orth_n_trunc_ttn!(result)
end

function _apply_TTNO(op::TTNO,t::TTN)
   if isleaf(t)
        s = op.r
        n = size(t.X,1)
        Us = ntuple(s) do i
            A = reshape((@views op.X[:,i]), n,n)    
            A*t.X
        end
        U = reduce(hcat,Us)
        return TTN(U)
    else
        # Process children with ntuple for stack allocation
        leaves = ntuple(length(t.leaves)) do i
            _apply_TTNO(op.leaves[i],t.leaves[i])
        end
        new_core = kronecker(op.X,t.X)
        return TTN(new_core, leaves)
    end
end