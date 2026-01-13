#########################################
# Low-rank matrix/tensor decomposition utils for AdvRBMs #
#########################################
"""
    is_symmetric(Q; atol=1e-10)

Numerical symmetry check.
"""
is_symmetric(Q::AbstractMatrix; atol=1e-8) =
    norm(Q - Q', Inf) ≤ atol


"""
    lowrank_decomp(Q, r; atol=1e-10)

Returns:
- Qr    : rank-r approximation
- vals  : eigenvalues or singular values
- U     : left eigenvectors or left singular vectors
- V     : right eigenvectors or right singular vectors
- kind  : :eigen or :svd
"""
function lowrank_decomp(Q::AbstractMatrix, r::Integer; atol=1e-10)
    @assert size(Q,1) == size(Q,2)
    @assert 1 ≤ r ≤ size(Q,1)

    if is_symmetric(Q; atol)
        F = eigen(Symmetric(Q))
        idx = sortperm(abs.(F.values), rev=true)[1:r]

        vals = F.values[idx]
        U    = F.vectors[:, idx]
        V    = U
        Qr   = U * Diagonal(vals) * U'

        return Qr, vals, U, V, :eigen
    else
        F = svd(Q)
        idx = 1:r  # singular values already sorted

        vals = F.S[idx]
        U    = F.U[:, idx]
        V    = F.V[:, idx]
        Qr   = U * Diagonal(vals) * V'

        return Qr, vals, U, V, :svd
    end
end

"""
    lowrank_decomp_stack(Q, r)

Q :: (n, n, N)
"""
function lowrank_decomp_stack(Q::AbstractArray{<:Real,3}, r::Integer; atol=1e-8)
    n1, n2, N = size(Q)
    @assert n1 == n2

    Qr   = similar(Q, n1, n1, N)
    vals = zeros(eltype(Q), r, N)
    U    = zeros(eltype(Q), n1, r, N)
    V    = zeros(eltype(Q), n1, r, N)
    kind = Vector{Symbol}(undef, N)

    for i in 1:N
        Qi = view(Q, :, :, i)
        Qr[:,:,i], vals[:,i], U[:,:,i], V[:,:,i], kind[i] =
            lowrank_decomp(Qi, r; atol)
    end

    return Qr, vals, U, V, kind
end

"""
    lowrank_decomp_tensor(Q, r)

Tensorized operator of shape (d…, d…, N)
"""
function lowrank_decomp_tensor(Q::AbstractArray, r::Integer; atol=1e-8)
    nd = ndims(Q)
    d  = nd ÷ 2

    left  = size(Q)[1:d]
    right = size(Q)[d+1:2d]
    @assert left == right

    batch = nd > 2d ? size(Q)[end] : 1
    n = prod(left)

    Qmat = reshape(Q, n, n, batch)

    Qr_mat, vals, U, V, kind = lowrank_decomp_stack(Qmat, r; atol)

    Qr = reshape(Qr_mat, left..., right..., batch)

    return Qr, vals, U, V, kind
end
