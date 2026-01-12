# ============================================================================
# LOW-RANK BILINEAR CLASSIFIER FOR SEQUENCE DATA
# ============================================================================
# This module provides memory-efficient bilinear classification for sequences
# using index-based encoding and low-rank factorization: x'UVz ≈ x'Wz
# ============================================================================

"""
    LowRankBilinearModel

A trained low-rank bilinear classifier for sequence data.
Uses factorization W ≈ UV' where U is (m × r) and V is (n × r).
Decision function: f(x,z) = x'UVz = (U'x)'(V'z)

# Fields
- `U::Matrix{Float32}`: Left factor matrix (m × r) in one-hot space
- `V::Matrix{Float32}`: Right factor matrix (n × r) in one-hot space
- `rank::Int`: Rank r of the factorization
- `x_length::Int`: Length of first sequence part (e.g., PDZ domain)
- `z_length::Int`: Length of second sequence part (e.g., peptide)
- `alphabet_size::Int`: Number of amino acids in alphabet
- `symmetric::Bool`: Whether U and V are constrained to be equal
- `convergence::Bool`: Whether optimization converged
- `final_loss::Float64`: Final loss value
- `iterations::Int`: Number of iterations taken
"""
struct LowRankBilinearModel
    U::Matrix{Float32}
    V::Matrix{Float32}
    rank::Int
    x_length::Int
    z_length::Int
    alphabet_size::Int
    symmetric::Bool
    convergence::Bool
    final_loss::Float64
    iterations::Int
end

"""
    BilinearCache

Workspace for computing bilinear forms without allocations.
"""
mutable struct BilinearCache
    a::Vector{Float32}  # U' * x (length r)
    b::Vector{Float32}  # V' * z (length r)
end

BilinearCache(r::Int) = BilinearCache(zeros(Float32, r), zeros(Float32, r))

"""
Clamp scores to prevent numerical overflow in sigmoid.
"""
@inline function clamp_score(s::Float32)
    return clamp(s, -20f0, 20f0)
end

"""
    loss_and_grad!(cache, g, params, x_idx, z_idx, y, m, n, r, symmetric)

Compute binary cross-entropy loss and gradient for index-encoded data.

This is the core optimization function that computes:
- Forward pass: score = (U'x)'(V'z) for each sample
- Loss: Binary cross-entropy
- Backward pass: Gradients w.r.t. U and V

# Arguments
- `cache::BilinearCache`: Workspace for intermediate computations
- `g::Vector{Float64}`: Output gradient vector (modified in-place)
- `params::Vector{Float64}`: Current parameters [U[:]; V[:]] (or just U if symmetric)
- `x_idx::Matrix{Int32}`: Indices for first sequence part (Lx × N)
- `z_idx::Matrix{Int32}`: Indices for second sequence part (Lz × N)
- `y::Vector{Float32}`: Binary labels in {0,1}
- `m::Int`: Dimension of U (alphabet_size * Lx)
- `n::Int`: Dimension of V (alphabet_size * Lz)
- `r::Int`: Rank of factorization
- `symmetric::Bool`: Whether to use U for both factors (V = U)

# Returns
- `Float32`: Loss value
"""
function loss_and_grad!(
    cache::BilinearCache,
    g::Vector{Float64},
    params::Vector{Float64},
    x_idx::Matrix{Int32},
    z_idx::Matrix{Int32},
    y::Vector{Float32},
    m::Int, n::Int, r::Int,
    symmetric::Bool
)
    N = size(x_idx, 2)
    @assert size(z_idx, 2) == N "x_idx and z_idx must have same number of samples"
    @assert length(y) == N "y must have same length as number of samples"

    fill!(g, 0.0)
    invN = 1f0 / Float32(N)
    loss = 0f0

    # Helper functions to access U and V from flattened params
    @inline function getU(i::Int, j::Int)
        return Float32(params[(j-1)*m + i])
    end
    
    @inline function addgU!(i::Int, j::Int, v::Float32)
        g[(j-1)*m + i] += Float64(v)
    end

    @inline function getV(i::Int, j::Int)
        base = m * r
        return Float32(params[base + (j-1)*n + i])
    end
    
    @inline function addgV!(i::Int, j::Int, v::Float32)
        base = m * r
        g[base + (j-1)*n + i] += Float64(v)
    end

    # Loop over all samples
    @inbounds for col in 1:N
        # Compute a = U' * x using indices
        fill!(cache.a, 0f0)
        for pos in 1:size(x_idx, 1)
            row = Int(x_idx[pos, col])
            for j in 1:r
                cache.a[j] += getU(row, j)
            end
        end

        # Compute b = V' * z (or U' * z if symmetric)
        fill!(cache.b, 0f0)
        if symmetric
            for pos in 1:size(z_idx, 1)
                row = Int(z_idx[pos, col])
                for j in 1:r
                    cache.b[j] += getU(row, j)
                end
            end
        else
            for pos in 1:size(z_idx, 1)
                row = Int(z_idx[pos, col])
                for j in 1:r
                    cache.b[j] += getV(row, j)
                end
            end
        end

        # Score: s = dot(a, b) = (U'x)' * (V'z)
        s = 0f0
        @simd for j in 1:r
            s += cache.a[j] * cache.b[j]
        end
        s = clamp_score(s)
        
        # Sigmoid prediction
        p = sigmoid(s)

        # Binary cross-entropy loss (with numerical stability)
        pc = clamp(p, 1f-7, 1f0 - 1f-7)
        yi = y[col]
        loss -= (yi * log(pc) + (1f0 - yi) * log(1f0 - pc)) * invN

        # Gradient: d(loss)/d(score)
        d = (p - yi) * invN

        # Backpropagate gradients
        if symmetric
            # Both parts use U
            for j in 1:r
                bj = cache.b[j]
                aj = cache.a[j]
                # Gradient from x part: d * (V'z) = d * b
                for pos in 1:size(x_idx, 1)
                    row = Int(x_idx[pos, col])
                    addgU!(row, j, d * bj)
                end
                # Gradient from z part: d * (U'x) = d * a
                for pos in 1:size(z_idx, 1)
                    row = Int(z_idx[pos, col])
                    addgU!(row, j, d * aj)
                end
            end
        else
            # Separate U and V
            for j in 1:r
                bj = cache.b[j]
                aj = cache.a[j]
                # Gradient for U: d * (V'z) = d * b
                for pos in 1:size(x_idx, 1)
                    row = Int(x_idx[pos, col])
                    addgU!(row, j, d * bj)
                end
                # Gradient for V: d * (U'x) = d * a
                for pos in 1:size(z_idx, 1)
                    row = Int(z_idx[pos, col])
                    addgV!(row, j, d * aj)
                end
            end
        end
    end

    return loss
end

"""
    minibatch_loss_and_grad!(cache, g, params, x_idx, z_idx, y, 
                             batch_indices, m, n, r, symmetric)

Compute loss and gradient for a single minibatch.

# Arguments
- `cache::BilinearCache`: Workspace for intermediate computations
- `g::Vector{Float64}`: Output gradient vector (modified in-place)
- `params::Vector{Float64}`: Current parameters
- `x_idx::Matrix{Int32}`: Full index matrix for first sequence part
- `z_idx::Matrix{Int32}`: Full index matrix for second sequence part
- `y::Vector{Float32}`: Full labels vector
- `batch_indices::Vector{Int}`: Indices of samples in this minibatch
- `m::Int`: Dimension of U
- `n::Int`: Dimension of V
- `r::Int`: Rank
- `symmetric::Bool`: Whether to use U for both factors

# Returns
- `Float32`: Loss value for this minibatch
"""
function minibatch_loss_and_grad!(
    cache::BilinearCache,
    g::Vector{Float64},
    params::Vector{Float64},
    x_idx::Matrix{Int32},
    z_idx::Matrix{Int32},
    y::Vector{Float32},
    batch_indices::Vector{Int},
    m::Int, n::Int, r::Int,
    symmetric::Bool
)
    batch_size = length(batch_indices)
    fill!(g, 0.0)
    inv_batch = 1f0 / Float32(batch_size)
    loss = 0f0

    # Helper functions to access U and V from flattened params
    @inline function getU(i::Int, j::Int)
        return Float32(params[(j-1)*m + i])
    end
    
    @inline function addgU!(i::Int, j::Int, v::Float32)
        g[(j-1)*m + i] += Float64(v)
    end

    @inline function getV(i::Int, j::Int)
        base = m * r
        return Float32(params[base + (j-1)*n + i])
    end
    
    @inline function addgV!(i::Int, j::Int, v::Float32)
        base = m * r
        g[base + (j-1)*n + i] += Float64(v)
    end

    # Loop over minibatch samples
    @inbounds for idx in batch_indices
        # Compute a = U' * x
        fill!(cache.a, 0f0)
        for pos in 1:size(x_idx, 1)
            row = Int(x_idx[pos, idx])
            for j in 1:r
                cache.a[j] += getU(row, j)
            end
        end

        # Compute b = V' * z (or U' * z if symmetric)
        fill!(cache.b, 0f0)
        if symmetric
            for pos in 1:size(z_idx, 1)
                row = Int(z_idx[pos, idx])
                for j in 1:r
                    cache.b[j] += getU(row, j)
                end
            end
        else
            for pos in 1:size(z_idx, 1)
                row = Int(z_idx[pos, idx])
                for j in 1:r
                    cache.b[j] += getV(row, j)
                end
            end
        end

        # Score and prediction
        s = 0f0
        @simd for j in 1:r
            s += cache.a[j] * cache.b[j]
        end
        s = clamp_score(s)
        p = sigmoid(s)

        # Binary cross-entropy loss
        pc = clamp(p, 1f-7, 1f0 - 1f-7)
        yi = y[idx]
        loss -= (yi * log(pc) + (1f0 - yi) * log(1f0 - pc)) * inv_batch

        # Gradient
        d = (p - yi) * inv_batch

        # Backpropagate
        if symmetric
            for j in 1:r
                bj = cache.b[j]
                aj = cache.a[j]
                for pos in 1:size(x_idx, 1)
                    row = Int(x_idx[pos, idx])
                    addgU!(row, j, d * bj)
                end
                for pos in 1:size(z_idx, 1)
                    row = Int(z_idx[pos, idx])
                    addgU!(row, j, d * aj)
                end
            end
        else
            for j in 1:r
                bj = cache.b[j]
                aj = cache.a[j]
                for pos in 1:size(x_idx, 1)
                    row = Int(x_idx[pos, idx])
                    addgU!(row, j, d * bj)
                end
                for pos in 1:size(z_idx, 1)
                    row = Int(z_idx[pos, idx])
                    addgV!(row, j, d * aj)
                end
            end
        end
    end

    return loss
end

"""
    train_lowrank_bilinear_classifier(x_idx::Matrix{Int32}, z_idx::Matrix{Int32},
                                      y::Vector{Int}, rank::Int;
                                      alphabet_size::Int=20,
                                      symmetric::Bool=false,
                                      max_epochs::Int=50,
                                      batch_size::Int=32,
                                      learning_rate::Float64=0.01,
                                      momentum::Float64=0.9,
                                      weight_decay::Float64=1e-4,
                                      verbose::Bool=false)

Train a low-rank bilinear classifier using Stochastic Gradient Descent with minibatches.

This function learns a factorized bilinear form W ≈ UV' where:
- U is (m × r) with m = alphabet_size * size(x_idx, 1)
- V is (n × r) with n = alphabet_size * size(z_idx, 1)
- Decision function: f(x,z) = x'UVz

Uses SGD with momentum and weight decay for optimization.

# Arguments
- `x_idx::Matrix{Int32}`: Index matrix for first sequence part (Lx × N)
- `z_idx::Matrix{Int32}`: Index matrix for second sequence part (Lz × N)
- `y::Vector{Int}`: Binary labels (0 or 1) of length N
- `rank::Int`: Rank r of the factorization (controls model complexity)
- `alphabet_size::Int`: Size of amino acid alphabet (default: 20)
- `symmetric::Bool`: Constrain V = U for symmetric bilinear form (default: false)
- `max_epochs::Int`: Maximum number of epochs (default: 50)
- `batch_size::Int`: Minibatch size (default: 32)
- `learning_rate::Float64`: Learning rate for SGD (default: 0.01)
- `momentum::Float64`: Momentum coefficient (default: 0.9)
- `weight_decay::Float64`: L2 regularization strength (default: 1e-4)
- `verbose::Bool`: Print training progress (default: false)

# Returns
- `LowRankBilinearModel`: Trained model
"""
function train_lowrank_bilinear_classifier(
    x_idx::Matrix{Int32},
    z_idx::Matrix{Int32},
    y::Vector{Int};
    rank::Int,
    alphabet_size::Int=20,
    symmetric::Bool=false,
    max_epochs::Int=50,
    batch_size::Int=32,
    learning_rate::Float64=0.01,
    momentum::Float64=0.9,
    weight_decay::Float64=1e-4,
    verbose::Bool=false
)
    N = size(x_idx, 2)
    
    # Input validation
    if size(z_idx, 2) != N
        error("Dimension mismatch: x_idx has $N samples but z_idx has $(size(z_idx, 2)) samples")
    end
    
    if length(y) != N
        error("Dimension mismatch: indices have $N samples but y has $(length(y)) labels")
    end
    
    if !all(label -> label in [0, 1], y)
        error("Labels must be binary (0 or 1). Found: $(unique(y))")
    end
    
    if rank < 1
        error("Rank must be at least 1, got $rank")
    end
    
    if batch_size < 1 || batch_size > N
        error("Batch size must be in range [1, $N], got $batch_size")
    end

    # Convert labels to Float32
    y_float = Float32.(y)
    
    # Dimensions in one-hot space
    Lx = size(x_idx, 1)
    Lz = size(z_idx, 1)
    m = alphabet_size * Lx
    n = alphabet_size * Lz
    
    # Initialize parameters with Xavier/He initialization
    scale = sqrt(2.0 / (m + n))
    if symmetric
        params = randn(Float64, m * rank) .* scale
        velocity = zeros(Float64, m * rank)
    else
        params = vcat(randn(Float64, m * rank) .* scale, 
                     randn(Float64, n * rank) .* scale)
        velocity = zeros(Float64, length(params))
    end

    # Create workspace
    cache = BilinearCache(rank)
    g = zeros(Float64, length(params))
    
    # Training loop
    n_batches = ceil(Int, N / batch_size)
    all_indices = collect(1:N)
    
    best_loss = Inf
    epochs_without_improvement = 0
    final_epoch = max_epochs
    
    if verbose
        println("Training with SGD:")
        println("  Samples: $N")
        println("  Batch size: $batch_size")
        println("  Batches per epoch: $n_batches")
        println("  Rank: $rank")
        println("  Symmetric: $symmetric")
        println()
    end
    
    for epoch in 1:max_epochs
        # Shuffle data at the start of each epoch
        shuffle!(all_indices)
        
        epoch_loss = 0.0
        
        # Process minibatches
        for batch in 1:n_batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, N)
            batch_indices = all_indices[start_idx:end_idx]
            
            # Compute gradient for this minibatch
            batch_loss = minibatch_loss_and_grad!(cache, g, params, x_idx, z_idx, 
                                                  y_float, batch_indices, 
                                                  m, n, rank, symmetric)
            
            # Add weight decay (L2 regularization) to gradient
            @. g += weight_decay * params
            
            # SGD with momentum update
            @. velocity = momentum * velocity - learning_rate * g
            @. params += velocity
            
            epoch_loss += batch_loss * length(batch_indices)
        end
        
        epoch_loss /= N
        
        # Check for improvement
        if epoch_loss < best_loss
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end
        
        # Print progress
        if verbose && (epoch % 5 == 0 || epoch == 1)
            println("Epoch $epoch/$max_epochs - Loss: $(round(epoch_loss, digits=6))")
        end
        
        # Early stopping if loss plateaus
        if epochs_without_improvement >= 20
            if verbose
                println("Early stopping at epoch $epoch (no improvement for 20 epochs)")
            end
            final_epoch = epoch
            break
        end
    end
    
    # Reshape into U and V matrices
    if symmetric
        U = reshape(Float32.(params), m, rank)
        V = U
    else
        U = reshape(Float32.(params[1:m*rank]), m, rank)
        V = reshape(Float32.(params[m*rank+1:end]), n, rank)
    end

    if verbose
        println("\nTraining completed!")
        println("  Final loss: $(round(best_loss, digits=6))")
        println("  Epochs: $final_epoch")
    end

    return LowRankBilinearModel(U, V, rank, Lx, Lz, alphabet_size, 
                               symmetric, true, best_loss, final_epoch)
end

"""
    predict(model::LowRankBilinearModel, x_idx::Matrix{Int32}, z_idx::Matrix{Int32})

Predict class probabilities using a trained low-rank bilinear model.

# Arguments
- `model::LowRankBilinearModel`: Trained model
- `x_idx::Matrix{Int32}`: Index matrix for first sequence part (Lx × N)
- `z_idx::Matrix{Int32}`: Index matrix for second sequence part (Lz × N)

# Returns
- `Vector{Float32}`: Predicted probabilities for class 1
"""
function predict(model::LowRankBilinearModel, x_idx::Matrix{Int32}, z_idx::Matrix{Int32})
    N = size(x_idx, 2)
    
    @assert size(z_idx, 2) == N "x_idx and z_idx must have same number of samples"
    @assert size(x_idx, 1) == model.x_length "x_idx has wrong length dimension"
    @assert size(z_idx, 1) == model.z_length "z_idx has wrong length dimension"
    
    predictions = zeros(Float32, N)
    a = zeros(Float32, model.rank)
    b = zeros(Float32, model.rank)
    
    @inbounds for col in 1:N
        # Compute a = U' * x
        fill!(a, 0f0)
        for pos in 1:size(x_idx, 1)
            row = Int(x_idx[pos, col])
            for j in 1:model.rank
                a[j] += model.U[row, j]
            end
        end
        
        # Compute b = V' * z
        fill!(b, 0f0)
        for pos in 1:size(z_idx, 1)
            row = Int(z_idx[pos, col])
            for j in 1:model.rank
                b[j] += model.V[row, j]
            end
        end
        
        # Score and sigmoid
        s = 0f0
        @simd for j in 1:model.rank
            s += a[j] * b[j]
        end
        predictions[col] = sigmoid(clamp_score(s))
    end
    
    return predictions
end

"""
    predict_class(model::LowRankBilinearModel, x_idx::Matrix{Int32}, 
                  z_idx::Matrix{Int32}; threshold=0.5)

Predict class labels using a trained low-rank bilinear model.

# Arguments
- `model::LowRankBilinearModel`: Trained model
- `x_idx::Matrix{Int32}`: Index matrix for first sequence part
- `z_idx::Matrix{Int32}`: Index matrix for second sequence part
- `threshold::Float64`: Classification threshold (default: 0.5)

# Returns
- `Vector{Int}`: Predicted class labels (0 or 1)
"""
function predict_class(model::LowRankBilinearModel, x_idx::Matrix{Int32}, 
                      z_idx::Matrix{Int32}; threshold=0.5)
    probs = predict(model, x_idx, z_idx)
    return Int.(probs .>= threshold)
end

function prevalence_threshold(
    model,
    X::AbstractMatrix,
    Z::AbstractMatrix,
    y::AbstractVector{<:Integer}
)
    n_features, n_samples = size(X)
    length(y) == n_samples ||
        error("Dimension mismatch: X has $n_samples samples but y has $(length(y)) labels")

    all(label -> label == 0 || label == 1, y) ||
        error("Labels must be binary (0 or 1)")

    # --- Model predictions ---
    scores = predict(model, X, Z)
    length(scores) == n_samples ||
        error("predict(model, X, Z) must return one score per sample")

    scores = Float64.(scores)  # ensure numeric stability

    # --- Number of true positives ---
    n_pos = sum(y)

    n_pos == 0 && error("No positive examples in y")
    n_pos == n_samples && return -Inf, collect(1:n_samples)

    # --- Sort by descending score ---
    sorted_idx = sortperm(scores; rev=true)

    # --- Threshold selection ---
    threshold = scores[sorted_idx[n_pos]]

    return threshold, sorted_idx
end
