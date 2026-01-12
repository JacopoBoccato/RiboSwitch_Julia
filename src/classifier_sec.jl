"""
    BilinearModel

A trained bilinear (quadratic) classifier for binary classification.
Decision function: f(x) = x'Wx + b (if bias) or f(x) = x'Wx (no bias)

# Fields
- `W::Matrix{Float64}`: Learned weight matrix (n_features × n_features)
- `bias::Float64`: Learned bias term (0.0 if fit_bias=false)
- `convergence::Bool`: Whether optimization converged
- `final_loss::Float64`: Final loss value
- `iterations::Int`: Number of iterations taken
- `has_bias::Bool`: Whether model includes bias term
"""

struct BilinearModel
    W::Matrix{Float64}
    bias::Float64
    convergence::Bool
    final_loss::Float64
    iterations::Int
    has_bias::Bool
end

"""
    bilinear_transform(x::Vector{Float64}, W::Matrix{Float64})

Compute the bilinear form x'Wx for a single sample.
"""
function bilinear_transform(x::Vector{Float64}, W::Matrix{Float64})
    return dot(x, W * x)
end

"""
    bilinear_transform(X::Matrix{Float64}, W::Matrix{Float64})

Compute the bilinear form x'Wx for multiple samples (columns of X).
Returns a vector of transformed values.
"""
function bilinear_transform(X::Matrix{Float64}, W::Matrix{Float64})
    n_samples = size(X, 2)
    result = zeros(n_samples)
    for i in 1:n_samples
        x = X[:, i]
        result[i] = dot(x, W * x)
    end
    return result
end

"""
    predict(model::BilinearModel, X::Matrix{Float64})

Predict class probabilities for new data using bilinear transformation.

# Arguments
- `model::BilinearModel`: Trained bilinear model
- `X::Matrix{Float64}`: Feature matrix (features × samples)

# Returns
- `Vector{Float64}`: Predicted probabilities for class 1
"""
function predict(model::BilinearModel, X::Matrix{Float64})
    quadratic_scores = bilinear_transform(X, model.W)
    
    if model.has_bias
        quadratic_scores .+= model.bias
    end
    
    return sigmoid(quadratic_scores)
end

"""
    predict_class(model::BilinearModel, X::Matrix{Float64}; threshold=0.5)

Predict class labels for new data.

# Arguments
- `model::BilinearModel`: Trained bilinear model
- `X::Matrix{Float64}`: Feature matrix (features × samples)
- `threshold::Float64`: Classification threshold (default: 0.5)

# Returns
- `Vector{Int}`: Predicted class labels (0 or 1)
"""
function predict_class(model::BilinearModel, X::Matrix{Float64}; threshold=0.5)
    probs = predict(model, X)
    return Int.(probs .>= threshold)
end

"""
    params_to_model(params::Vector{Float64}, n_features::Int, has_bias::Bool)

Convert parameter vector to W matrix and bias.
"""
function params_to_model(params::Vector{Float64}, n_features::Int, has_bias::Bool)
    n_W_params = n_features * n_features
    W = reshape(params[1:n_W_params], n_features, n_features)
    bias = has_bias ? params[end] : 0.0
    return W, bias
end

"""
    model_to_params(W::Matrix{Float64}, bias::Float64, has_bias::Bool)

Convert W matrix and bias to parameter vector.
"""
function model_to_params(W::Matrix{Float64}, bias::Float64, has_bias::Bool)
    params = vec(W)
    if has_bias
        push!(params, bias)
    end
    return params
end

"""
    train_bilinear_classifier(X::Matrix{Float64}, y::Vector{Int}; 
                              regularization=0.01, max_iter=1000, 
                              fit_bias=true, symmetric=true)

Train a bilinear (quadratic) classifier using Optim.jl.
Decision boundary: x'Wx + b = 0 (with bias) or x'Wx = 0 (without bias)

# Arguments
- `X::Matrix{Float64}`: Feature matrix of size (features × n_samples)
- `y::Vector{Int}`: Binary labels (0 or 1) of length n_samples
- `regularization::Float64`: L2 regularization parameter (default: 0.01)
- `max_iter::Int`: Maximum number of optimization iterations (default: 1000)
- `fit_bias::Bool`: Whether to include bias/intercept term (default: true)
- `symmetric::Bool`: Whether to constrain W to be symmetric (default: true)

# Returns
- `BilinearModel`: Trained model with weight matrix W, bias, and convergence info
"""
function train_bilinear_classifier(X::Matrix{Float64}, y::Vector{Int}; 
                                  regularization=0.01, max_iter=1000, 
                                  fit_bias=true, symmetric=true)
    # Input validation
    n_features, n_samples = size(X)
    
    if length(y) != n_samples
        error("Dimension mismatch: X has $n_samples samples but y has $(length(y)) labels")
    end
    
    if !all(label -> label in [0, 1], y)
        error("Labels must be binary (0 or 1). Found: $(unique(y))")
    end
    
    # Convert y to Float64 for computation
    y_float = Float64.(y)
    
    # Number of parameters
    n_W_params = n_features * n_features
    n_params = fit_bias ? n_W_params + 1 : n_W_params
    
    # Define loss function (binary cross-entropy with L2 regularization)
    function loss(params)
        W, bias = params_to_model(params, n_features, fit_bias)
        
        # Make W symmetric if requested
        if symmetric
            W = (W + W') / 2
        end
        
        # Compute bilinear scores: x'Wx for each sample
        scores = bilinear_transform(X, W)
        
        if fit_bias
            scores .+= bias
        end
        
        # Apply sigmoid
        predictions = sigmoid(scores)
        
        # Clip predictions to avoid log(0)
        predictions = clamp.(predictions, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        nll = -mean(y_float .* log.(predictions) .+ (1 .- y_float) .* log.(1 .- predictions))
        
        # L2 regularization on W (not on bias)
        reg_term = regularization * sum(W.^2) / 2
        
        return nll + reg_term
    end
    
    # Define gradient
    function gradient!(G, params)
        W, bias = params_to_model(params, n_features, fit_bias)
        
        # Make W symmetric if requested
        if symmetric
            W = (W + W') / 2
        end
        
        # Compute bilinear scores
        scores = bilinear_transform(X, W)
        
        if fit_bias
            scores .+= bias
        end
        
        # Predictions and errors
        predictions = sigmoid(scores)
        errors = predictions .- y_float
        
        # Gradient w.r.t. W
        # d/dW[x'Wx] = xx' + (xx')' = 2xx' for symmetric W, xx' for general W
        G_W = zeros(n_features, n_features)
        for i in 1:n_samples
            x = X[:, i]
            xx_outer = x * x'
            if symmetric
                G_W .+= 2 * errors[i] * xx_outer
            else
                # For general W: d/dW[x'Wx] = xx' (gradient is not symmetric)
                G_W .+= errors[i] * xx_outer
            end
        end
        G_W ./= n_samples
        
        # Add regularization gradient
        G_W .+= regularization * W
        
        # Gradient w.r.t. bias
        if fit_bias
            G_bias = mean(errors)
            G .= model_to_params(G_W, G_bias, fit_bias)
        else
            G .= vec(G_W)
        end
        
        return G
    end
    
    # Initialize parameters
    # Start with small random values for W
    W0 = 0.01 * randn(n_features, n_features)
    if symmetric
        W0 = (W0 + W0') / 2  # Make initial W symmetric
    end
    bias0 = 0.0
    params0 = model_to_params(W0, bias0, fit_bias)
    
    # Optimize using L-BFGS
    result = optimize(loss, gradient!, params0, LBFGS(), 
                     Optim.Options(iterations=max_iter, show_trace=false))
    
    # Extract results
    params_opt = Optim.minimizer(result)
    W_opt, bias_opt = params_to_model(params_opt, n_features, fit_bias)
    
    # Enforce symmetry one final time if requested
    if symmetric
        W_opt = (W_opt + W_opt') / 2
    end
    
    converged = Optim.converged(result)
    final_loss = Optim.minimum(result)
    iterations = Optim.iterations(result)
    
    if !converged
        @warn "Optimization did not converge after $iterations iterations"
    end
    
    return BilinearModel(W_opt, bias_opt, converged, final_loss, iterations, fit_bias)
end

"""
    get_decision_boundary_level(model::BilinearModel)

Get the decision boundary level (useful for visualization).
For a bilinear model, this is 0 (the level set where x'Wx + b = 0).
"""
