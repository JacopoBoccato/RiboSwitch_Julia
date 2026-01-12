"Linear Classifiers and Regressors"
# ============================================================================
# LOGISTIC REGRESSION (Binary Classification)
# ============================================================================

"""
    LogisticModel

A trained logistic regression model for binary classification.

# Fields
- `weights::Vector{Float64}`: Learned weight vector (includes bias if fit_bias=true)
- `convergence::Bool`: Whether optimization converged
- `final_loss::Float64`: Final loss value
- `iterations::Int`: Number of iterations taken
- `has_bias::Bool`: Whether model includes bias term
"""
struct LogisticModel
    weights::Vector{Float64}
    convergence::Bool
    final_loss::Float64
    iterations::Int
    has_bias::Bool
end

"""
    sigmoid(z)

Compute the sigmoid function: σ(z) = 1 / (1 + exp(-z))
"""
sigmoid(z) = 1 ./ (1 .+ exp.(-z))

"""
    predict(model::LogisticModel, X::Matrix{Float64})

Predict class probabilities for new data.

# Arguments
- `model::LogisticModel`: Trained model
- `X::Matrix{Float64}`: Feature matrix (features × samples)

# Returns
- `Vector{Float64}`: Predicted probabilities for class 1
"""
function predict(model::LogisticModel, X::Matrix{Float64})
    if model.has_bias
        w = model.weights[1:end-1]
        b = model.weights[end]
        return sigmoid.(X' * w .+ b)
    else
        return sigmoid.(X' * model.weights)
    end
end


"""
    predict_class(model::LogisticModel, X::Matrix{Float64}; threshold=0.5)

Predict class labels for new data.

# Arguments
- `model::LogisticModel`: Trained model
- `X::Matrix{Float64}`: Feature matrix (features × samples)
- `threshold::Float64`: Classification threshold (default: 0.5)

# Returns
- `Vector{Int}`: Predicted class labels (0 or 1)
"""
function predict_class(model::LogisticModel, X::Matrix{Float64}; threshold=0.5)
    probs = predict(model, X)
    return Int.(probs .>= threshold)
end

"""
    train_linear_classifier(X::Matrix{Float64}, y::Vector{Int}; 
                           regularization=0.01, max_iter=1000, fit_bias=true)

Train a logistic regression classifier using Optim.jl.

# Arguments
- `X::Matrix{Float64}`: Feature matrix of size (features × n_samples)
- `y::Vector{Int}`: Binary labels (0 or 1) of length n_samples
- `regularization::Float64`: L2 regularization parameter (default: 0.01)
- `max_iter::Int`: Maximum number of optimization iterations (default: 1000)
- `fit_bias::Bool`: Whether to include bias/intercept term (default: true)

# Returns
- `LogisticModel`: Trained model with weights and convergence information
"""
function train_linear_classifier(
    X::Matrix{Float64},
    y::Vector{Int};
    regularization = 0.01,
    lr = 1e-2,
    batch_size = 256,
    epochs = 10,
    fit_bias = true,
    shuffle = true
)
    n_features, n_samples = size(X)

    length(y) == n_samples ||
        error("Dimension mismatch: X has $n_samples samples but y has $(length(y)) labels")

    all(label -> label == 0 || label == 1, y) ||
        error("Labels must be binary (0 or 1)")

    # Parameters
    w = zeros(Float64, n_features)
    b = fit_bias ? 0.0 : 0.0

    y_float = Float64.(y)
    indices = collect(1:n_samples)

    for epoch in 1:epochs
        shuffle && Random.shuffle!(indices)

        for batch in Iterators.partition(indices, batch_size)
            Xb = @view X[:, batch]      # (features × batch)
            yb = @view y_float[batch]  # (batch)

            # Linear predictor
            z = Xb' * w                # (batch)
            fit_bias && (z .+= b)

            p = sigmoid.(z)
            err = p .- yb

            # Gradients
            gw = (Xb * err) / length(batch) .+ regularization .* w
            w .-= lr .* gw

            if fit_bias
                b -= lr * mean(err)
            end
        end
    end

    final_weights = fit_bias ? vcat(w, b) : w

    return LogisticModel(
        final_weights,
        true,        # convergence (SGD finished epochs)
        NaN,         # final_loss (or compute once if you want)
        epochs,
        fit_bias
    )
end
# ============================================================================
# LINEAR REGRESSION (Continuous Variable Prediction)
# ============================================================================

"""
    LinearRegressionModel

A trained linear regression model for continuous variable prediction.

# Fields
- `weights::Vector{Float64}`: Learned weight vector (includes bias if fit_bias=true)
- `convergence::Bool`: Whether optimization converged
- `final_loss::Float64`: Final loss value (MSE + regularization)
- `iterations::Int`: Number of iterations taken
- `has_bias::Bool`: Whether model includes bias term
"""
struct LinearRegressionModel
    weights::Vector{Float64}
    convergence::Bool
    final_loss::Float64
    iterations::Int
    has_bias::Bool
end

"""
    predict(model::LinearRegressionModel, X::Matrix{Float64})

Predict continuous values for new data.

# Arguments
- `model::LinearRegressionModel`: Trained model
- `X::Matrix{Float64}`: Feature matrix (features × samples)

# Returns
- `Vector{Float64}`: Predicted continuous values
"""
function predict(model::LinearRegressionModel, X::Matrix{Float64})
    if model.has_bias
        w = model.weights[1:end-1]
        b = model.weights[end]
        return vec(X' * w .+ b)
    else
        return vec(X' * model.weights)
    end
end


function train_linear_regressor(
    X::Matrix{Float64},
    y::Vector{Float64};
    regularization = 0.01,
    lr = 1e-2,
    batch_size = 256,
    epochs = 10,
    fit_bias = true,
    shuffle = true
)
    n_features, n_samples = size(X)

    length(y) == n_samples ||
        error("Dimension mismatch: X has $n_samples samples but y has $(length(y)) values")

    # Parameters
    w = zeros(Float64, n_features)
    b = 0.0

    indices = collect(1:n_samples)

    for epoch in 1:epochs
        shuffle && Random.shuffle!(indices)

        for batch in Iterators.partition(indices, batch_size)
            Xb = @view X[:, batch]
            yb = @view y[batch]

            ŷ = Xb' * w
            fit_bias && (ŷ .+= b)

            r = ŷ .- yb

            gw = (Xb * r) / length(batch) .+ regularization .* w
            w .-= lr .* gw

            fit_bias && (b -= lr * mean(r))
        end
    end

    # Final loss (single full pass)
    ŷ = X' * w
    fit_bias && (ŷ .+= b)
    mse = mean((ŷ .- y).^2) + regularization * sum(w.^2) / 2

    # ⬇⬇⬇ IMPORTANT PART ⬇⬇⬇
    final_weights = fit_bias ? vcat(w, b) : w

    return LinearRegressionModel(
        final_weights,
        true,       # convergence
        mse,
        epochs,
        fit_bias
    )
end
