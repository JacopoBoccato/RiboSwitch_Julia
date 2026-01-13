# Generate synthetic data
n_features, n_samples = 3, 100
X = randn(n_features, n_samples)
y = Int.(rand(n_samples) .> 0.5)

# Train model with bias
model = train_linear_classifier(X, y)

# Train model without bias
model_no_bias = train_linear_classifier(X, y, fit_bias=false)

# Make predictions
X_test = randn(n_features, 10)
probs = predict(model, X_test)
labels = predict_class(model, X_test)


n_features, n_samples = 5, 200
X = randn(n_features, n_samples)
y = vec(randn(1, n_samples) * X) .+ 0.1 * randn(n_samples)  # Linear relation + noise

# Train model with bias
model = train_linear_regressor(X, y)

# Train model without bias (data passes through origin)
model_no_bias = train_linear_regressor(X, y, fit_bias=false)

# Make predictions
X_test = randn(n_features, 10)
y_pred = predict(model, X_test)

# Evaluate
y_test = vec(randn(1, 10) * X_test)
mse = mean((y_pred .- y_test).^2)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

function demo_classifier()
    println("=== Logistic Classifier Demo ===\n")
    
    # Generate synthetic data
    using Random
    Random.seed!(42)
    n_features, n_samples = 5, 200
    
    # Create linearly separable data
    w_true = randn(n_features)
    X = randn(n_features, n_samples)
    y_prob = sigmoid(w_true' * X .+ 0.5)
    y = Int.(rand(n_samples) .< vec(y_prob))
    
    println("Training on $n_samples samples with $n_features features")
    println("Class distribution: $(sum(y)) positive, $(n_samples - sum(y)) negative\n")
    
    # Train model with bias
    println("--- With Bias ---")
    model = train_linear_classifier(X, y, regularization=0.01)
    println("  Converged: $(model.convergence)")
    println("  Final loss: $(round(model.final_loss, digits=4))")
    println("  Iterations: $(model.iterations)")
    y_pred = predict_class(model, X)
    accuracy = mean(y_pred .== y)
    println("  Training accuracy: $(round(accuracy * 100, digits=2))%\n")
    
    # Train model without bias
    println("--- Without Bias ---")
    model_no_bias = train_linear_classifier(X, y, regularization=0.01, fit_bias=false)
    println("  Converged: $(model_no_bias.convergence)")
    println("  Final loss: $(round(model_no_bias.final_loss, digits=4))")
    println("  Iterations: $(model_no_bias.iterations)")
    y_pred_no_bias = predict_class(model_no_bias, X)
    accuracy_no_bias = mean(y_pred_no_bias .== y)
    println("  Training accuracy: $(round(accuracy_no_bias * 100, digits=2))%\n")
end

function demo_regressor()
    println("=== Linear Regressor Demo ===\n")
    
    # Generate synthetic data
    using Random
    Random.seed!(42)
    n_features, n_samples = 5, 200
    
    # Create linear relationship
    w_true = randn(n_features)
    bias_true = 2.5
    X = randn(n_features, n_samples)
    y = vec(w_true' * X) .+ bias_true .+ 0.5 * randn(n_samples)
    
    println("Training on $n_samples samples with $n_features features")
    println("True bias: $bias_true\n")
    
    # Train model with bias
    println("--- With Bias ---")
    model = train_linear_regressor(X, y, regularization=0.01)
    println("  Converged: $(model.convergence)")
    println("  Final loss: $(round(model.final_loss, digits=4))")
    println("  Iterations: $(model.iterations)")
    println("  Learned bias: $(round(model.weights[end], digits=3))")
    y_pred = predict(model, X)
    mse = mean((y_pred .- y).^2)
    r2 = 1 - sum((y .- y_pred).^2) / sum((y .- mean(y)).^2)
    println("  Training MSE: $(round(mse, digits=4))")
    println("  R²: $(round(r2, digits=4))\n")
    
    # Train model without bias
    println("--- Without Bias ---")
    model_no_bias = train_linear_regressor(X, y, regularization=0.01, fit_bias=false)
    println("  Converged: $(model_no_bias.convergence)")
    println("  Final loss: $(round(model_no_bias.final_loss, digits=4))")
    println("  Iterations: $(model_no_bias.iterations)")
    y_pred_no_bias = predict(model_no_bias, X)
    mse_no_bias = mean((y_pred_no_bias .- y).^2)
    r2_no_bias = 1 - sum((y .- y_pred_no_bias).^2) / sum((y .- mean(y)).^2)
    println("  Training MSE: $(round(mse_no_bias, digits=4))")
    println("  R²: $(round(r2_no_bias, digits=4))\n")
end

# ============================================================================
# DEMO FUNCTION
# ============================================================================

function demo_bilinear_classifier()
    println("=== Bilinear Classifier Demo ===\n")
    
    using Random
    Random.seed!(42)
    
    # Generate synthetic data with quadratic decision boundary
    n_features, n_samples = 3, 300
    X = randn(n_features, n_samples)
    
    # True quadratic decision boundary: x1² + 2*x2² - x3² + 0.5 > 0
    W_true = diagm([1.0, 2.0, -1.0])
    bias_true = 0.5
    
    scores = bilinear_transform(X, W_true) .+ bias_true
    y = Int.(scores .> 0)
    
    println("Generated data with quadratic decision boundary")
    println("True W diagonal: [1.0, 2.0, -1.0]")
    println("True bias: $bias_true")
    println("Class distribution: $(sum(y)) positive, $(n_samples - sum(y)) negative\n")
    
    # Train with bias
    println("--- Training with Bias (symmetric W) ---")
    model = train_bilinear_classifier(X, y, regularization=0.01, fit_bias=true, symmetric=true)
    println("  Converged: $(model.convergence)")
    println("  Final loss: $(round(model.final_loss, digits=4))")
    println("  Iterations: $(model.iterations)")
    println("  Learned W diagonal: $(round.(diag(model.W), digits=3))")
    println("  Learned bias: $(round(model.bias, digits=3))")
    
    y_pred = predict_class(model, X)
    accuracy = mean(y_pred .== y)
    println("  Training accuracy: $(round(accuracy * 100, digits=2))%\n")
    
    # Train without bias
    println("--- Training without Bias (symmetric W) ---")
    # Generate new data without bias for fair comparison
    scores_no_bias = bilinear_transform(X, W_true)
    y_no_bias = Int.(scores_no_bias .> 0)
    
    model_no_bias = train_bilinear_classifier(X, y_no_bias, regularization=0.01, 
                                              fit_bias=false, symmetric=true)
    println("  Converged: $(model_no_bias.convergence)")
    println("  Final loss: $(round(model_no_bias.final_loss, digits=4))")
    println("  Iterations: $(model_no_bias.iterations)")
    println("  Learned W diagonal: $(round.(diag(model_no_bias.W), digits=3))")
    
    y_pred_no_bias = predict_class(model_no_bias, X)
    accuracy_no_bias = mean(y_pred_no_bias .== y_no_bias)
    println("  Training accuracy: $(round(accuracy_no_bias * 100, digits=2))%\n")
    
    # Comparison with linear classifier
    println("--- Why Bilinear? Comparison with Linear Classifier ---")
    println("Linear classifiers cannot capture quadratic decision boundaries.")
    println("Bilinear form x'Wx allows modeling ellipsoidal, hyperbolic,")
    println("and other quadratic decision surfaces.\n")
end

# Example
# Generate synthetic data with quadratic decision boundary
n_features, n_samples = 3, 200
X = randn(n_features, n_samples)

# Create quadratic decision boundary: x1² + x2² - x3² > 0
W_true = diagm([1.0, 1.0, -1.0])
scores = [dot(X[:, i], W_true * X[:, i]) for i in 1:n_samples]
y = Int.(scores .> 0)

# Train bilinear classifier
model = train_bilinear_classifier(X, y)

# Make predictions
X_test = randn(n_features, 10)
probs = predict(model, X_test)
labels = predict_class(model, X_test)

# Train without bias (decision boundary passes through origin)
model_no_bias = train_bilinear_classifier(X, y, fit_bias=false)
