using Pkg

# Activate the package environment (repo root)
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# Now the package is on LOAD_PATH
using RiboSwitch_Julia

# Optional: selective import
using RiboSwitch_Julia: RNA_ALPHABET, one_hot_encode, alphabet, predict,
                       count_alphabet_matrix, thermophilic_label, train_lowrank_bilinear_classifier, prevalence_threshold,
                       train_linear_classifier, train_linear_regressor, rna_onehot, confusion_matrix, display_confusion_matrix, plot_confusion_matrix, encode_indices
using Optim
using LinearAlgebra
using RestrictedBoltzmannMachines, AdvRBMs
using Statistics, DataFrames, CSV, Random
using Plots, MLUtils, Measures

df = CSV.read("/home/jacopo/RiboSwitch_Julia/artifacts/Ribo_aligned.tsv", DataFrame; delim='\t')
X, removed_idx = one_hot_encode(df.aligned_sequence, alphabet)
y = df.TOME_Predicted_OGT_Celsius
y = y[setdiff(1:end, removed_idx)]
labels = thermophilic_label(y; threshold=45)
println(mean(labels))

println(length(removed_idx), " sequences were removed due to invalid characters.")
"""
data = (X, labels)
(train_data, test_data) = splitobs(data, at=0.7, shuffle=true)

X_train = collect(train_data[1])
y_train = collect(train_data[2])

X_test = collect(test_data[1])
y_test = collect(test_data[2])

model = train_linear_classifier(X_train, y_train; regularization=0.01, epochs=2000, fit_bias=true, verbose=false)
threshold, idx = prevalence_threshold(model, X_test, y_test)

cm = confusion_matrix(
    model,
    X_test,
    y_test;
    threshold=threshold
)
display_confusion_matrix(cm)
plot_confusion_matrix(cm, "results/riboswitch_cm_linear.png")

X_counts, removed_idx = count_alphabet_matrix(alphabet, df.aligned_sequence)   
X = Float64.(X_counts)   # explicit, safe conversion

y = df.TOME_Predicted_OGT_Celsius
y = y[setdiff(1:end, removed_idx)]
labels = thermophilic_label(y; threshold=45) 

data = (X, labels)
(train_data, test_data) = splitobs(data, at=0.7, shuffle=true)

X_train = collect(train_data[1])
y_train = collect(train_data[2])

X_test = collect(test_data[1])
y_test = collect(test_data[2])

model = train_linear_classifier(X_train, y_train; regularization=0.01, epochs=2000, fit_bias=true, verbose=false)
threshold, idx = prevalence_threshold(model, X_test, y_test)

cm = confusion_matrix(
    model,
    X_test,
    y_test;
    threshold=threshold
)
display_confusion_matrix(cm)
plot_confusion_matrix(cm, "results/riboswitch_cm_counts.png")
"""
rna_idx, removed = encode_indices(df.aligned_sequence, RNA_ALPHABET, 5)
    
data = (rna_idx, labels)
(train_data, test_data) = splitobs(data, at=0.7, shuffle=true)

X_train = collect(train_data[1])
y_train = collect(train_data[2])

X_test = collect(test_data[1])
y_test = collect(test_data[2])

model = train_lowrank_bilinear_classifier(
    X_train,X_train, y_train, 
    rank=5, 
    alphabet_size=10,
    symmetric=false,
    max_epochs=2000,
    batch_size=256,
    learning_rate=0.02,
    momentum=0.9,
    weight_decay=2e-3,
    verbose=true
)
scores = predict(model, X_test, X_test)
@show minimum(scores), mean(scores), maximum(scores), std(scores)

threshold, idx = prevalence_threshold(model, X_test, X_test, y_test)

cm = confusion_matrix(
    model,
    X_test,
    X_test,
    y_test;
    threshold=threshold
)
display_confusion_matrix(cm)
plot_confusion_matrix(cm, "results/riboswitch_cm_bilinear10.png")