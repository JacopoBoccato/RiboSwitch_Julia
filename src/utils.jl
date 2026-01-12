"""
Useful function to preprocess sequences
"""

function count_alphabet_matrix(alphabet, sequences)
    # Convert alphabet to index lookup
    alpha_index = Dict(a => i for (i, a) in enumerate(alphabet))
    valid_chars = Set(alphabet)
    n_alpha = length(alphabet)

    isempty(sequences) && error("No sequences provided")

    # Indices of valid sequences
    valid_idx = [i for (i, s) in enumerate(sequences) if all(c -> c in valid_chars, s)]
    removed_idx = setdiff(1:length(sequences), valid_idx)

    isempty(valid_idx) && error("All sequences contain invalid characters")

    valid_seqs = sequences[valid_idx]

    # Length check AFTER filtering
    seq_length = length(valid_seqs[1])
    all(length(s) == seq_length for s in valid_seqs) ||
        error("All sequences must have the same length")

    # Output matrix
    counts = zeros(Int, n_alpha, seq_length)

    # Count
    for seq in valid_seqs
        for (pos, elem) in enumerate(seq)
            counts[alpha_index[elem], pos] += 1
        end
    end

    return counts, removed_idx
end


function thermophilic_label(x::AbstractVector{<:Real}; threshold::Real=50.0)
    out = Vector{Int}(undef, length(x))
    @inbounds for i in eachindex(x)
        out[i] = x[i] ≥ threshold ? 1 : 0
    end
    return out
end

"""
    ConfusionMatrix

Structure to store confusion matrix results for binary classification.

# Fields
- `true_positive::Int`: Number of correctly predicted positive samples
- `true_negative::Int`: Number of correctly predicted negative samples
- `false_positive::Int`: Number of incorrectly predicted positive samples (Type I error)
- `false_negative::Int`: Number of incorrectly predicted negative samples (Type II error)
- `total::Int`: Total number of samples
"""
struct ConfusionMatrix
    true_positive::Int
    true_negative::Int
    false_positive::Int
    false_negative::Int
    total::Int
end

function confusion_matrix(model::Union{LogisticModel, BilinearModel}, 
                         X::Matrix{Float64}, 
                         y::Vector{Int}; 
                         threshold=0.5)
    # Input validation
    n_features, n_samples = size(X)
    
    if length(y) != n_samples
        error("Dimension mismatch: X has $n_samples samples but y has $(length(y)) labels")
    end
    
    if !all(label -> label in [0, 1], y)
        error("Labels must be binary (0 or 1). Found: $(unique(y))")
    end
    
    # Get predictions
    y_pred = predict_class(model, X, threshold=threshold)
    
    # Compute confusion matrix components
    tp = sum((y .== 1) .& (y_pred .== 1))
    tn = sum((y .== 0) .& (y_pred .== 0))
    fp = sum((y .== 0) .& (y_pred .== 1))
    fn = sum((y .== 1) .& (y_pred .== 0))
    
    return ConfusionMatrix(tp, tn, fp, fn, n_samples)
end

function display_confusion_matrix(cm::ConfusionMatrix)
    println("\n" * "="^50)
    println("CONFUSION MATRIX")
    println("="^50)
    println()
    println("                 Predicted")
    println("                Positive  Negative")
    println("              ┌──────────┬──────────┐")
    @printf("Actual   Pos  │  %6d  │  %6d  │\n", cm.true_positive, cm.false_negative)
    println("              ├──────────┼──────────┤")
    @printf("         Neg  │  %6d  │  %6d  │\n", cm.false_positive, cm.true_negative)
    println("              └──────────┴──────────┘")
    println()
    println("TP (True Positive):  $(cm.true_positive)")
    println("TN (True Negative):  $(cm.true_negative)")
    println("FP (False Positive): $(cm.false_positive)")
    println("FN (False Negative): $(cm.false_negative)")
    println("Total Samples:       $(cm.total)")

end

function rna_onehot(seq::AbstractString; alphabet::AbstractVector{Char}=RNA_ALPHABET)
    L = lastindex(seq)
    K = length(alphabet)
    X = falses(K, L)
    @inbounds for j in 1:L
        c = seq[j]
        idx = findfirst(==(c), alphabet)
        if idx === nothing
            error("Character '$c' not found in alphabet $(String(alphabet))")
        end
        X[idx, j] = true
    end
    return X
end

function rna_onehot(
    seqs::Vector{<:AbstractString};
    alphabet::AbstractVector{Char} = RNA_ALPHABET
)
    isempty(seqs) && return BitArray{2}[], Int[]

    valid_chars = Set(alphabet)

    # indices of sequences with valid characters
    valid_idx = [i for (i, s) in enumerate(seqs) if all(c -> c in valid_chars, s)]
    removed_idx = setdiff(1:length(seqs), valid_idx)

    isempty(valid_idx) && return BitArray{2}[], removed_idx

    valid_seqs = seqs[valid_idx]

    # length check AFTER filtering
    Ls = length.(valid_seqs)
    all(==(Ls[1]), Ls) || error("Sequences must all have the same length.")

    encoded = [rna_onehot(s; alphabet=alphabet) for s in valid_seqs]

    return encoded, removed_idx
end


function stack_onehot(onehots::Vector{<:AbstractMatrix})
    isempty(onehots) && error("Empty one-hot list")

    A, L = size(onehots[1])
    N = length(onehots)

    X = zeros(Float64, A * L, N)

    for (i, M) in enumerate(onehots)
        X[:, i] .= vec(Float64.(M))
    end

    return X
end

function one_hot_encode(seqs, alphabet)
    isempty(seqs) && return zeros(Float64, 0, 0), Int[]

    valid_chars = Set(alphabet)
    idx_map = Dict(a => i for (i, a) in enumerate(alphabet))
    A = length(alphabet)

    # find valid indices
    valid_idx = [i for (i, s) in enumerate(seqs) if all(c -> c in valid_chars, s)]
    removed_idx = setdiff(1:length(seqs), valid_idx)

    isempty(valid_idx) && return zeros(Float64, 0, 0), removed_idx

    valid_seqs = seqs[valid_idx]

    # length check AFTER filtering
    L = length(valid_seqs[1])
    all(length(s) == L for s in valid_seqs) ||
        error("All sequences must have the same length")

    N = length(valid_seqs)
    X = zeros(Float64, A * L, N)

    # encode
    for (n, seq) in enumerate(valid_seqs)
        for (pos, c) in enumerate(seq)
            X[(pos - 1) * A + idx_map[c], n] = 1f0
        end
    end

    return X, removed_idx
end

