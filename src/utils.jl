"""
Useful function to preprocess sequences
"""

function count_alphabet_matrix(alphabet, sequences)
    alpha_index = Dict(a => i for (i, a) in enumerate(alphabet))
    valid_chars = Set(alphabet)
    n_alpha = length(alphabet)

    isempty(sequences) && error("No sequences provided")

    # Indices of valid sequences
    valid_idx = [i for (i, s) in enumerate(sequences) if all(c -> c in valid_chars, s)]
    removed_idx = setdiff(1:length(sequences), valid_idx)

    isempty(valid_idx) && error("All sequences contain invalid characters")

    valid_seqs = sequences[valid_idx]
    n_sequences = length(valid_seqs)

    # Output matrix: (alphabet_size × n_sequences)
    counts = zeros(Int, n_alpha, n_sequences)

    @inbounds for (j, seq) in enumerate(valid_seqs)   # sequence index
        for c in seq
            counts[alpha_index[c], j] += 1
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
Function to encode sequences using indices (faster to train bilinear models)
"""
function encode_indices(
    sequences::Vector{String},
    aa_to_index::Dict{Char,Int},
    alphabet_size::Int
)
    isempty(sequences) && return Matrix{Int32}(undef, 0, 0), Int[]

    # Keep only sequences with valid characters
    valid_idx = [
        i for (i, s) in enumerate(sequences)
        if all(c -> haskey(aa_to_index, c), s)
    ]
    removed_idx = setdiff(1:length(sequences), valid_idx)

    isempty(valid_idx) && return Matrix{Int32}(undef, 0, 0), removed_idx

    valid_seqs = sequences[valid_idx]

    # Length check AFTER filtering
    L_pdz = length(valid_seqs[1])
    @assert all(length(s) == L_pdz for s in valid_seqs) "All sequences must have equal length"

    N = length(valid_seqs)

    # Output: (positions × sequences)
    pdz_idx = Matrix{Int32}(undef, L_pdz, N)

    @inbounds for (j, s) in enumerate(valid_seqs)
        for pos in 1:L_pdz
            aa = s[pos]
            aa_idx = aa_to_index[aa]  # safe after filtering
            pdz_idx[pos, j] = Int32((pos - 1) * alphabet_size + aa_idx)
        end
    end

    return pdz_idx, removed_idx
end

function encode_indices(
    sequences::Vector{String},
    alphabet::Vector{Char},
    alphabet_size::Int
)
    aa_to_index = Dict(a => i for (i, a) in enumerate(alphabet))
    return encode_indices(sequences, aa_to_index, alphabet_size)
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

function confusion_matrix(model::LogisticModel, 
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

function confusion_matrix(model::LowRankBilinearModel, 
                         x_idx::Matrix{Int32}, 
                         z_idx::Matrix{Int32},
                         y::Vector{Int}; 
                         threshold=0.5)
    # Input validation
    n_samples = size(x_idx, 2)
    
    if size(z_idx, 2) != n_samples
        error("Dimension mismatch: x_idx has $n_samples samples but z_idx has $(size(z_idx, 2)) samples")
    end
    
    if length(y) != n_samples
        error("Dimension mismatch: indices have $n_samples samples but y has $(length(y)) labels")
    end
    
    if !all(label -> label in [0, 1], y)
        error("Labels must be binary (0 or 1). Found: $(unique(y))")
    end
    
    # Get predictions
    y_pred = predict_class(model, x_idx, z_idx, threshold=threshold)
    
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

function plot_confusion_matrix(cm::ConfusionMatrix, filename::String="confusion_matrix.png")
    # Definiamo le etichette
    labels = ["Negative (0)", "Positive (1)"]
    
    # Prepariamo il canvas vuoto con impostazioni pulite
    p = plot(
        xlim=(0.5, 2.5), ylim=(0.5, 2.5),
        xticks=(1:2, labels), yticks=(1:2, labels),
        xlabel="Predicted", ylabel="Actual",
        title="Confusion Matrix (n=$(cm.total))",
        aspect_ratio=:equal,
        legend=false,
        grid=false,
        framestyle=:box,
        background_color=:white,
        margin=10mm
    )

    # Definiamo colori chiari e comprensibili (Pastello)
    # Verde chiaro per i corretti (TP, TN), Rosso/Arancio chiaro per gli errori (FP, FN)
    c_correct = RGB(0.9, 1.0, 0.9) # Verde chiarissimo
    c_error   = RGB(1.0, 0.9, 0.9) # Rosso chiarissimo

    # Disegniamo i 4 quadrati colorati come sfondo
    # Coordinate: TN(1,1), FP(2,1), FN(1,2), TP(2,2)
    # [x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]
    plot!(p, [0.5, 1.5, 1.5, 0.5], [0.5, 0.5, 1.5, 1.5], seriestype=:shape, fillcolor=c_correct, linecolor=:grey) # TN
    plot!(p, [1.5, 2.5, 2.5, 1.5], [0.5, 0.5, 1.5, 1.5], seriestype=:shape, fillcolor=c_error,   linecolor=:grey) # FP
    plot!(p, [0.5, 1.5, 1.5, 0.5], [1.5, 1.5, 2.5, 2.5], seriestype=:shape, fillcolor=c_error,   linecolor=:grey) # FN
    plot!(p, [1.5, 2.5, 2.5, 1.5], [1.5, 1.5, 2.5, 2.5], seriestype=:shape, fillcolor=c_correct, linecolor=:grey) # TP

    # Aggiungiamo il testo perfettamente centrato
    # Usiamo halign e valign per la precisione
    style = (12, :black, :center)
    annotate!(p, 1, 1, text("TN\n$(cm.true_negative)", style...))
    annotate!(p, 2, 1, text("FP\n$(cm.false_positive)", style...))
    annotate!(p, 1, 2, text("FN\n$(cm.false_negative)", style...))
    annotate!(p, 2, 2, text("TP\n$(cm.true_positive)", style...))

    # Salvataggio
    mkpath(dirname(filename))
    savefig(p, filename)
    println("Plot pulito salvato in: $filename")
    return p
end