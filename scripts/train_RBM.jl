#!/usr/bin/env julia
using Distributed
using CUDA

# ----------------------------------------------------------
# Launch workers: one per GPU
# ----------------------------------------------------------
if nprocs() == 1
    ngpus = length(CUDA.devices())
    @info "Found $ngpus GPUs"
    ngpus == 0 && error("No CUDA GPUs detected!")
    addprocs(ngpus)
end

@info "Workers launched: $(nworkers())"
@everywhere using CUDA

# ----------------------------------------------------------
# Imports shared across workers
# ----------------------------------------------------------
@everywhere begin
    using Pkg
    using LinearAlgebra, Statistics, Random
    using RestrictedBoltzmannMachines: RBM, PottsGumbel, Binary, ReLU, nsReLU,
        log_pseudolikelihood, initialize!, sample_v_from_v, pcd!,
        sample_h_from_v, standardize, gpu, cpu
    # Activate the package environment (repo root)
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    # Now the package is on LOAD_PATH
    using RiboSwitch_Julia
    using RiboSwitch_Julia: RNA_ALPHABET, one_hot_encode, alphabet, predict,
                       count_alphabet_matrix, thermophilic_label, train_lowrank_bilinear_classifier, prevalence_threshold,
                       train_linear_classifier, train_linear_regressor, rna_onehot, confusion_matrix, display_confusion_matrix, plot_confusion_matrix, encode_indices
    using Optimisers: Adam
    using Serialization
    using AdvRBMs
    using JLD2
    using DataFrames
    using XLSX
    using Optim
end


df = CSV.read("/home/jacopo/RiboSwitch_Julia/artifacts/Ribo_aligned.tsv", DataFrame; delim='\t')
X, removed_idx = one_hot_encode(df.aligned_sequence, alphabet)
y = df.TOME_Predicted_OGT_Celsius
y = y[setdiff(1:end, removed_idx)]
labels = thermophilic_label(y; threshold=45)

q = AdvRBMs.calc_q(X, labels)
Q_full = AdvRBMs.calc_Q(X, labels)
Q = lowrank_decomp_tensor(Q_full, 5)
# ----------------------------------------------------------
# Define model-training task (runs on each worker)
# ----------------------------------------------------------
@everywhere function train_rbm_job(N::Int, fname::String, hidden_range, gpu_id::Int)

    CUDA.device!(gpu_id)
    q_gpu = cu(q)
    Q_gpu = cu(Q)

    # Build RBM on CPU first
    rbm = RBM(PottsGumbel((5, 108)), Binary((150,)), zeros(Float32, 5, 108, 150))

    # Move RBM parameters to GPU
    rbm_gpu = gpu(rbm)

    # Move data last (biggest)
    data_gpu = cu(X)

    initialize!(rbm_gpu, data_gpu)

    training_ok = true
    try
        AdvRBMs.advpcd!(rbm_gpu, data_gpu;
            batchsize = 256,
            iters     = 300000,
            steps     = 100,
            optim     = Adam(1f-5, (0.9, 0.999)),
            l2l1_weights = 0.01,
            qs = [q_gpu],
            Qs = [Q_gpu],
            ℋs = [CartesianIndices((hidden_range,))]
        )
    catch e
        training_ok = false
        @warn "Training failed on pid=$(myid()) for $fname: $e"
    end

    if !training_ok
        @warn "Skipping save for failed model $fname"
        return nothing
    end

    rbm_cpu = cpu(rbm_gpu)
    JLD2.@save fname rbm_cpu
    @info "Saved trained RBM: $fname"
    return fname
end
# ----------------------------------------------------------
# Job definitions
# ----------------------------------------------------------
jobs = [
    (1, "RNA_binary_fs_20.jld2", 21:150),
    (1, "RNA_binary_fs_15.jld2",16:150),
    (1, "RNA_binary_fs_10.jld2",11:150),
    (1, "RNA_binary_fs_5.jld2",6:150)
]

# ----------------------------------------------------------
# Multi-GPU Dispatch
# ----------------------------------------------------------
wks = workers()
ngpus = length(CUDA.devices())

@assert length(wks) == ngpus "Expected one worker per GPU, got $(length(wks)) workers for $ngpus GPUs"

# Split jobs into chunks, one chunk per worker
job_chunks = [jobs[i:length(wks):end] for i in 1:length(wks)]

@everywhere function run_job_chunk(jobs_chunk, gpu_id)
    # Bind this worker to its GPU once
    CUDA.device!(gpu_id)
    @info "Worker $(myid()) bound to GPU $gpu_id"

    for (N, fname, hrange) in jobs_chunk
        @info "[Worker $(myid())] starting $fname on GPU=$gpu_id"
        train_rbm_job(N, fname, hrange, gpu_id)
        @info "[Worker $(myid())] finished $fname on GPU=$gpu_id"
        CUDA.reclaim()  # clean up cached memory before next job
    end

    return nothing
end

futs = Future[]
for (chunk_idx, pid) in enumerate(wks)
    gpu_id = chunk_idx - 1  # 0-based indices
    push!(futs, @spawnat pid run_job_chunk(job_chunks[chunk_idx], gpu_id))
end

@info "Waiting for training jobs..."
fetch.(futs)
@info "All RBM jobs finished!"
