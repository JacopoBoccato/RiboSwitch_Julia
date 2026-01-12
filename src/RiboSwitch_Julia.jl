module RiboSwitch_Julia

const alphabet = "ACGU-"
const RNA_ALPHABET = ['A','C','G','U','-']

using Optim
using LinearAlgebra
using RestrictedBoltzmannMachines, AdvRBMs
using Statistics, DataFrames, CSV, Random, Printf

include("classifier_f.jl")
include("classifier_sec.jl")
include("utils.jl")
end # module RiboSwitch_Julia
