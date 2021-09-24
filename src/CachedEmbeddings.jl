module CachedEmbeddings

# "internal" deps
import CachedArrays: CachedArrays, CachedArray
import EmbeddingTables: AbstractEmbeddingTable

# stdlib

# deps

include("synchronization.jl")

struct FeatureCache{C <: AbstractArray}
    # How many valid features are in this cache
    insert::Threads.Atomic{Int}
    count::Threads.Atomic{Int}

    # The original index of the corresponding feature vector.
    # If an entry is empty, the parent will be 0.
    parent::Vector{Int}

    # The actual cached data
    data::C
end

struct CachedEmbedding{S,T,C <: FeatureCache} <: AbstractEmbeddingTable{S,T}
    # Remote store
    base::C
    pointers::Vector{Ptr{Nothing}}
    # Local
    cachelock::Threads.SpinLock
    caches::Vector{C}
end

end
