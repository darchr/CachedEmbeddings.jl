struct FeatureCache{C <: AbstractArray}
    # How many valid features are in this cache
    insert::Threads.Atomic{Int}
    maxlen::Int

    # The cached data.
    # N.B. - This may optionally include metadata more metadata than just the pure
    # data array.
    data::C
end

# Default definitions
maxlength(x::AbstractVector) = length(x)
maxlength(x::AbstractMatrix) = size(x, 2)

function FeatureCache(data::C) where {C}
    maxlen = maxlength(data)
    insert = Threads.Atomic{Int}(1)
    return FeatureCache(insert, maxlen, data)
end

"""
    acquire!(cache::FeatureCache) -> Union{Int, Nothing}

Return a valid column index to insert a new item into `cache`.
If `cache` is full, return `nothing`.
"""
function acquire!(cache::FeatureCache)
    col = Threads.atomic_add!(cache.insert, 1)
    if col > cache.maxlen
        Threads.atomic_sub!(cache.insert, 1)
        return nothing
    end
    return col
end

