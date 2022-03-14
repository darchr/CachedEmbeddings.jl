struct CachePage{C<:AbstractArray}
    # How many valid features are in this cache
    insert::Threads.Atomic{Int}
    maxlen::Int

    # The cached data.
    # N.B. - This may optionally include more metadata than just the pure data array.
    backedges::Vector{UInt64}
    data::C
end

# Default definitions
maxlength(x::AbstractVector) = length(x)
maxlength(x::AbstractMatrix) = size(x, 2)
filledcols(x::CachePage) = x.insert[] - 1
Base.sizeof(x::CachePage) = sizeof(x.data)
reset!(x::CachePage) = (x.insert[] = 1)

function CachePage(data::C) where {C}
    maxlen = maxlength(data)
    backedges = zeros(UInt64, maxlen)
    insert = Threads.Atomic{Int}(1)
    return CachePage(insert, maxlen, backedges, data)
end

"""
    acquire!(cache::CachePage) -> Union{Int, Nothing}

Return a valid column index to insert a new item into `cache`.
If `cache` is full, return `nothing`.
"""
function acquire!(cache::CachePage)
    col = Threads.atomic_add!(cache.insert, 1)
    if col > cache.maxlen
        Threads.atomic_sub!(cache.insert, 1)
        return nothing
    end
    return col
end

"""
    unsafe_unwrap(cache::CachePage)

Return the raw data array wrapped by `cache`.
The underlying data should only be accessed on the slice provided by a call to
`acquire!(cache)`.
"""
unsafe_unwrap(cache::CachePage) = cache.data, cache.backedges
