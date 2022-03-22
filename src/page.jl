struct CachePage{C<:AbstractArray}
    # How many valid features are in this cache
    insert::Threads.Atomic{Int}
    maxlen::Int

    # Backedges store the original vector ID
    backedges::Vector{UInt64}
    # The actual data itself.
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
    unsafe_unwrap(cache::CachePage, col::Integer) -> NamedTuple{(:data_pointer, :backedge_pointer)}

Return a `NamedTuple` `nt` with fields `data_pointer` and `backedge_pointer`.
Field `nt.data_pointer` is a pointer to a column of data in `cache`.
Field `nt.backedge_pointer` is a pointer to the backedge slot for the corresponding column.

This function is marked unsafe because the return pointers to not prevent `cache` from
being collected by the GC.
"""
function unsafe_unwrap(cache::CachePage, col::Integer)
    (; data, backedges) = cache
    data_pointer = EmbeddingTables.columnpointer(data, col)
    backedge_pointer = pointer(backedges, col)
    return (; data_pointer, backedge_pointer)
end
