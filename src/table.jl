function table_and_column!(buffer::CircularBuffer{<:FeatureCache}, init::F) where {F}
    # Need to handle the cases where we're starting with an empty cache buffer.
    while isempty(buffer)
        if trylock(buffer)
            isempty(buffer) && push!(buffer, init())
            unlock(buffer)
        end
        GC.safepoint()
    end

    cache = buffer[]
    col = acquire!(cache)

    # Happy Path
    col === nothing || return unsafe_unwrap(cache), col

    # Sad Path
    while true
        if trylock(buffer)
            # Need to try again after acquiring the lock because a new cache may have
            # been added while we were in the safe point or trying to acquire the lock
            # in the first place.
            cache = buffer[]
            col = acquire!(cache)
            if col === nothing
                push!(buffer, init())
            else
                unlock(buffer)
                return unsafe_unwrap(cache), col
            end
            unlock(buffer)
        end

        # Try again
        cache = @inbounds buffer[]
        col = acquire!(cache)
        col === nothing || return unsafe_unwrap(cache), col
        GC.safepoint()
    end
end

#####
##### CachedEmbedding
#####

"""
    CachedEmbedding{S,T,C <: AbstractArray,N,U}

## Type Parameters
* `S` - EmbeddingTables lookup type.
* `T` - Embedding table element type.
* `C` - The type of the base data array.
* `N` - Number of tag bits.
* `F` - Type of the cache block initialization function.
"""
mutable struct CachedEmbedding{S,T,C<:AbstractMatrix,N,F} <: AbstractEmbeddingTable{S,T}
    # Remote store
    pointers::Vector{TaggedPtr{N}}
    base::C
    buffer::CircularBuffer{FeatureCache{C}}
    generation::Int
    init::F
end

Base.size(A::CachedEmbedding) = size(A.base)
Base.getindex(A::CachedEmbedding, i::Int) = A.base[i]
Base.setindex!(A::CachedEmbedding, v, i::Int) = setindex!(A.base, v, i)
nbits(::CachedEmbedding{<:Any,<:Any,<:Any,N}) where {N} = N
EmbeddingTables.example(A::CachedEmbedding) = A.base

"""
    next!(table::CachedEmbedding)

Set `table` to a new generation.
"""
function next!(table::CachedEmbedding)
    generation = table.generation
    next = (generation == (2^nbits(table) - 1)) ? 1 : (generation + 1)
    table.generation = next
    return next
end

# Constructor
const ITEMS_PER_CACHE_BLOCK = 512
struct DefaultInit{C}
    base::C
    blocksize::Int
end

function (f::DefaultInit)()
    base = f.base
    newcache = FeatureCache(
        similar(
            base,
            eltype(base),
            (featuresize(base) + _pad(eltype(base)), f.blocksize),
        ),
    )
    return newcache
end

function CachedEmbedding{S}(
    base::C,
    ::Val{N};
    max_buffer_length = 1000,
    init::F = DefaultInit(base, ITEMS_PER_CACHE_BLOCK),
) where {S,N,T,C<:AbstractMatrix{T},F}
    # Initialize the original pointers
    pointers = map(axes(base, 2)) do col
        return TaggedPtr{N}(columnpointer(base, col))
    end
    buffer = CircularBuffer{FeatureCache{C}}(max_buffer_length)
    generation = 1
    return CachedEmbedding{S,T,C,N,F}(pointers, base, buffer, generation, init)
end

# Initialize the cache
initialize!(table::CachedEmbedding, v) = push!(table.buffer, FeatureCache(v))

_pad(::Type{Float32}) = 2
# This level really wants to be inlined because the happy path is quite short.
# Keep `_columnpointer` non-inlined because it requires a non-trivial amount of work.
function EmbeddingTables.columnpointer(
    table::CachedEmbedding{<:Any,T},
    i::Integer,
) where {T}
    Base.@_inline_meta
    generation = table.generation
    own, ptr, tag = acquire!(pointer(table.pointers, i), generation)
    # Fast path - just return the pointer
    return own ? _columnpointer(table, Ptr{T}(ptr), tag, i) : Ptr{T}(ptr)
end

function _columnpointer(table::CachedEmbedding, ptr::Ptr{T}, tag, i) where {T}
    generation = table.generation

    # Slow path - need to copy the data array.
    data, col = table_and_column!(table.buffer, table.init)
    # Store the original column in the first region of data
    dst_ptr = columnpointer(data, col)
    setbackedge!(dst_ptr, i, false)
    dst_ptr += sizeof(UInt64)

    # Copy over data
    for j in axes(table, 1)
        EmbeddingTables.@_ivdep_meta
        EmbeddingTables.@_interleave_meta(8)
        unsafe_store!(dst_ptr, unsafe_load(ptr, j), j)
    end

    # If the original tag is not zero, then we need to clear the backedge for the old
    # cache location.
    iszero(tag) || setbackedge!(ptr, zero(UInt64))

    # Update original pointer and return
    update_with_tag!(pointer(table.pointers, i), dst_ptr, generation)
    return Ptr{T}(dst_ptr)
end

@inline function getbackedge(ptr::Ptr, isdataptr = true)
    subval = isdataptr ? sizeof(UInt64) : 0
    return unsafe_load(Ptr{UInt64}(ptr) - subval)
end

@inline function setbackedge!(ptr::Ptr, v, isdataptr = true)
    subval = isdataptr ? sizeof(UInt64) : 0
    return unsafe_store!(Ptr{UInt64}(ptr) - subval, v)
end

#####
##### Cleanup Methods
#####

"""
    unsafe_drop!(table::CachedEmbedding, maxblocks)

Remove at most `maxblocks` from `table`'s cache.
This is called unsafe because it does not write back any cached data to the backing
store.
"""
function unsafe_drop!(
    table::CachedEmbedding{<:Any,<:Any,<:Any,N},
    maxblocks::Integer,
) where {N}
    dropped = 1
    cleanup!(table.buffer) do cache
        dropped > maxblocks && return (false, nothing)
        dropped += 1
        # Need to update backedge pointers.
        data = cache.data
        base = table.base
        for col in axes(data, 2)
            ptr = columnpointer(data, col)
            backedge = getbackedge(ptr, false)
            # If the backedge has been zeroed, then there's nothing to do.
            # Otherwise, we need to update the top level pointer back to its original
            # location.
            iszero(backedge) && continue
            table.pointers[backedge] = TaggedPtr{N}(columnpointer(base, backedge))
        end
        _maybe_free(data)

        return (true, nothing)
    end
    return nothing
end

_maybe_free(_) = nothing
_maybe_free(x::CachedArrays.CachedArray) = CachedArrays.unsafe_free(x)

# Steps required for `columnpointer`
# 1. Get ahold of the pointer to the column.
# 2. If that pointer is not remote - simply return the pointer.
# 3. If that pointer IS remote.
#   3-1. Try Lock pointer
#       3-1-1. Success: Acquire destination in a cache array.
#           3-1-1-1. Success:
#               Copy over data.
#               Set column index in cache.
#               Update and unlock pointer.
#           3-1-1-2. Failure: Try to acquire lock to allocate a new table.
#               3-1-1-2-1. Success:
#                   Allocate new cache table.
#                   Append cache table to caches.
#                   Goto (3-1-1-1).
#               3-1-1-2-2. Failure:
#                   Implication - another thread is allocating the new table.
#                   Short sleep.
#                   Goto (3-1-1) - hopefully after sleep, we'll be successful this time.
#      3-1-2. Failure: Another thread is moving the data. Simply return the masked pointer.
#
# Questions: How do we know if a pointer is cached?
#   Need to look at the "N" most recent cached where "N" is the number of cached
#   allocated on this lookup.
#
#   Maybe we can tag the lower bits of the locking pointer with a round number and compare
#   it with the current round. Then we could check if a vector has been cached with a single
#   64-bit load.
#
#   How many bits do we have available?
#   Lets assume BF16 elements (2 bytes per element.)
#   With a featuresize of 1 - we bet 1 free bit, which we need for locking.
#   With a feature size of 8, each vector occupies 16 bytes (4 bits of address space.).
#   Thus, we get `4 - 1 = 3` bits for round numbers.
#   This should do a pretty good job.
