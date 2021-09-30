#####
##### CachedEmbedding
#####

# TODO: What if we can't cache the whole ensemble lookup?

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
    pointers::Vector{TaggedPtr{N}}
    # Remote store
    base::C
    cache::CircularBuffer{CachePage{C}}
    pagecache::Vector{CachePage{C}}
    generation::Int
    init::F

    # Target characteristics for the `cache` and `pagecache`.
    # Keep these as part of the `Embedding` itself to allow customization on a per-table
    # level.
    targetbytes::Int
    targetreserve::Int
end

Base.size(A::CachedEmbedding) = size(A.base)
Base.getindex(A::CachedEmbedding, i::Int) = A.base[i]
Base.setindex!(A::CachedEmbedding, v, i::Int) = setindex!(A.base, v, i)
nbits(::CachedEmbedding{<:Any,<:Any,<:Any,N}) where {N} = N
EmbeddingTables.example(A::CachedEmbedding) = A.base

function table_and_column!(
    table::CachedEmbedding,
    # buffer::CircularBuffer{C},
    # pagecache::AbstractVector{C},
    # init::F,
)
    buffer, pagecache, init = table.cache, table.pagecache, table.init
    # Need to handle the cases where we're starting with an empty cache buffer.
    # Always check the pagecache first.
    # Only fall back to explicitly requesting more memory if the pagecache is empty.
    while isempty(buffer)
        if trylock(buffer)
            page = isempty(pagecache) ? init() : pop!(pagecache)
            push!(buffer, page)
            unlock(buffer)
        end
        GC.safepoint()
    end

    cache = @inbounds buffer[]
    col = acquire!(cache)

    # Happy Path
    col === nothing || return unsafe_unwrap(cache), col

    # Sad Path
    while true
        if trylock(buffer)
            # Need to try again after acquiring the lock because a new cache may have
            # been added while we were in the safe point or trying to acquire the lock
            # in the first place.
            cache = @inbounds buffer[]
            col = acquire!(cache)
            if col === nothing
                page = isempty(pagecache) ? init() : pop!(pagecache)
                push!(buffer, page)
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

function cachesummary(table::CachedEmbedding)
    # Number and total size of active pages
    cachepages = 0
    cachesize = 0
    for page in table.cache
        cachepages += 1
        cachesize += sizeof(page)
    end
    return (; cachepages, cachesize)
end

function reservesummary(table::CachedEmbedding)
    # Number of reserve pages and size
    reservepages = 0
    reservesize = 0
    for page in table.pagecache
        reservepages += 1
        reservesize += sizeof(page)
    end
    return (; reservepages, reservesize)
end

"""
    summary(table::CachedEmbedding) -> NamedTuple

Return a summary of the cache state for `table`.
The properties of the returned `NamedTuple` are:
* `cachepages` - The number of active pages in `table`'s cache.
* `cachesize` - The total data memory footprint of the cache.
* `reservepages` - The number of pages reserved to be inserted into the cache.
* `reservesize` - The total memory footprint of the page cache.
"""
summary(table::CachedEmbedding) = merge(cachesummary(table), reservesummary(table))

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
    newcache = CachePage(
        similar(base, eltype(base), (featuresize(base) + _pad(eltype(base)), f.blocksize)),
    )
    return newcache
end

function CachedEmbedding{S}(
    base::C,
    ::Val{N};
    targetbytes::Integer = typemax(UInt32),
    targetreserve::Integer = typemax(UInt32),
    max_buffer_length = 1000,
    init::F = DefaultInit(base, ITEMS_PER_CACHE_BLOCK),
) where {S,N,T,C<:AbstractMatrix{T},F}
    # Initialize the original pointers
    pointers = map(axes(base, 2)) do col
        return TaggedPtr{N}(columnpointer(base, col))
    end
    cache = CircularBuffer{CachePage{C}}(max_buffer_length)
    pagecache = Vector{CachePage{C}}()
    generation = 1
    return CachedEmbedding{S,T,C,N,F}(
        pointers,
        base,
        cache,
        pagecache,
        generation,
        init,
        targetbytes,
        targetreserve,
    )
end

# TODO: Get rid of this.
_pad(::Type{Float32}) = 2
settarget!(table::CachedEmbedding, bytes) = table.targetbytes = bytes
settarget!(bytes::Integer) = Base.Fix2(settarget!, bytes)

setreserve!(table::CachedEmbedding, bytes) = table.targetreserve = bytes
setreserve!(bytes::Integer) = Base.Fix2(setreserve!, bytes)

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

const TIMES = [Vector{Int}() for _ in Base.OneTo(Threads.nthreads())]

function _columnpointer(
    table::CachedEmbedding{EmbeddingTables.Static{N},T},
    ptr::Ptr{T},
    tag,
    i,
) where {N,T}
    generation = table.generation

    # Slow path - need to copy the data array.
    #data, col = table_and_column!(table.cache, table.pagecache, table.init)
    # start = time_ns()
    data, col = table_and_column!(table)

    # Store the original column in the first region of data
    dst_ptr = columnpointer(data, col)
    setbackedge!(dst_ptr, unsigned(i), false)
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
    # stop = time_ns()
    # push!(TIMES[Threads.threadid()], stop - start)
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

function unsafe_cleanup!(table::CachedEmbedding)
    unsafe_drop!(table, table.targetbytes)
    shrink_pagecache!(table, table.targetreserve)
    return nothing
end

"""
    unsafe_drop!(table::CachedEmbedding, maxblocks)

Remove at most `maxblocks` from `table`'s cache.
This is called unsafe because it does not write back any cached data to the backing
store.
"""
function unsafe_drop!(
    table::CachedEmbedding{<:Any,<:Any,<:Any,N},
    targetsize::Integer,
    currentsize = cachesummary(table).cachesize,
) where {N}
    cleanup!(table.cache) do page
        currentsize <= targetsize && return (false, nothing)
        # Need to update backedge pointers.
        data = page.data
        base = table.base
        for col in axes(data, 2)
            ptr = columnpointer(data, col)
            backedge = getbackedge(ptr, false)
            # If the backedge has been zeroed, then there's nothing to do.
            # Otherwise, we need to update the top level pointer back to its original
            # location.
            iszero(backedge) && continue
            table.pointers[backedge] = TaggedPtr{N}(columnpointer(base, backedge))
            setbackedge!(ptr, zero(UInt64), false)
        end

        reset!(page)
        push!(table.pagecache, page)
        currentsize -= sizeof(page)
        return (true, nothing)
    end
    return nothing
end

function shrink_pagecache!(
    table::CachedEmbedding,
    targetsize,
    currentsize = reservesummary(table).reservesize,
)
    pagesfreed = 0
    pagecache = table.pagecache
    while currentsize > targetsize && !isempty(pagecache)
        page = pop!(pagecache)
        currentsize -= sizeof(page)
        _maybe_free(page)
        pagesfreed += 1
    end
    return nothing
end

_maybe_free(_) = nothing
_maybe_free(x::CachedArrays.CachedArray) = CachedArrays.unsafe_free(x)
_maybe_free(x::CachePage{<:CachedArrays.CachedArray}) = _maybe_free(unsafe_unwrap(x))

#####
##### Convenience Functions
#####

const Ensemble = AbstractVector{<:CachedEmbedding}

settarget!(tables::Ensemble, bytes) = foreach(settarget!(bytes), tables)
setreserve!(tables::Ensemble, bytes) = foreach(setreserve!(bytes), tables)
next!(tables::Ensemble) = foreach(next!, tables)
unsafe_cleanup!(tables::Ensemble) = foreach(unsafe_cleanup!, tables)

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
