#####
##### Strategies
#####

abstract type AbstractEmbeddingStrategy end
struct BlockBased <: AbstractEmbeddingStrategy end

struct GenerationBased <: AbstractEmbeddingStrategy
    current_generations::Vector{Int}
end
GenerationBased() = GenerationBased(Int[])

Base.length(v::GenerationBased) = length(v.current_generations)
function Base.popfirst!(v::GenerationBased)
    return popfirst_and_shift(v.current_generations)
end
function Base.push!(v::GenerationBased, generation::Integer)
    return push!(v.current_generations, generation)
end

#####
##### CachedEmbedding
#####

"""
    CachedEmbedding{S,T,G,C <: AbstractArray,N,F}

## Type Parameters
* `S` - EmbeddingTables lookup type.
* `T` - Embedding table element type.
* `G` - CachingStratetgy
* `C` - The type of the base data array.
* `N` - Number of tag bits.
* `F` - Type of the cache block initialization function.
"""
mutable struct CachedEmbedding{S,T,G<:AbstractEmbeddingStrategy,C<:AbstractMatrix,N,F} <:
               AbstractEmbeddingTable{S,T}
    strategy::G
    pointers::Vector{TaggedPtrPair{N}}
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

# Aliases
const BlockBasedEmbedding{S,T} = CachedEmbedding{S,T,BlockBased}
const GenerationBasedEmbedding{S,T} = CachedEmbedding{S,T,GenerationBased}

# Simple Functions
@inline Base.size(A::CachedEmbedding) = size(A.base)
@inline strategy(A::CachedEmbedding) = A.strategy

nbits(::CachedEmbedding{<:Any,<:Any,<:Any,<:Any,N}) where {N} = N
EmbeddingTables.example(A::CachedEmbedding) = A.base
arraytype(::CachedEmbedding{<:Any,<:Any,<:Any,C}) where {C} = C

# @inline function base_addresses(table::CachedEmbedding{S,T}) where {S,T}
#     (; base) = table
#     base_address = UInt(pointer(base))
#     length = sizeof(base)
#     return base_address:(base_address + length - sizeof(T))
# end
#
# function inbase(ptr::Ptr{T}, table::CachedEmbedding{S,T}) where {S,T}
#     return in(UInt(ptr), base_addresses(table))
# end

"""
    allocate_in_cache!(table::CachedEmbedding) -> NamedTuple{(:data_pointer, :backedge_pointer)}

"""
@inline function allocate_in_cache!(table::CachedEmbedding)
    (; cache) = table
    # Happy path
    page = @inbounds cache[end]
    if page !== nothing
        col = acquire!(page)
        col !== nothing && return unsafe_unwrap(page, col)
    end

    # Slow path
    return allocate_in_cache_add_page!(table)
end

# Slow path - a new page is needed for the buffer.
@noinline function allocate_in_cache_add_page!(table)
    (; cache, pagecache, init) = table
    while true
        if trylock(cache)
            # Need to try again after acquiring the lock because a new page may have
            # been added while we were in the safe point or trying to acquire the lock
            # in the first place.
            page = @inbounds cache[end]
            if page !== nothing
                col = acquire!(page)
                if col !== nothing
                    unlock(cache)
                    return unsafe_unwrap(page, col)
                end
            end
            push!(cache, isempty(pagecache) ? init() : pop!(pagecache))
            unlock(cache)
        end
        GC.safepoint()

        # # Try again outside of lock - maybe another thread managed to populate this slot.
        page = @inbounds cache[end]
        if page !== nothing
            col = acquire!(page)
            col !== nothing && return unsafe_unwrap(page, col)
        end
    end
end

@inline current_generation(table::CachedEmbedding) = table.generation
function _next(table::CachedEmbedding)
    generation = current_generation(table)
    return (generation == (2^nbits(table) - 1)) ? 1 : (generation + 1)
end

"""
    next!(table::CachedEmbedding)

Set `table` to a new generation.
"""
function next!(table::CachedEmbedding)
    next = _next(table)
    table.generation = next
    return next
end

function next!(table::GenerationBasedEmbedding)
    next = _next(table)
    push!(strategy(table), next)
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
    newcache = CachePage(similar(base, eltype(base), (featuresize(base), f.blocksize)))
    return newcache
end

function CachedEmbedding{S}(
    base::C,
    ::Val{N};
    strategy::G = BlockBased(),
    targetbytes::Integer = typemax(UInt32),
    targetreserve::Integer = typemax(UInt32),
    max_buffer_length = 1000,
    init::F = DefaultInit(base, ITEMS_PER_CACHE_BLOCK),
) where {S,N,T,C<:AbstractMatrix{T},G,F}
    # Initialize the original pointers
    pointers = map(axes(base, 2)) do col
        return TaggedPtrPair{N}(columnpointer(base, col))
    end
    cache = CircularBuffer{CachePage{C}}(max_buffer_length)
    pagecache = Vector{CachePage{C}}()
    generation = 1
    return CachedEmbedding{S,T,G,C,N,F}(
        strategy,
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

settarget!(table::CachedEmbedding, bytes) = table.targetbytes = bytes
setreserve!(table::CachedEmbedding, bytes) = table.targetreserve = bytes
disable!(table::CachedEmbedding) = (table.generation = 0)
enable!(table::CachedEmbedding) = (table.generation = 1)

function taggedpointer(pointers::Vector{TaggedPtrPair{N}}, i) where {N}
    return Ptr{TaggedPtr{N}}(pointer(pointers, i))
end

function settags!(table::CachedEmbedding{<:Any,<:Any,<:Any,N}, tag) where {N}
    pointers = table.pointers
    for i in eachindex(pointers)
        ptr, backedge = pointers[i]
        update_with_tag!(pointer(pointers, i), ptr[], backedge, tag)
    end
end

Base.@propagate_inbounds function backedgepointer(
    pointers::Vector{TaggedPtrPair{N}}, i
) where {N}
    @boundscheck checkbounds(pointers, i)
    return getfield(@inbounds(pointers[i]), :backedge)
end

# In a generic indexing context, don't try to move any data, simply return the pointer
function EmbeddingTables.columnpointer(
    table::CachedEmbedding{<:Any,T}, i::Integer, ::EmbeddingTables.IndexingContext
) where {T}
    Base.@_inline_meta
    return follow(T, table.pointers[i])
end

# This level really wants to be inlined because the happy path is quite short.
# Keep `_columnpointer` non-inlined because it requires a non-trivial amount of work.
function EmbeddingTables.columnpointer(
    table::CachedEmbedding{<:Any,T}, i::Integer, ::EmbeddingTables.Forward
) where {T}
    Base.@_inline_meta
    generation = table.generation
    own, ptr, tag = acquire!(taggedpointer(table.pointers, i), generation)
    # Fast path - just return the pointer
    if !own
        return Ptr{T}(ptr)
    else
        backedge = @inbounds(backedgepointer(table.pointers, i))
        return _columnpointer!(table, Ptr{T}(ptr), tag, backedge, i)
    end
end

const TIMES = [Vector{Int}() for _ in Base.OneTo(Threads.nthreads())]

function cachevector!(
    table::CachedEmbedding{EmbeddingTables.Static{N},T},
    ptr::Ptr{T},
    current_tag,
    current_backedge::Ptr{UInt64},
    i,
) where {N,T}
    new_tag = current_generation(table)
    # Slow path - need to copy the data array.
    (; data_pointer, backedge_pointer) = allocate_in_cache!(table)

    # Store the original column in the first region of data
    unsafe_store!(backedge_pointer, unsigned(i))

    # Copy over data
    # loadtype = EmbeddingTables.simdtype(EmbeddingTables.Static{N}(), T)
    # temp = EmbeddingTables.load(loadtype, ptr)
    # EmbeddingTables.store(temp, data_pointer)
    for j in axes(table, 1)
        EmbeddingTables.@_ivdep_meta
        EmbeddingTables.@_interleave_meta(8)
        unsafe_store!(data_pointer, unsafe_load(ptr, j), j)
    end

    # If the original tag is not zero, then we need to clear the backedge for the old
    # cache location.
    !iszero(current_tag) && unsafe_store!(current_backedge, zero(UInt64))

    # Update original pointer and return
    update_with_tag!(pointer(table.pointers, i), data_pointer, backedge_pointer, new_tag)
    return Ptr{T}(data_pointer)
end

"""
    _columnpointer!(table::BlockBasedEmbedding, ptr::Ptr{T}, tag, backedge, i)

Move vector `i` into the most recent cache page in `table` if its tag differs from `table`'s
current generation. Doing this will update the tag to the most recent generation and
invalidate the previously cached version of the vector.
"""
@inline function _columnpointer!(
    table::BlockBasedEmbedding{<:Any,T}, ptr::Ptr{T}, current_tag, backedge::Ptr{UInt64}, i
) where {T}
    return cachevector!(table, ptr, current_tag, backedge, i)
end

"""
    _columnpointer!(table::BlockBasedEmbedding, ptr::Ptr{T}, tag, backedge, i)

If embedding vector `i` is already cached, simply update its generation to the current
generation in `table`.
Otherwise, the vector will be cached as well.
"""
function _columnpointer!(
    table::GenerationBasedEmbedding{<:Any,T},
    ptr::Ptr{T},
    current_tag,
    backedge::Ptr{UInt64},
    i,
) where {T}
    generation = current_generation(table)
    # If this entry is already cached, simply update its tag.
    if !iszero(current_tag)
        update_with_tag!(pointer(table.pointers, i), ptr, backedge, generation)
        return ptr
    end
    return cachevector!(table, ptr, current_tag, backedge, i)
end

#####
##### Static Preparation
#####

function prepopulate!(table::CachedEmbedding{<:Any,T}, tag) where {T}
    (; targetbytes, pointers) = table
    nvectors = min(
        size(table, 2), div(targetbytes, sizeof(eltype(table)) * featuresize(table))
    )
    for v in Base.OneTo(nvectors)
        own, ptr, _tag = acquire!(taggedpointer(pointers, v), tag)
        if own
            backedge = backedgepointer(pointers, v)
            cachevector!(table, Ptr{T}(ptr), _tag, backedge, v)
        end
    end
    return nothing
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
    table::CachedEmbedding, targetsize::Integer, currentsize = cachesummary(table).cachesize
)
    N = nbits(table)
    (; base, cache) = table
    cleanup!(cache) do page
        currentsize <= targetsize && return false
        # Need to update backedge pointers.
        (; backedges) = page
        for col in eachindex(backedges)
            backedge = @inbounds backedges[col]
            # If the backedge has been zeroed, then there's nothing to do.
            # Otherwise, we need to update the top level pointer back to its original
            # location.
            iszero(backedge) && continue
            table.pointers[backedge] = TaggedPtrPair{N}(columnpointer(base, backedge))
            @inbounds backedges[col] = zero(UInt64)
        end

        reset!(page)
        push!(table.pagecache, page)
        currentsize -= sizeof(page)
        return true
    end
    return nothing
end

function unsafe_drop_noclean!(table::CachedEmbedding, npages::Integer)
    pagescleaned = 0
    cleanup!(table.cache) do page
        pagescleaned >= npages && return false
        reset!(page)
        push!(table.pagecache, page)
        pagescleaned += 1
        return true
    end
    return nothing
end

function shrink_pagecache!(
    table::CachedEmbedding,
    targetsize = table.targetreserve,
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
_maybe_free(x::CachePage{<:CachedArrays.CachedArray}) = _maybe_free(x.data)

#####
##### Convenience Functions
#####

const Ensemble = AbstractVector{<:CachedEmbedding}

macro forward_ensemble(f, args...)
    f = esc(f)
    args = map(esc, args)
    return :($f(tables::Ensemble, $(args...)) = foreach(x -> $f(x, $(args...)), tables))
end

@forward_ensemble settarget! bytes
@forward_ensemble setreserve! bytes
@forward_ensemble next!
@forward_ensemble unsafe_cleanup!
@forward_ensemble disable!
@forward_ensemble enable!

#####
##### Summary
#####

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

