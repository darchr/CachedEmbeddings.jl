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

Base.IndexStyle(::CachedEmbedding) = Base.IndexCartesian()
Base.size(A::CachedEmbedding) = size(A.base)

function Base.getindex(A::CachedEmbedding, (i, j)::Vararg{Int,2})
    return Base.unsafe_load(columnpointer(A, j, Update()), i)
end

function Base.setindex!(A::CachedEmbedding, v, (i, j)::Vararg{Int,2})
    return Base.unsafe_store!(columnpointer(A, j, Update()), i)
end
nbits(::CachedEmbedding{<:Any,<:Any,<:Any,N}) where {N} = N
EmbeddingTables.example(A::CachedEmbedding) = A.base

function table_and_column!(table::CachedEmbedding)
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
    col === nothing || return (unsafe_unwrap(cache)..., col)

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
                return (unsafe_unwrap(cache)..., col)
            end
            unlock(buffer)
        end

        # Try again
        cache = @inbounds buffer[]
        col = acquire!(cache)
        col === nothing || return (unsafe_unwrap(cache)..., col)
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
    newcache = CachePage(similar(base, eltype(base), (featuresize(base), f.blocksize)))
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
        return TaggedPtrPair{N}(columnpointer(base, col))
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
    pointers::Vector{TaggedPtrPair{N}},
    i,
) where {N}
    @boundscheck checkbounds(pointers, i)
    return getfield(@inbounds(pointers[i]), :backedge)
end

function EmbeddingTables.columnview(
    table::CachedEmbedding,
    i::Integer,
    context::EmbeddingTables.IndexingContext,
)
    Base.@_inline_meta
    return StrideArraysCore.PtrArray(
        columnpointer(table, i, context),
        (EmbeddingTables.featuresize(table),),
    )
end

function EmbeddingTables.columnview(
    table::CachedEmbedding,
    _,
    i::Integer,
    context::EmbeddingTables.IndexingContext,
)
    Base.@_inline_meta
    return EmbeddingTables.columnview(table, i, context)
end

# In the updating context, don't try to do any data movement.
function EmbeddingTables.columnpointer(
    table::CachedEmbedding{<:Any,T},
    i::Integer,
    ::EmbeddingTables.Update,
) where {T}
    Base.@_inline_meta
    return follow(T, table.pointers[i])
end

# This level really wants to be inlined because the happy path is quite short.
# Keep `_columnpointer` non-inlined because it requires a non-trivial amount of work.
function EmbeddingTables.columnpointer(
    table::CachedEmbedding{<:Any,T},
    i::Integer,
    ::EmbeddingTables.Forward,
) where {T}
    Base.@_inline_meta
    generation = table.generation
    own, ptr, tag = acquire!(taggedpointer(table.pointers, i), generation)
    # Fast path - just return the pointer
    if !own
        return Ptr{T}(ptr)
    else
        backedge = backedgepointer(table.pointers, i)
        return _columnpointer!(table, Ptr{T}(ptr), tag, backedge, i)
    end
end

const TIMES = [Vector{Int}() for _ in Base.OneTo(Threads.nthreads())]

function _columnpointer!(
    table::CachedEmbedding{<:Any,T},
    ptr::Ptr{T},
    tag,
    backedge::Ptr{UInt64},
    i,
) where {T}
    generation = table.generation

    # Slow path - need to copy the data array.
    data, backedges, col = table_and_column!(table)

    # Store the original column in the first region of data
    @inbounds backedges[col] = i

    # Copy over data
    # start = time_ns()
    dst_ptr = columnpointer(data, col)
    for j in axes(table, 1)
        EmbeddingTables.@_ivdep_meta
        EmbeddingTables.@_interleave_meta(8)
        unsafe_store!(dst_ptr, unsafe_load(ptr, j), j)
    end
    # stop = time_ns()
    # push!(TIMES[Threads.threadid()], stop - start)

    # If the original tag is not zero, then we need to clear the backedge for the old
    # cache location.
    iszero(tag) || unsafe_store!(backedge, zero(UInt64))

    # Update original pointer and return
    backedge_ptr = pointer(backedges, col)
    update_with_tag!(pointer(table.pointers, i), dst_ptr, backedge_ptr, generation)
    return Ptr{T}(dst_ptr)
end

#####
##### Static Preparation
#####

function prepopulate!(table::CachedEmbedding{<:Any,T}, tag) where {T}
    targetbytes = table.targetbytes
    nvectors = min(
        size(table, 2),
        div(targetbytes, sizeof(eltype(table)) * featuresize(table))
    )
    pointers = table.pointers
    for v in Base.OneTo(nvectors)
        own, ptr, _tag = acquire!(taggedpointer(pointers, v), tag)
        if own
            backedge = backedgepointer(pointers, v)
            _columnpointer!(table, Ptr{T}(ptr), _tag, backedge, v)
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
    table::CachedEmbedding{<:Any,<:Any,<:Any,N},
    targetsize::Integer,
    currentsize = cachesummary(table).cachesize,
) where {N}
    cleanup!(table.cache) do page
        currentsize <= targetsize && return (false, nothing)
        # Need to update backedge pointers.
        backedges = page.backedges
        base = table.base
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
        return (true, nothing)
    end
    return nothing
end

function unsafe_drop_noclean!(table::CachedEmbedding, npages::Integer)
    pagescleaned = 0
    cleanup!(table.cache) do page
        pagescleaned >= npages && return (false, nothing)

        reset!(page)
        push!(table.pagecache, page)
        pagescleaned += 1
        return (true, nothing)
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
_maybe_free(x::CachePage{<:CachedArrays.CachedArray}) = _maybe_free(unsafe_unwrap(x))

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
##### Writebacks
#####

"""
    get_writeback_work(table, targetsize, [currerntsize])

Return a vector of cache pages that need to be updated with respect to their base.
"""
function get_writeback_work(
    table::CachedEmbedding{<:Any,<:Any,C},
    targetsize::Integer = table.targetbytes,
    currentsize = cachesummary(table).cachesize,
) where {C}
    v = Vector{Tuple{typeof(table),CachePage{C}}}()
    cleanedpages = Int[]
    return get_writeback_work!(v, cleanedpages, table, targetsize, currentsize)
end

function get_writeback_work!(
    v::AbstractVector,
    cleanedpages::AbstractVector{<:Integer},
    table::CachedEmbedding,
    targetsize::Integer = table.targetbytes,
    currentsize = cachesummary(table).cachesize,
)
    npages = 0
    for page in table.cache
        currentsize <= targetsize && break
        push!(v, (table, page))
        currentsize -= sizeof(page)
        npages += 1
    end
    push!(cleanedpages, npages)
    return v, cleanedpages
end

"""
    writeback!(tables::Ensemble)

Set each table to its assigned cache and reserve size.
Write back data in any evicted page to its original source.
"""
function writeback!(tables::Ensemble, splitsize = 4, nthreads = 12)
    # First, queue up all the work we need to perform.
    work, npages = get_writeback_work(tables[1])
    for i = 2:length(tables)
        get_writeback_work!(work, npages, tables[i])
    end

    # Next, use our poor man's load balancing to divide up the work among threads.
    count = Threads.Atomic{Int}(1)
    divisor = length(work)
    divisor == 0 && return nothing

    len = divisor * splitsize
    Polyester.@batch (per = core) for i in Base.OneTo(nthreads)
        while true
            k = Threads.atomic_add!(count, 1)
            k > len && break

            # Convert into a big-little index.
            j, i = EmbeddingTables._divrem_index(k, divisor)
            table, page = work[i]
            data, backedges = page.data, page.backedges

            pagelen = length(backedges)
            worksize = EmbeddingTables.cdiv(pagelen, splitsize)

            start = (j - 1) * worksize + 1
            stop = min(j * worksize, pagelen)

            dataview = view(data, axes(tables, 1), start:stop)
            backedgeview = view(backedges, start:stop)
            writeback!(table, dataview, backedgeview)
        end
    end

    # We're done writing back - now just cleanup.
    Polyester.@batch (per = core) for i in eachindex(tables)
        unsafe_drop_noclean!(tables[i], npages[i])
        shrink_pagecache!(tables[i])
    end
    return nothing
end

function writeback!(
    table::CachedEmbedding{S,T,<:Any,N},
    data::AbstractMatrix,
    backedges::AbstractVector,
) where {S,T,N}
    base = table.base
    for (datacol, backedge) in enumerate(backedges)
        iszero(backedge) && continue

        # Have a non-zero backedge.
        # Need to non-temporally writeback.
        dst = columnpointer(base, backedge)
        src = columnpointer(data, datacol)
        copyto!(S, dst, src)

        # Zero out the saved backedge for safety.
        @inbounds backedges[datacol] = zero(eltype(backedges))
        table.pointers[backedge] = TaggedPtrPair{N}(columnpointer(base, backedge))
    end
end

const AVX512BYTES = 64

@generated function Base.copyto!(
    ::Type{EmbeddingTables.Static{N}},
    dst::Ptr{T},
    src::Ptr{T},
) where {N,T}
    # Compute how many load and store instructions we need.
    numops, remainder = divrem(N * sizeof(T), AVX512BYTES)
    if !iszero(remainder)
        error("Can't handle weirdly sized embedding tables yet!")
    end

    # TODO: Handle SIMD.Vec length parameter automatically.
    loads = load_impl(SIMD.Vec{16,T}, numops)
    stores = store_impl(SIMD.Vec{16,T}, numops)
    return quote
        Base.@_inline_meta
        $(loads...)
        $(stores...)
    end
end

