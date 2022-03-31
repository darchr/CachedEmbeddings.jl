#####
##### Writebacks
#####

"""
    get_writeback_work(table, targetsize, [currerntsize])

Return a vector of cache pages that need to be updated with respect to their base.
"""
function get_writeback_work(
    table::CachedEmbedding,
    targetsize::Integer = table.targetbytes,
    currentsize = cachesummary(table).cachesize,
)
    C = arraytype(table)
    v = Vector{Tuple{typeof(table),CachePage{C}}}()
    cleanedpages = Int[]
    return get_writeback_work!(v, cleanedpages, table, targetsize, currentsize)
end

function get_writeback_work(tables::Ensemble)
    example = tables[begin]
    C = arraytype(example)
    v = Vector{Tuple{typeof(example),CachePage{C}}}()
    cleanedpages = Int[]
    for table in tables
        get_writeback_work!(v, cleanedpages, table)
    end
    return (v, cleanedpages)
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
    work, npages = get_writeback_work(tables)

    # Next, use our poor man's load balancing to divide up the work among threads.
    count = Threads.Atomic{Int}(1)
    divisor = length(work)
    divisor == 0 && return nothing

    len = divisor * splitsize
    Polyester.@batch (per = thread) for i in Base.OneTo(nthreads)
        while true
            k = Threads.atomic_add!(count, 1)
            k > len && break

            # Convert into a big-little index.
            j, i = EmbeddingTables._divrem_index(k, divisor)
            table, page = work[i]

            # N.B. - can't use
            # (; data, backedges) = page
            # because Polyester's macro can't handle it.
            data, backedges = page.data, page.backedges

            pagelen = length(backedges)
            worksize = EmbeddingTables.cdiv(pagelen, splitsize)

            start = (j - 1) * worksize + 1
            stop = min(j * worksize, pagelen)

            dataview = view(data, axes(tables, 1), start:stop)
            backedgeview = view(backedges, start:stop)
            writeback!(table, dataview, backedgeview)
        end
        # sfence to make sure all data written nontemporally by the inner
        # `writeback!` is visible.
        EmbeddingTables.sfence()
    end

    # We're done writing back - now just cleanup.
    Polyester.@batch (per = thread) for i in eachindex(tables)
        unsafe_drop_noclean!(tables[i], npages[i])
        shrink_pagecache!(tables[i])
    end
    return nothing
end

function writeback!(
    table::CachedEmbedding{S,T}, data::AbstractMatrix, backedges::AbstractVector
) where {S,T}
    N = nbits(table)
    base = table.base
    for (datacol, backedge) in enumerate(backedges)
        iszero(backedge) && continue

        # Have a non-zero backedge.
        # Need to non-temporally writeback.
        dst = columnpointer(base, backedge)
        src = columnpointer(data, datacol)
        copyto_nt!(S, dst, src)

        # Zero out the saved backedge for safety.
        @inbounds backedges[datacol] = zero(eltype(backedges))
        table.pointers[backedge] = TaggedPtrPair{N}(columnpointer(base, backedge))
    end
end

const AVX512BYTES = 64
@generated function copyto_nt!(
    ::Type{EmbeddingTables.Static{N}}, dst::Ptr{T}, src::Ptr{T}
) where {N,T}
    # Compute how many load and store instructions we need.
    numops, remainder = divrem(N * sizeof(T), AVX512BYTES)
    if !iszero(remainder)
        error("Can't handle weirdly sized embedding tables yet!")
    end

    # TODO: Handle SIMD.Vec length parameter automatically.
    loads = load_impl(SIMD.Vec{16,T}, numops)
    stores = store_nt_impl(SIMD.Vec{16,T}, numops)
    return quote
        Base.@_inline_meta
        $(loads...)
        $(stores...)
    end
end

