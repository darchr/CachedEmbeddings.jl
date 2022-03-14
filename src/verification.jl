#####
##### Invariance Checks
#####

"""
    check(table::CachedEmbedding; clean = true) -> Dict

Check the invariants for `table` to see if something went wrong.
Keyword argument `clean` should be `true` if no updates have happened to the
embedding table. This will verify the contents of any cached features with the
ground truth features in the base table.
"""
function check(table::CachedEmbedding; clean = true)
    seen_generations = Set{Int}()

    # First, check that all pointers that point to the embedding table have generation
    # zero - and all pointers pointing outside the embedding table have non-zero
    # generations.
    base = table.base
    base_start = UInt(pointer(base))
    base_end = UInt(pointer(base, lastindex(base)))
    base_range = base_start:base_end

    pointers = table.pointers
    pointer_tag_check = true
    for ptrpair in pointers
        ptr = ptrpair.tagged
        tag = gettag(ptr)
        push!(seen_generations, tag)
        if iszero(tag)
            pointer_tag_check &= in(UInt(ptr[]), base_range)
        else
            pointer_tag_check &= !in(UInt(ptr[]), base_range)
        end
    end
    results = @dict(seen_generations, pointer_tag_check)

    # Valid Backedge Check
    # Each cached feature should have a backedge to its original column.
    # There should only be one backedge per cached column.
    cached_columns = Set{Int}()
    valid_backedge_check = true
    clean_check = true
    #for (col, (ptr, backedge_ptr)) in enumerate(pointers)
    for (col, ptrpair) in enumerate(pointers)
        ptr = ptrpair.tagged
        backedge_ptr = ptrpair.backedge

        tag = gettag(ptr)
        iszero(tag) && continue
        # We have a cached feature.
        # Make sure its backedge is set correctly.
        push!(cached_columns, col)
        backedge = unsafe_load(backedge_ptr)
        valid_backedge_check &= (backedge == col)
        if backedge != col
            @show (backedge, col)
        end
        if clean
            original = EmbeddingTables.columnview(base, col)
            dataptr = Ptr{eltype(original)}(ptr[])
            cached = StrideArraysCore.PtrArray(dataptr, (length(original),))
            clean_check &= (cached == original)
        end
    end
    merge!(results, @dict(cached_columns, valid_backedge_check, clean_check))

    # Unique Backedge Check
    num_holes = 0
    seen_columns = Set{Int}()
    unique_backedge_check = true
    for cache in table.cache
        maxcols = filledcols(cache)
        for col in Base.OneTo(maxcols)
            # Check backedge
            backedge = cache.backedges[col]
            if iszero(backedge)
                num_holes += 1
            else
                unique_backedge_check &= !in(backedge, seen_columns)
                push!(seen_columns, backedge)
            end
        end
    end

    matching_backedge_check = (seen_columns == cached_columns)
    merge!(results, @dict(num_holes, unique_backedge_check, matching_backedge_check))

    return results
end

_format(x::String) = titlecase(replace(x, "_" => " "))
_format(x::Symbol) = _format(string(x))

getkey(x, key) = getproperty(x, key)
getkey(x::Dictionaries.AbstractDictionary, key) = getindex(x, key)
getkey(x::AbstractDict, key) = getindex(x, key)

function passed(nt; verbose = true)
    success = true
    for key in keys(nt)
        if occursin("check", string(key))
            this_success = getkey(nt, key)
            success &= this_success
            if verbose
                if this_success
                    color = :green
                    msg = "Passed"
                    bold = true
                else
                    color = :red
                    msg = "Failed"
                    bold = false
                end
                printstyled("$(_format(key)): $msg\n"; color, bold)
            end
        end
    end
    return success
end
