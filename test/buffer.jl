function spintest!(buffer::CachedEmbeddings.CircularBuffer, ntests, allocsize)
    push!(buffer, CachedEmbeddings.CachePage(Vector{Tuple{Int,Int}}(undef, allocsize)))
    @test length(buffer) == 1
    count = Threads.Atomic{Int}(1)
    Threads.@threads for i in Base.OneTo(Threads.nthreads())
        while true
            v = Threads.atomic_add!(count, 1)
            v > ntests && break

            while true
                GC.safepoint()
                cache = buffer[]
                col = CachedEmbeddings.acquire!(cache)
                exit = false

                # The cache is full, try to add a new cache to the buffer.
                if col === nothing
                    if trylock(buffer)
                        # Try again - a new cache might have been added while we
                        # were trying to acquire the lock.
                        cache = buffer[]
                        col = CachedEmbeddings.acquire!(cache)

                        # Still failed, so we definitely need to add a new cache region.
                        if col === nothing
                            newcache = CachedEmbeddings.CachePage(
                                Vector{Tuple{Int,Int}}(undef, allocsize)
                            )
                            push!(buffer, newcache)

                        else
                            # Someone else managed to insert a new feature cache before we
                            # could. In this case, just update like normal.
                            cache.data[col] = (col, v)
                            exit = true
                        end
                        unlock(buffer)
                    end
                else
                    cache.data[col] = (col, v)
                    exit = true
                end
                exit && break
            end
        end
    end
end

@testset "Testing CircularBuffer" begin
    # First, test basic functionality of the buffer.
    buffer = CachedEmbeddings.CircularBuffer{Vector{Int}}(5)
    @test isempty(buffer) == true
    @test CachedEmbeddings.isfull(buffer) == false
    @test length(buffer) == 0

    # Do some stuff with locks
    try
        @test trylock(buffer) == true
        @test trylock(buffer) == false
        push!(buffer, [5])
        @test isempty(buffer) == false
        @test CachedEmbeddings.isfull(buffer) == false
        @test length(buffer) == 1
        @test buffer[lastindex(buffer)] == [5]
    finally
        unlock(buffer)
    end

    ntests = 10000
    allocsize = 500

    @test Threads.nthreads() > 1
    T = CachedEmbeddings.CachePage{Vector{Tuple{Int,Int}}}
    buffer = CachedEmbeddings.CircularBuffer{T}(43)

    # Run this tests multiple times, setting the buffer's tail pointer to the head pointer.
    # This will test that the wrap-around functionality is working correctly.
    bigtests = 20
    for _ in 1:bigtests
        CachedEmbeddings.cleanup!(buffer)
        @test isempty(buffer)
        spintest!(buffer, ntests, allocsize)
        # Now - we check the results.
        @test length(buffer) == ceil(Int, ntests / allocsize)
        caches = collect(buffer)
        for cache in caches
            cols = getindex.(cache.data, 1)
            @test cols == Base.OneTo(length(cols))
        end
        counts = mapreduce(x -> getindex.(x.data, 2), vcat, caches)
        @test sort(counts) == Base.OneTo(length(counts))
    end
end
