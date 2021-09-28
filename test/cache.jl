@testset "Testing FeatureCache" begin
    data = Vector{Tuple{Int,Int}}(undef, 200)
    cache = CachedEmbeddings.FeatureCache(data)
    @test isa(cache, CachedEmbeddings.FeatureCache)

    # Make sure we're running with more than 1 thread to accurately measure the concurrency
    # properties of this code.
    @test Threads.nthreads() > 1
    count = Threads.Atomic{Int}(1)
    Threads.@threads for i in Base.OneTo(Threads.nthreads())
        while true
            col = CachedEmbeddings.acquire!(cache)
            col === nothing && break

            # Sleep for a random amount of time to make sure the ordering of the
            # values acquired from `count` are shuffled.
            sleep(rand() / 1000)
            v = Threads.atomic_add!(count, 1)
            cache.data[col] = (col, v)
            GC.safepoint()
        end
    end

    # If we made it here - horray! The above code finished running!
    data = getindex.(cache.data, 1)
    parents = getindex.(cache.data, 2)
    @test data == Base.OneTo(length(data))
    @test !issorted(parents)
    @test sort(parents) == Base.OneTo(length(data))
    @test cache.insert[] == 201
end
