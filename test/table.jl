@testset "Testing Table Mechanics" begin
    base = rand(Float32, 16, 10_000)
    A = CachedEmbeddings.CachedEmbedding{Static{featuresize(base)}}(base, Val(3))
    @test CachedEmbeddings.nbits(A) == 3
    @test CachedEmbeddings.strategy(A) == CachedEmbeddings.BlockBased()
    B = SimpleEmbedding{Static{featuresize(base)}}(base)

    # Single Lookups
    num_generations = 100
    max_cache_length = 5
    for _ in Base.OneTo(num_generations)
        inds = rand(1:size(base, 2), 100)
        outA = lookup(A, inds)
        outB = lookup(B, inds)
        @test outA == outB

        # Set up for the next generation.
        CachedEmbeddings.next!(A)
        cache_blocks = length(A.cache)
        CachedEmbeddings.unsafe_drop!(A, cache_blocks - max_cache_length)
        result = CachedEmbeddings.check(A; clean = true)
        @test CachedEmbeddings.passed(result; verbose = false)
    end

    # Ensemble Lookups
    num_generations = 100
    max_cache_length = 5
    nlookups = 40
    for _ in Base.OneTo(num_generations)
        inds = rand(1:size(base, 2), 40, 100)
        outA = lookup(A, inds)
        outB = lookup(B, inds)
        @test outA == outB

        # Set up for the next generation.
        CachedEmbeddings.next!(A)
        cache_blocks = length(A.cache)
        CachedEmbeddings.unsafe_drop!(A, cache_blocks - max_cache_length)
        result = CachedEmbeddings.check(A; clean = true)
        @test CachedEmbeddings.passed(result; verbose = false)
    end

    # Now start multithreading
    num_generations = 100
    batchsize = 1024
    max_cache_length = 5
    nlookups = 40
    strategy = EmbeddingTables.PreallocationStrategy()
    vA = [A]
    vB = [B]

    for _ in Base.OneTo(num_generations)
        inds = rand(1:size(base, 2), 40, batchsize)
        outA = maplookup(strategy, vA, inds)
        outB = maplookup(strategy, vB, inds)
        @test outA == outB

        # Set up for the next generation.
        CachedEmbeddings.next!(A)
        cache_blocks = length(A.cache)
        CachedEmbeddings.unsafe_drop!(A, cache_blocks - max_cache_length)
        result = CachedEmbeddings.check(A; clean = true)
        @test CachedEmbeddings.passed(result; verbose = false)
    end
end

@testset "Testing Writebacks" begin
    batchsize = 128
    featuresize = 64
    ncols = 10_000

    base = randn(Float32, featuresize, ncols)
    S = Static{featuresize}
    reference = SimpleEmbedding{S}(copy(base))
    cached = CachedEmbeddings.CachedEmbedding{S}(
        copy(base),
        Val(3);
        # Target roughly 5 batches in the cache with 2 batches in reserve
        targetbytes = 5 * featuresize * batchsize * sizeof(Float32),
        targetreserve = 2 * featuresize * batchsize * sizeof(Float32),
        init = CachedEmbeddings.DefaultInit(base, batchsize),
    )

    opt = EmbeddingTables.Flux.Descent(1.0)
    for i in Base.OneTo(7)
        inds = rand(Base.OneTo(ncols), batchsize)
        # Perform Lookup
        out_cached = lookup(cached, inds)
        out_reference = lookup(reference, inds)
        @test out_cached == out_reference

        back_cached = SparseEmbeddingUpdate{S}(out_cached, inds)
        back_reference = SparseEmbeddingUpdate{S}(out_reference, inds)

        EmbeddingTables.Flux.update!(opt, cached, back_cached)
        EmbeddingTables.Flux.update!(opt, reference, back_reference)

        @test cached == reference

        # Since updates are not being written back to the main memory yet, the base should
        # become stale.
        @test cached.base != reference

        # Should fail the test if we think it's clean
        results = CachedEmbeddings.check(cached; clean = true)
        @test CachedEmbeddings.passed(results; verbose = false) == false
        results = CachedEmbeddings.check(cached; clean = false)
        @test CachedEmbeddings.passed(results; verbose = false) == true

        # Prepare for next iteration.
        CachedEmbeddings.next!(cached)
    end

    # Now start messing around with the state of the cache.
    # Since we sized a page to be just around the size of a batch and we don't expect
    # many repeats, expect the same number of pages as iterations.
    numpages = length(cached.cache)
    @test numpages == 7

    # Now, the number of pages in the cache exceeds our perscribed limit.
    # So, when we get work for cleanup - we would expect a couple of pages
    work, npages = CachedEmbeddings.get_writeback_work(cached)
    @test length(work) == only(npages)
    @test npages == [2]

    # Performing a writeback should not change the apparent state of the cache.
    CachedEmbeddings.writeback!([cached])
    @test CachedEmbeddings.passed(
        CachedEmbeddings.check(cached; clean = false); verbose = false
    )
    @test cached == reference

    @test length(cached.cache) == 5
    @test length(cached.pagecache) == 2
end
