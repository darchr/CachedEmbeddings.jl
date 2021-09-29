@testset "Testing Table Mechanics" begin
    base = rand(Float32, 16, 10_000)
    A = CachedEmbeddings.CachedEmbedding{Static{featuresize(base)}}(base, Val(3))
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
        cache_blocks = length(A.buffer)
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
        cache_blocks = length(A.buffer)
        CachedEmbeddings.unsafe_drop!(A, cache_blocks - max_cache_length)
        result = CachedEmbeddings.check(A; clean = true)
        @test CachedEmbeddings.passed(result; verbose = false)
    end
end
