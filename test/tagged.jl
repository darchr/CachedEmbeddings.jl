@testset "Testing Tagged" begin
    # Basic Utils
    v = UInt(0x1905e88)
    p = CachedEmbeddings.TaggedPtr{3}(v)
    @test CachedEmbeddings.value(p) === v
    @test isa(p[], Ptr{Nothing})
    @test UInt(p[]) == v & ~7

    # More advanced utils
    p = CachedEmbeddings.TaggedPtr{3}(0)
    @test CachedEmbeddings.gettag(p) == 0
    @test CachedEmbeddings.iscached(p, 0) == true
    @test CachedEmbeddings.iscached(p, 1) == false

    @test CachedEmbeddings.gettag(p) == 0
    p = CachedEmbeddings.settag(p, 5)
    @test CachedEmbeddings.gettag(p) == 5
    @test CachedEmbeddings.iscached(p, 0) == false
    @test CachedEmbeddings.iscached(p, 1) == false
    @test CachedEmbeddings.iscached(p, 5) == true

    # Test `acquire`
    x = Vector{CachedEmbeddings.TaggedPtrPair{3}}()
    push!(x, CachedEmbeddings.TaggedPtrPair{3}(0x1230))
    own, val = CachedEmbeddings.acquire!(CachedEmbeddings.taggedpointer(x, 1), 1)
    @test own == true
    @test val == Ptr{Nothing}(UInt(0x1230))

    # Second time should return false since this `TaggedPtr` is now owned.
    @test CachedEmbeddings.gettag(x[1].tagged) == 1
    @test CachedEmbeddings.iscached(x[1].tagged, 1) == true
    own, val = CachedEmbeddings.acquire!(CachedEmbeddings.taggedpointer(x, 1), 1)
    @test own == false
    @test val == Ptr{Nothing}(UInt(0x1230))

    # Force set the tag back to zero.
    CachedEmbeddings.release!(CachedEmbeddings.taggedpointer(x, 1), UInt(0x1230), 2)
    @test CachedEmbeddings.gettag(x[1].tagged) == 2
    @test x[1].tagged[] == Ptr{Nothing}(UInt(0x1230))

    # Finally, test that the update function works
    newptr = Ptr{Nothing}(typemax(UInt) & ~7)
    ptr = CachedEmbeddings.update_with_tag!(pointer(x, 1), newptr, Ptr{UInt64}(1), 5)
    @test ptr == newptr
    @test CachedEmbeddings.gettag(x[1].tagged) == 5
    @test x[1].backedge == Ptr{UInt64}(1)
    @test CachedEmbeddings.follow(Nothing, x[1]) == newptr
end
