module CachedEmbeddings

# "internal" deps
import CachedArrays: CachedArrays, CachedArray
import EmbeddingTables:
    EmbeddingTables,
    AbstractEmbeddingTable,
    columnpointer,
    featuresize,
    IndexingContext,
    Forward,
    Update

# stdlib

# deps
import Dictionaries
import Polyester
import SIMD
import StrideArraysCore

macro dict(syms...)
    keys = map(QuoteNode, syms)
    values = map(esc, syms)
    return :(Dictionaries.Dictionary{Symbol,Any}(($(keys...),), ($(values...),)))
end

include("atomics.jl")
import .Atomics: Atomics
include("buffer.jl")
include("tagged.jl")
include("page.jl")
include("nontemporal.jl")
include("table.jl")
include("verification.jl")

end
