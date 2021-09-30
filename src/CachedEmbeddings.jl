module CachedEmbeddings

# "internal" deps
import CachedArrays: CachedArrays, CachedArray
import EmbeddingTables: EmbeddingTables, AbstractEmbeddingTable, columnpointer, featuresize

# stdlib

# deps
import Dictionaries
import StaticArrays: SVector
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
include("cache.jl")
include("table.jl")
include("verification.jl")

end
