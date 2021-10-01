"""
    _mov(src::Ptr, dest::Ptr, ::Val{N}) where {N}

Unroll move `N` elements from `src` to `ptr`.
"""
@generated function _mov(
        ::Type{SIMD.Vec{N,T}},
        dest::Ptr{UInt8},
        src::Ptr{UInt8},
        ::Val{U},
    ) where {N,T,U}

    loads = load_impl(SIMD.Vec{N,T}, U)
    stores = store_impl(SIMD.Vec{N,T}, U)
    return quote
        $(Expr(:meta, :inline))
        _src = convert(Ptr{$T}, src)
        _dest = convert(Ptr{$T}, dest)
        $(loads...)
        $(stores...)
    end
end

function load_impl(::Type{T}, U::Integer) where {T <: SIMD.Vec}
    return map(0:U-1) do j
        x = Symbol("i_$j")
        # Vector Load
        # Val(true) implies aligned load
        :($x = SIMD.vload($T, src + $(sizeof(T)) * $j, nothing, Val(true)))
    end
end

function store_impl(::Type{T}, U::Integer) where {T <: SIMD.Vec}
    return map(0:U-1) do j
        x = Symbol("i_$j")
        # Vector Store
        # first Val(true) implies aligned store
        # second Val(true) implies nontemporal store
        #
        # See SIMD.jl for documentation.
        :(SIMD.vstore($x, dst + $(sizeof(T)) * $j, nothing, Val(true), Val(true)))
    end
end
