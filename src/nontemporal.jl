function load_impl(::Type{T}, U::Integer) where {T<:SIMD.Vec}
    return map(0:(U - 1)) do j
        x = Symbol("i_$j")
        # Vector Load
        # Val(true) implies aligned load
        :($x = SIMD.vload($T, src + $(sizeof(T)) * $j, nothing, Val(true)))
    end
end

function store_nt_impl(::Type{T}, U::Integer) where {T<:SIMD.Vec}
    return map(0:(U - 1)) do j
        x = Symbol("i_$j")
        # Vector Store
        # first Val(true) implies aligned store
        # second Val(true) implies nontemporal store
        #
        # See SIMD.jl for documentation.
        :(SIMD.vstore($x, dst + $(sizeof(T)) * $j, nothing, Val(true), Val(true)))
    end
end
