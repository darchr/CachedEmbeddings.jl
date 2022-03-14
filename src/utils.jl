Base.@propagate_inbounds function popfirst_and_shift!(v::AbstractVector)
    @boundscheck checkbounds(v, firstindex(v))
    x = v[firstindex(v)]
    for i in 1:(length(v) - 1)
        @inbounds v[i] = v[i + 1]
    end
    resize!(v, length(v) - 1)
    return x
end
