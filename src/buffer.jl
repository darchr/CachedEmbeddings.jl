struct CircularBuffer{T}
    head::Threads.Atomic{Int}
    tail::Threads.Atomic{Int}
    writelock::Threads.SpinLock
    maxlen::Int
    buffer::Vector{Union{Nothing,T}}
end

# constructors
function CircularBuffer{T}(maxlen::Integer) where {T}
    head = Threads.Atomic{Int}(1)
    tail = Threads.Atomic{Int}(1)
    writelock = Threads.SpinLock()
    buffer = Vector{Union{Nothing,T}}([nothing for _ in Base.OneTo(maxlen)])
    return CircularBuffer{T}(head, tail, writelock, maxlen, buffer)
end

# methods
inc(buffer::CircularBuffer, v) = (v == buffer.maxlen) ? one(v) : (v + one(v))
dec(buffer::CircularBuffer, v::T) where {T} =
    (isone(v)) ? convert(T, buffer.maxlen) : (v - one(v))

@inline head(buffer::CircularBuffer) = buffer.head[]
@inline tail(buffer::CircularBuffer) = buffer.tail[]
Base.isempty(buffer::CircularBuffer) = (head(buffer) == tail(buffer))
isfull(buffer::CircularBuffer) = (inc(buffer, head(buffer)) == tail(buffer))

function Base.length(buffer::CircularBuffer)
    h = head(buffer)
    t = tail(buffer)
    return (h >= t) ? (h - t) : ((buffer.maxlen - t) + h)
end

@inline Base.trylock(buffer::CircularBuffer) = trylock(buffer.writelock)
@inline Base.lock(buffer::CircularBuffer) = lock(buffer.writelock)
@inline Base.unlock(buffer::CircularBuffer) = unlock(buffer.writelock)

Base.@propagate_inbounds function Base.getindex(buffer::CircularBuffer)
    @boundscheck isempty(buffer) && throw(BoundsError(buffer))
    # TODO: Perform this unsafely to avoid the type check?
    v = buffer.buffer[dec(buffer, head(buffer))]
    v === nothing && error("Something went wrong")
    return v
end

Base.@propagate_inbounds function Base.push!(buffer::CircularBuffer{T}, v::T) where {T}
    @boundscheck isfull(buffer) && throw(BoundsError(buffer))
    i = head(buffer)
    @inbounds(buffer.buffer[i] = v)
    buffer.head[] = inc(buffer, i)
    return buffer
end

# NB: This function is not thread safe.
function Base.iterate(buffer::CircularBuffer, i = tail(buffer))
    i == head(buffer) && return nothing
    return @inbounds(buffer.buffer[i]), inc(buffer, i)
end

cleanup!(buffer::CircularBuffer) = cleanup!(_ -> (true, nothing), buffer)
function cleanup!(f::F, buffer::CircularBuffer) where {F}
    Base.@lock buffer begin
        while true
            isempty(buffer) && return nothing
            i = tail(buffer)
            canclean, newval = f(buffer.buffer[i])
            if canclean
                buffer.buffer[i] = newval
                buffer.tail[] = inc(buffer, i)
            else
                return nothing
            end
        end
    end
end

