struct CircularBuffer{T}
    head::Threads.Atomic{Int}
    tail::Threads.Atomic{Int}
    writelock::Threads.SpinLock
    maxlen::Int
    buffer::Vector{T}
end

# constructors
function CircularBuffer(buffer::Vector, maxlen::Integer)
    head = Threads.Atomic{Int}(1)
    tail = Threads.Atomic{Int}(1)
    writelock = Threads.SpinLock()
    return CircularBuffer(head, tail, writelock, maxlen, buffer)
end

function CircularBuffer(f::F, maxlen::Integer) where {F}
    buffer = [f(0) for _ in Base.OneTo(maxlen)]
    return CircularBuffer(buffer, maxlen)
end

# methods
inc(buffer::CircularBuffer, v) = (v == buffer.maxlen) ? one(v) : (v + one(v))
dec(buffer::CircularBuffer, v::T) where {T} =
    (isone(v)) ? convert(T, buffer.maxlen) : (v - one(v))

@inline head(buffer::CircularBuffer) = buffer.head[]
@inline tail(buffer::CircularBuffer) = buffer.tail[]
isempty(buffer::CircularBuffer) = (head(buffer) == tail(buffer))
isfull(buffer::CircularBuffer) = (inc(buffer, head(buffer)) == tail(buffer))

function Base.length(buffer::CircularBuffer)
    h = head(buffer)
    t = tail(buffer)
    return (h >= t) ? (h - t) : ((buffer.maxlen - t) + h)
end

Base.trylock(buffer::CircularBuffer) = trylock(buffer.writelock)
Base.unlock(buffer::CircularBuffer) = unlock(buffer.writelock)

Base.@propagate_inbounds function Base.getindex(buffer::CircularBuffer)
    @boundscheck isempty(buffer) && throw(BoundsError(buffer))
    return buffer.buffer[dec(buffer, head(buffer))]
end

Base.@propagate_inbounds function Base.push!(buffer::CircularBuffer{T}, v::T) where {T}
    @boundscheck isfull(buffer) && throw(BoundsError(buffer))
    h = head(buffer)
    @inbounds(buffer.buffer[h] = v)
    buffer.head[] = inc(buffer, h)
    return buffer
end

# NB: This function is not thread safe.
# This is mainly for testing.
function Base.iterate(buffer::CircularBuffer, i = tail(buffer))
    i == head(buffer) && return nothing
    return @inbounds(buffer.buffer[i]), inc(buffer, i)
end
