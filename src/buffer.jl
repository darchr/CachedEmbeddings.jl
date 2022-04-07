"""
    CircularBuffer{T}

A thread-safe(ish) circular buffer containing objects of type `T`.
To add elements of type `T` to the buffer, the container must be locked using
`Base.lock` or `Base.trylock`. Once the lock is held, objects can be added using
`Base.push!`.

Threads may safely access this container in a read-only manner while objects are being
added.

To remove items from the buffer, see [`cleanup!`](@ref)

*NOTE*: The function `Base.push!` is not itself threadsafe.
"""
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
function dec(buffer::CircularBuffer, v::T) where {T}
    return (isone(v)) ? convert(T, buffer.maxlen) : (v - one(v))
end
Base.lastindex(buffer::CircularBuffer) = dec(buffer, head(buffer))

@inline head(buffer::CircularBuffer) = buffer.head[]
@inline sethead!(buffer::CircularBuffer, i::Integer) = (buffer.head[] = i)
@inline tail(buffer::CircularBuffer) = buffer.tail[]
@inline settail!(buffer::CircularBuffer, i::Integer) = (buffer.tail[] = i)

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

@inline function Base.getindex(buffer::CircularBuffer, i::Integer = lastindex(buffer))
    return buffer.buffer[i]
end
@inline function Base.setindex!(
    buffer::CircularBuffer{T}, v::Union{T,Nothing}, i::Integer
) where {T}
    return (buffer.buffer[i] = v)
end

Base.@propagate_inbounds function Base.push!(buffer::CircularBuffer{T}, v::T) where {T}
    @boundscheck isfull(buffer) && throw(BoundsError(buffer))
    i = head(buffer)
    @inbounds(buffer[i] = v)
    # Perform an atomic write to head.
    # The write will ensure that all previous stores (i.e., update to underlying vector)
    # are visible before the write completes.o
    #
    # In other words, no chance of another thread seeing the head be incremented BEFORE
    # the writes to the underlying buffere are complete.
    sethead!(buffer, inc(buffer, i))
    return buffer
end

# NB: This function is not thread safe.
function Base.iterate(buffer::CircularBuffer, i = tail(buffer))
    i == head(buffer) && return nothing
    return @inbounds(buffer[i]), inc(buffer, i)
end

"""
    cleanup!([f], buffer::CircularBuffer{T})

Remove all items from `buffer`, beginning from the tail of the buffer to the head.

If the optional function `f` is passed, than this function is called on each item
in the buffer. This function must have a return value of type `Bool` that indicates if
`cleanup!` should continue (`true`) of stop prematurely (`false`).

NOTE*: This function is not threadsafe and should only be called while no other threads
are trying to access the container.
"""
cleanup!(buffer::CircularBuffer) = cleanup!(Returns(true), buffer)
function cleanup!(f::F, buffer::CircularBuffer) where {F}
    while true
        isempty(buffer) && return nothing
        i = tail(buffer)
        canclean = f(buffer[i])
        if canclean
            buffer[i] = nothing
            settail!(buffer, inc(buffer, i))
        else
            return nothing
        end
    end
end
