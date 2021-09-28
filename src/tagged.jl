struct TaggedPtr{N}
    value::UInt
end

@inline value(x::TaggedPtr) = x.value
@inline primitive(ptr::Ptr{TaggedPtr{N}}) where {N} = Ptr{UInt}(ptr)

function Base.show(io::IO, ptr::TaggedPtr{N}) where {N}
    print(io, "Tagged Ptr: ($(ptr[]), $(gettag(ptr)))")
end
@inline Base.getindex(x::TaggedPtr{N}) where {N} = Ptr{Nothing}(value(x) & ~mask(N))

@inline gettag(v::TaggedPtr{N}) where {N} = gettag(value(v), N)
@inline gettag(v::UInt, nbits) = v & mask(nbits)

@inline settag(v::TaggedPtr{N}, x) where {N} = TaggedPtr{N}(settag(value(v), N, x))
@inline function settag(v::UInt, nbits, x)
    m = mask(nbits)
    return (v & ~m) | (x & m)
end

@inline mask(n) = mask(UInt, n)
@inline mask(::Type{T}, n) where {T} = (T(2)^n) - one(T)

@inline iscached(v::TaggedPtr{N}, tag) where {N} = iscached(value(v), N, tag)
@inline iscached(v::UInt, n, tag) = (gettag(v, n) == tag)

@inline function acquire!(::Type{TaggedPtr{N}}, ptr::Ptr{UInt}, tag) where {N}
    v = TaggedPtr{N}(Atomics.atomic_ptr_load(ptr))
    success = false
    if !iscached(v, tag)
        vnew = settag(v, tag)
        u = TaggedPtr{N}(Atomics.atomic_ptr_cas!(ptr, value(v), value(vnew)))
        success = (u === v)
    end
    return (success, v[])
end

"""
    acquire!(ptr::Ptr{TaggedPtr{N}}, tag) -> (Bool, Ptr{Nothing})

Retrieve the value pointed to by `ptr`.
If the tag for the retrieved `TaggedPtr` does not match `tag`, then attempt to acquire
ownership of `ptr` by setting its tag to `tag`.

The first element of the return tuple is `true` if ownership of `ptr` is acquired.
In this case, the caller is responsible for correctly updating the value of `ptr` to a valid
cached location. If the first element if `false`, then this value is either already cached
or owned by another thread. In this case, the caller has no responsibility and may return
the retrieved `Ptr{Nothing}` directly.

This function is (at least, **SHOULD** be) threadsafe.
"""
function acquire!(ptr::Ptr{TaggedPtr{N}}, tag) where {N}
    return acquire!(TaggedPtr{N}, primitive(ptr), tag)
end

function update_with_tag!(ptrptr::Ptr{TaggedPtr{N}}, newptr::Ptr, tag) where {N}
    v = settag(UInt(newptr), N, tag)
    Atomics.atomic_ptr_xchg!(primitive(ptrptr), v)
    return newptr
end
