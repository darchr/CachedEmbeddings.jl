const inttypes = (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
const arithmetictypes = (inttypes...,)
const atomictypes = (arithmetictypes..., Bool)

const IntTypes = Union{inttypes...}
const ArithmeticTypes = Union{arithmetictypes...}
const AtomicTypes = Union{atomictypes...}
const WORD_SIZE = Base.Sys.WORD_SIZE

const llvmtypes = IdDict{Any,String}(
    Bool => "i8",  # julia represents bools with 8-bits for now. # TODO: is this okay?
    Int8 => "i8",
    UInt8 => "i8",
    Int16 => "i16",
    UInt16 => "i16",
    Int32 => "i32",
    UInt32 => "i32",
    Int64 => "i64",
    UInt64 => "i64",
    Int128 => "i128",
    UInt128 => "i128",
    # Float16 => "half",
    # Float32 => "float",
    # Float64 => "double",
)

# The below code is basically copied from Base.jl
# Just modified to work on raw pointers instead of `Threads.Atomic`.

# All atomic operations have acquire and/or release semantics, depending on
# whether the load or store values. Most of the time, this is what one wants
# anyway, and it's only moderately expensive on most hardware.
for typ in atomictypes
    lt = llvmtypes[typ]
    rt = "$lt, $lt*"
    @eval function atomic_ptr_load(x::Ptr{$typ})
        return Base.llvmcall(
            $"""
            %ptr = inttoptr i$WORD_SIZE %0 to $lt*
            %rv = load atomic $rt %ptr acquire, align $(Base.gc_alignment(typ))
            ret $lt %rv
            """,
            $typ,
            Tuple{Ptr{$typ}},
            x,
        )
    end

    # @eval setindex!(x::Atomic{$typ}, v::$typ) =
    #     GC.@preserve x llvmcall($"""
    #              %tr = inttoptr i$WORD_SIZE %0 to $lt*
    #              store atomic $lt %1, $lt* %ptr release, align $(gc_alignment(typ))
    #              ret void
    #              """, Cvoid, Tuple{Ptr{$typ}, $typ}, unsafe_convert(Ptr{$typ}, x), v)

    # Note: atomic_cas! succeeded (i.e. it stored "new") if and only if the result is "cmp"
    @eval function atomic_ptr_cas!(x::Ptr{$typ}, cmp::$typ, new::$typ)
        return Base.llvmcall(
            $"""
            %ptr = inttoptr i$WORD_SIZE %0 to $lt*
            %rs = cmpxchg $lt* %ptr, $lt %1, $lt %2 acq_rel acquire
            %rv = extractvalue { $lt, i1 } %rs, 0
            ret $lt %rv
            """,
            $typ,
            Tuple{Ptr{$typ},$typ,$typ},
            x,
            cmp,
            new,
        )
    end

    arithmetic_ops = [:add, :sub]
    for rmwop in [arithmetic_ops..., :xchg, :and, :nand, :or, :xor, :max, :min]
        rmw = string(rmwop)
        fn = Symbol("atomic_ptr", rmw, "!")
        if (rmw == "max" || rmw == "min") && typ <: Unsigned
            # LLVM distinguishes signedness in the operation, not the integer type.
            rmw = "u" * rmw
        end
        if rmwop in arithmetic_ops && !(typ <: ArithmeticTypes)
            continue
        end
        @eval function $fn(x::Ptr{$typ}, v::$typ)
            return Base.llvmcall(
                $"""
                %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                %rv = atomicrmw $rmw $lt* %ptr, $lt %1 acq_rel
                ret $lt %rv
                """,
                $typ,
                Tuple{Ptr{$typ},$typ},
                x,
                v,
            )
        end
    end
end
