"""
    CKTSO

Julia wrapper for [CKTSO](https://github.com/chenxm1986/cktso), a parallel sparse direct
solver specialised for SPICE-style circuit simulation.

CKTSO is distributed as a prebuilt shared library plus a license key file, and is not
redistributable, so this package does not ship it. Point `CKTSO` at your copy with the
`CKTSO_LIBRARY` environment variable, or with `CKTSO.set_library!`:

```julia
ENV["CKTSO_LIBRARY"] = "/path/to/libcktso_l.so"   # before `using CKTSO`
```

The license key (`cktso.lic`) has to sit next to the library file.
"""
module CKTSO

using Libdl: Libdl
using LinearAlgebra: LinearAlgebra, ldiv!
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, rowvals, getcolptr

export CKTSOSolver, cktso, cktso!

# --------------------------------------------------------------------------------------
# library discovery

const _LIBRARY = Ref{String}("")
const _HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _SYMBOLS = Dict{Symbol, Ptr{Cvoid}}()

"""
    CKTSO.set_library!(path)

Point this package at a CKTSO shared library. The license key file `cktso.lic` must live
in the same directory. Takes effect immediately for subsequent factorizations.
"""
function set_library!(path::AbstractString)
    isfile(path) || throw(ArgumentError("no CKTSO library at $path"))
    lic = joinpath(dirname(abspath(path)), "cktso.lic")
    isfile(lic) || @warn "no cktso.lic next to the library; CKTSO will report a license error" lic
    _HANDLE[] == C_NULL || Libdl.dlclose(_HANDLE[])
    empty!(_SYMBOLS)
    _HANDLE[] = C_NULL
    _LIBRARY[] = abspath(path)
    return _LIBRARY[]
end

"""
    CKTSO.library()

Path of the CKTSO shared library in use, or an error explaining how to set one.
"""
function library()
    isempty(_LIBRARY[]) && throw(
        ErrorException(
            "CKTSO has no library configured. CKTSO is not redistributable, so this " *
                "package does not ship it. Download it from " *
                "https://github.com/chenxm1986/cktso and either set " *
                "ENV[\"CKTSO_LIBRARY\"] before `using CKTSO`, or call " *
                "CKTSO.set_library!(path). Keep cktso.lic beside the library."
        )
    )
    return _LIBRARY[]
end

is_available() = !isempty(_LIBRARY[])

# `ccall` will not take a library path from a local variable, and the path is only known
# at run time, so the entry points are resolved to function pointers once and cached.
function _sym(name::Symbol)
    return get!(_SYMBOLS, name) do
        if _HANDLE[] == C_NULL
            _HANDLE[] = Libdl.dlopen(library())
        end
        Libdl.dlsym(_HANDLE[], name)
    end
end

function __init__()
    path = get(ENV, "CKTSO_LIBRARY", "")
    if !isempty(path) && isfile(path)
        _LIBRARY[] = abspath(path)
    end
    return nothing
end

# --------------------------------------------------------------------------------------
# return codes

const _RETURN_CODES = Dict(
    0 => "successful",
    -1 => "invalid instance handle",
    -2 => "argument error",
    -3 => "invalid matrix",
    -4 => "out of memory",
    -5 => "structurally singular",
    -6 => "numerically singular",
    -7 => "threads error",
    -8 => "matrix not analyzed",
    -9 => "matrix not factorized",
    -10 => "not supported",
    -11 => "file error",
    -12 => "integer overflow",
    -13 => "resource leak",
    -99 => "license error",
    -100 => "unknown error",
)

struct CKTSOError <: Exception
    code::Int
    op::String
end

function Base.showerror(io::IO, e::CKTSOError)
    reason = get(_RETURN_CODES, e.code, "undocumented code")
    print(io, "CKTSO ", e.op, " failed: ", reason, " (", e.code, ")")
    return if e.code == -99
        print(io, ". The cktso.lic key file has to sit next to the library.")
    end
end

_check(code::Integer, op::AbstractString) = code == 0 || throw(CKTSOError(Int(code), op))

# --------------------------------------------------------------------------------------
# the factorization

"""
    CKTSOSolver

A CKTSO solver instance holding the analysis and numeric factorization of a sparse matrix,
named the way CKTSO's own C API names it. Build one with [`cktso`](@ref) and solve with
`\\` or `ldiv!`. Refresh the values without repeating the symbolic analysis using
[`cktso!`](@ref).
"""
mutable struct CKTSOSolver{Tv <: Float64} <: LinearAlgebra.Factorization{Tv}
    inst::Ptr{Cvoid}
    n::Int64
    # `SparseMatrixCSC` is column major and CKTSO reads `ap`/`ai` as rows, so these are
    # the pattern of `transpose(A)` as far as CKTSO is concerned. Solving with
    # `row0_column1 = 1` puts it back, which is why nothing is transposed here.
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    nzval::Vector{Tv}
end

Base.size(F::CKTSOSolver) = (Int(F.n), Int(F.n))
Base.size(F::CKTSOSolver, i::Integer) = i <= 2 ? Int(F.n) : 1

function _destroy!(F::CKTSOSolver)
    if F.inst != C_NULL
        ccall(_sym(:CKTSO_L_DestroySolver), Cint, (Ptr{Cvoid},), F.inst)
        F.inst = C_NULL
    end
    return nothing
end

"""
    cktso(A::SparseMatrixCSC{Float64, <:Integer}; threads = 0)

Analyze and factorize `A`, returning a [`CKTSOSolver`](@ref). `threads = 0` uses
every physical core, `-1` every logical core.
"""
function cktso(A::SparseMatrixCSC{Float64, <:Integer}; threads::Integer = 0)
    n = LinearAlgebra.checksquare(A)
    library()  # fail early with the configuration message

    colptr = Vector{Int64}(getcolptr(A) .- 1)
    rowval = Vector{Int64}(rowvals(A) .- 1)
    nzval = Vector{Float64}(nonzeros(A))

    inst = Ref{Ptr{Cvoid}}(C_NULL)
    iparm = Ref{Ptr{Cint}}(C_NULL)
    oparm = Ref{Ptr{Clonglong}}(C_NULL)
    _check(
        ccall(
            _sym(:CKTSO_L_CreateSolver), Cint,
            (Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Clonglong}}), inst, iparm, oparm
        ), "CreateSolver"
    )

    F = CKTSOSolver{Float64}(inst[], Int64(n), colptr, rowval, nzval)
    finalizer(_destroy!, F)

    _check(
        ccall(
            _sym(:CKTSO_L_Analyze), Cint,
            (Ptr{Cvoid}, Cuchar, Clonglong, Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Cdouble}, Cint),
            F.inst, 0, F.n, F.colptr, F.rowval, F.nzval, Cint(threads)
        ), "Analyze"
    )
    _check(
        ccall(
            _sym(:CKTSO_L_Factorize), Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Cuchar), F.inst, F.nzval, 0
        ), "Factorize"
    )
    return F
end

"""
    cktso!(F::CKTSOSolver, A::SparseMatrixCSC)

Refactorize `A` reusing `F`'s symbolic analysis. `A` must have the same sparsity pattern
as the matrix `F` was built from. This is the operation CKTSO is built around: circuit
simulation refactorizes the same pattern over and over with new values.
"""
function cktso!(F::CKTSOSolver, A::SparseMatrixCSC{Float64, <:Integer})
    n = LinearAlgebra.checksquare(A)
    n == F.n || throw(DimensionMismatch("factorization is $(F.n)x$(F.n), matrix is $n x $n"))
    # Comparing only `nnz` would let a matrix with the same count but a different
    # pattern through, and CKTSO would then refactorize against the analysis of a
    # different matrix and return a wrong answer rather than an error.
    (
        length(nonzeros(A)) == length(F.nzval) &&
            all(i -> getcolptr(A)[i] - 1 == F.colptr[i], eachindex(F.colptr)) &&
            all(i -> rowvals(A)[i] - 1 == F.rowval[i], eachindex(F.rowval))
    ) ||
        throw(ArgumentError("sparsity pattern changed; build a new factorization with `cktso`"))
    copyto!(F.nzval, nonzeros(A))
    _check(
        ccall(
            _sym(:CKTSO_L_Refactorize), Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}), F.inst, F.nzval
        ), "Refactorize"
    )
    return F
end

function LinearAlgebra.ldiv!(x::Vector{Float64}, F::CKTSOSolver, b::Vector{Float64})
    length(b) == F.n || throw(DimensionMismatch("rhs has length $(length(b)), expected $(F.n)"))
    length(x) == F.n || throw(DimensionMismatch("solution has length $(length(x)), expected $(F.n)"))
    _check(
        ccall(
            _sym(:CKTSO_L_Solve), Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cuchar, Cuchar),
            # `row0_column1 = 1`: the pattern went in as CSC, so ask for the column
            # reading, which is `A x = b` rather than `transpose(A) x = b`.
            F.inst, b, x, 0, 1
        ), "Solve"
    )
    return x
end

LinearAlgebra.ldiv!(F::CKTSOSolver, b::Vector{Float64}) = ldiv!(b, F, copy(b))
Base.:\(F::CKTSOSolver, b::Vector{Float64}) = ldiv!(similar(b), F, b)

end # module
