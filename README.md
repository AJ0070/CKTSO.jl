# CKTSO.jl

Julia wrapper for [CKTSO](https://github.com/chenxm1986/cktso), a parallel sparse direct
solver written for SPICE-style circuit simulation. CKTSO keeps the symbolic analysis and
refactorizes the same sparsity pattern with new values, which is the operation circuit
simulation spends its time in.

## Getting the library

CKTSO is distributed as a prebuilt shared library plus a license key file, and is not
redistributable, so this package does not bundle it. Upstream ships builds for **Linux
x86_64** (several glibc vintages) and **Windows x86_64**; there is no macOS or ARM build,
so this package cannot work on those platforms.

Download a build from the [CKTSO repository](https://github.com/chenxm1986/cktso), keep
`cktso.lic` in the same directory as the library, and point this package at it:

```julia
ENV["CKTSO_LIBRARY"] = "/path/to/libcktso_l.so"   # before `using CKTSO`
```

or at any time:

```julia
using CKTSO
CKTSO.set_library!("/path/to/libcktso_l.so")
```

`CKTSO.is_available()` reports whether one is configured. Every solver call throws a
`CKTSO.CKTSOError` explaining what to do if one is not.

## Use

```julia
using CKTSO, SparseArrays, LinearAlgebra

A = sprand(1000, 1000, 0.001) + 3I
b = rand(1000)

F = cktso(A)        # analyze and factorize
x = F \ b

# same pattern, new values: reuses the analysis, which is the point of CKTSO
B = copy(A)
nonzeros(B) .= nonzeros(A) .* 1.01
cktso!(F, B)
x2 = F \ b
```

`cktso!` refuses a matrix whose sparsity pattern differs from the one that was analyzed,
rather than refactorizing against the wrong analysis and returning a wrong answer.

A `SparseMatrixCSC` is passed through without conversion. CKTSO reads the index arrays as
rows, so Julia's column-major arrays describe the transpose, and the solve asks for the
column reading to put it back.

## Errors

CKTSO's return codes surface as `CKTSO.CKTSOError`, carrying the code and the operation
that failed. A singular matrix throws, as does the license error you get when `cktso.lic`
is not beside the library.

## Scope

Real, square, double-precision systems through the 64-bit-index (`_L`) entry points.
CKTSO's complex and 32-bit-index interfaces are not wrapped.
