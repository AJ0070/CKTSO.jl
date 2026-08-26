using CKTSO
using LinearAlgebra
using SparseArrays
using Random
using Test

@testset "CKTSO" begin
    if !CKTSO.is_available()
        @info "CKTSO library not configured; set CKTSO_LIBRARY to run the solver tests"
        @test_throws ErrorException CKTSO.library()
    else
        Random.seed!(392)

        @testset "solves a nonsymmetric system" begin
            n = 40
            A = sprand(n, n, 0.15) + 3I
            b = rand(n)
            F = cktso(A)
            @test size(F) == (n, n)
            x = F \ b
            @test x ≈ Matrix(A) \ b rtol = 1.0e-9
            # Not the transpose: CKTSO reads the CSC arrays as rows, so the column mode
            # in `ldiv!` is what keeps this from silently solving `A'x = b`.
            @test !isapprox(x, Matrix(A)' \ b, rtol = 1.0e-6)
        end

        @testset "refactorizes the same pattern" begin
            n = 30
            A = sprand(n, n, 0.2) + 3I
            F = cktso(A)
            b = rand(n)
            @test F \ b ≈ Matrix(A) \ b rtol = 1.0e-9

            # same pattern, new values: this is the path CKTSO exists for
            B = copy(A)
            nonzeros(B) .= nonzeros(A) .* (1 .+ rand(length(nonzeros(A))))
            cktso!(F, B)
            @test F \ b ≈ Matrix(B) \ b rtol = 1.0e-9
        end

        @testset "rejects a changed pattern" begin
            n = 12
            # A banded pattern, so the corner below is definitely a structural zero.
            A = spdiagm(-1 => -ones(n - 1), 0 => fill(4.0, n), 1 => -ones(n - 1))
            F = cktso(A)

            C = copy(A)
            C[1, n] = 1.0
            @test nnz(C) == nnz(A) + 1
            @test_throws ArgumentError cktso!(F, C)

            # Same number of stored entries, different positions: this has to be caught
            # too, or CKTSO refactorizes against the wrong analysis and returns garbage.
            Ia, Ja, Va = findnz(A)
            Ia[1], Ja[1] = 1, n          # move one entry to a structurally empty slot
            D = sparse(Ia, Ja, Va, n, n)
            @test nnz(D) == nnz(A)
            @test_throws ArgumentError cktso!(F, D)
        end

        @testset "checks shapes" begin
            n = 10
            F = cktso(sprand(n, n, 0.3) + 3I)
            @test_throws DimensionMismatch F \ rand(n + 1)
            @test_throws DimensionMismatch cktso!(F, sprand(n + 1, n + 1, 0.3) + 3I)
        end

        @testset "reports a singular matrix" begin
            A = sparse([1.0 0.0; 0.0 0.0])
            @test_throws CKTSO.CKTSOError cktso(A)
        end
    end
end
