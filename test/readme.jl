# The README is the first thing anyone runs, so its examples are executed here rather
# than trusted. Blocks that point at a library path are configuration snippets, not
# runnable code, and are skipped.

function readme_julia_blocks(path::AbstractString)
    blocks = String[]
    buf = nothing
    for line in eachline(path)
        if buf === nothing
            startswith(line, "```julia") && (buf = IOBuffer())
        elseif startswith(line, "```")
            push!(blocks, String(take!(buf)))
            buf = nothing
        else
            println(buf, line)
        end
    end
    buf === nothing || error("unterminated code fence in $path")
    return blocks
end

@testset "README examples run" begin
    readme = joinpath(@__DIR__, "..", "README.md")
    blocks = readme_julia_blocks(readme)
    # If the extractor ever stops matching, this testset would pass by running nothing.
    @test length(blocks) >= 3

    runnable = filter(b -> !occursin("/path/to/", b), blocks)
    @test !isempty(runnable)

    for (i, code) in enumerate(runnable)
        sandbox = Module(Symbol(:READMEBlock, i))
        # Passes only if the block runs to completion without throwing.
        @test (include_string(sandbox, code); true)
    end
end
