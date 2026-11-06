# Wall-clock timing without the garbage collector in the measurement.  The solves allocate
# nothing; only the setup (workspace construction) allocates.  A full collection runs first
# (garbage of earlier runs), then the GC is switched off for the timed call.

"""Seconds for one call of f(), GC excluded."""
function timed_nogc(f)
    GC.gc()
    GC.enable(false)
    try
        return @elapsed f()
    finally
        GC.enable(true)
    end
end

"""Minimum over ≥ 3 GC-free calls of f(); short calls are repeated until ≈ 1 s was measured
(at most 15 calls), so points of a few milliseconds are not reordered by system noise."""
function min_time(f)
    ts = Float64[]; t0 = time()
    while length(ts) < 3 || (length(ts) < 15 && time() - t0 < 1.0)
        push!(ts, timed_nogc(f))
    end
    return minimum(ts)
end
