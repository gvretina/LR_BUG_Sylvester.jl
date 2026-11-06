# The paper figures from results/<dim>/<case>/data → results/<dim>/<case>/figures (PDF, EPS, PNG);
# figures combining several cases → results/<dim>/figures.
#   julia --project=. scripts/plot.jl
# Style of the published figures: CairoMakie, LaTeX fonts, seaborn-colorblind palette,
# fontsize 24, sizes in multiples of 1.5 inch.  BUG = blue ●, RKSM / TBRK = green ▲.
using CairoMakie, MakiePublication, Colors, LaTeXStrings, Serialization, DelimitedFiles, Printf

const RESULTS = normpath(joinpath(@__DIR__, "..", "results"))
const C = RGB.(MakiePublication.seaborn_colorblind())
const COLOR = Dict("BUG" => C[1], "RKSM" => C[3], "TBRK" => C[3], "ref" => C[4], "X" => C[8])
const MARKER = Dict("BUG" => :circle, "RKSM" => :utriangle, "TBRK" => :utriangle)
const RES_LABEL = L"\Vert \mathcal{L}(Y\,) - B \Vert_F / \Vert B \Vert_F"

set_theme!(merge(theme_latexfonts(), Theme(fontsize = 24, palette = (color = C,),
    Axis = (titlefont = :regular,), Legend = (framevisible = true,),
    ScatterLines = (markersize = 10, linewidth = 1.5))))

figsize(w, h) = (round(Int, 1.5 * 96w), round(Int, 1.5 * 96h))     # w × h inches, ×1.5
"""Save as PDF, EPS and PNG in results/<path...>/figures."""
function savefig(fig, name, path...)
    dir = mkpath(joinpath(RESULTS, path..., "figures"))
    for ext in ("pdf", "eps", "png")
        save(joinpath(dir, "$name.$ext"), fig)
    end
    println(joinpath(dir, name))
end
datafile(dim, case, name) = joinpath(RESULTS, dim, case, "data", name)
load(dim, case, name) = (f = datafile(dim, case, name); isfile(f) ? deserialize(f) : (println("missing ", f); nothing))
legend_entry(m) = [LineElement(color = COLOR[m]), MarkerElement(color = COLOR[m], marker = MARKER[m], markersize = 10)]

pow10ticks(lo, hi; step = 1) = (ks = floor(Int, log10(lo)):step:ceil(Int, log10(hi));
                                (10.0 .^ ks, [latexstring("10^{$k}") for k in ks]))
pow2ticks(lo, hi; step = 2) = (ks = floor(Int, log2(lo)):step:ceil(Int, log2(hi));
                               (2.0 .^ ks, [latexstring("2^{$k}") for k in ks]))

"""Log-axis ticks on [lo, hi]: decades, or 1-2-5 / 1...9 mantissas when fewer than 2 decades fall inside."""
function timeticks(lo, hi)
    for ms in ((1,), (1, 2, 5), 1:9)
        v = [m * 10.0^k for k in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in ms]
        v = filter(x -> lo / 1.05 ≤ x ≤ 1.05hi, v)
        length(v) ≥ 2 || continue
        ms == (1,) && return pow10ticks(lo, hi)
        return (v, [latexstring(@sprintf("%g", x)) for x in v])
    end
    return pow10ticks(lo, hi)
end

# -- matrix, per case: residual vs iterations | σ_k/σ₁ | |σ_X − σ_Y| --------------
function matrix_figure(d, name)
    fig = Figure(size = figsize(7, 3))
    ax1 = Axis(fig[1, 1:2]; ylabel = RES_LABEL, yscale = log10)
    for m in ("BUG", "RKSM")
        res = d[Symbol(m)].res
        scatterlines!(ax1, 0:length(res)-1, res; color = COLOR[m], marker = MARKER[m], label = m)
    end
    hlines!(ax1, d.LXr; color = COLOR["ref"], label = L"\Vert \mathcal{L}(X_r\,) - B \Vert_F / \Vert B \Vert_F")
    axislegend(ax1; position = :rt, labelsize = 20)

    kmax = min(length(d.σX), 3d.rX); σ1 = d.σX[1]
    ax2 = Axis(fig[1, 3:4]; ylabel = L"\sigma_k\,/\sigma_1", yscale = log10, xticks = [1, d.rX, kmax])
    scatterlines!(ax2, 1:kmax, d.σX[1:kmax] ./ σ1; color = COLOR["X"], marker = :circle, markersize = 8, label = L"\sigma_X")
    for m in ("BUG", "RKSM")
        σ = d[Symbol(m)].σ
        scatterlines!(ax2, 1:length(σ), σ ./ σ1; color = COLOR[m], marker = MARKER[m],
                      label = latexstring("\\sigma_Y\\ \\mathrm{$m}"))
    end
    hlines!(ax2, d.θ / σ1; color = COLOR["ref"], label = L"\vartheta")
    axislegend(ax2; position = :rt, labelsize = 18)

    r = min(length(d.BUG.σ), length(d.RKSM.σ), d.rX)
    gap(σ) = max.(abs.(d.σX[1:r] .- σ[1:r]), 1e-18σ1)
    dv = [gap(d.BUG.σ); gap(d.RKSM.σ)]
    ax3 = Axis(fig[1, 5:6]; ylabel = L"|\sigma_X - \sigma_Y\,|", yscale = log10, xticks = [1, r ÷ 2, r],
               yticks = pow10ticks(minimum(dv), maximum(dv)))
    for m in ("BUG", "RKSM")
        scatterlines!(ax3, 1:r, gap(d[Symbol(m)].σ); color = COLOR[m], marker = MARKER[m], label = m)
    end
    axislegend(ax3; position = :lt, labelsize = 18)
    Label(fig[2, 1:2], "Iterations"; padding = (0, 0, 0, 0))             # both x labels in one row
    Label(fig[2, 3:6], "Index of singular value"; padding = (0, 0, 0, 0))
    rowgap!(fig.layout, 1, 5)
    savefig(fig, name, "matrix", d.case)
end

# -- tensor, per case: residual vs iterations ---------------------------------
"""Title with the multilinear rank of the final Y (one tuple if both methods agree, else two lines)."""
function rank_title(d)
    tup(r) = "(" * join(r, ",\\, ") * ")"
    d.BUG.ranks == d.TBRK.ranks && return latexstring("Multilinear rank of \$Y = $(tup(d.BUG.ranks))\$")
    return [latexstring("Multilinear rank of \$Y\$"), latexstring("BUG \$$(tup(d.BUG.ranks))\$, TBRK \$$(tup(d.TBRK.ranks))\$")]
end

function tensor_figure(d, name)
    fig = Figure(size = figsize(4, 3))
    title = rank_title(d)
    ax = Axis(fig[1, 1]; ylabel = RES_LABEL, xlabel = "Iterations", yscale = log10,
              title = title isa Vector ? title[1] : title, subtitle = title isa Vector ? title[2] : "",
              subtitlesize = 24)
    for m in ("BUG", "TBRK")
        res = d[Symbol(m)].res
        scatterlines!(ax, 0:length(res)-1, res; color = COLOR[m], marker = MARKER[m], label = m)
    end
    isfinite(d.LXr) && hlines!(ax, d.LXr; color = COLOR["ref"], label = L"\Vert \mathcal{L}(X_r\,) - B \Vert_F / \Vert B \Vert_F")
    axislegend(ax; position = :rt)
    savefig(fig, name, "tensor", d.case)
end

for case in ("laplacian", "random"), n in (4096, 16384)
    d = load("matrix", case, "per_case_n$n.jls"); d === nothing || matrix_figure(d, "fig_matrix_$(case)_n$n")
end
for case in ("laplacian", "random"), n in (512, 4096)
    d = load("tensor", case, "per_case_n$n.jls"); d === nothing || tensor_figure(d, "fig_tucker_$(case)_n$n")
end
# convection-diffusion: one size, n = 4096
let d = load("matrix", "convection_diffusion", "per_case_n4096.jls")
    d === nothing || matrix_figure(d, "fig_matrix_convection_diffusion_n4096")
end
let d = load("tensor", "convection_diffusion", "per_case_n4096.jls")
    d === nothing || tensor_figure(d, "fig_tucker_convection_diffusion_n4096")
end

# -- accuracy vs wall-clock time: 2 × 2, and one figure per case at its largest n -----
const CASE_TITLE = Dict("laplacian" => "Laplacian", "random" => "Random", "convection_diffusion" => "Convection-diffusion")
const ERR_LABEL = Dict("matrix" => L"\Vert X_r - Y \Vert_F / \Vert X_r \Vert_F", "tensor" => RES_LABEL)
const METHODS = Dict("matrix" => ("BUG", "RKSM"), "tensor" => ("BUG", "TBRK"))

"""One accuracy-vs-time axis at fig[pos...] from the error_vs_time data d (y: error in 2-D, residual in 3-D)."""
function error_vs_time_axis!(pos, d; xlabel = "Wall-clock time [s]", ylabel = ERR_LABEL[d.dim])
    tall = vcat([c.t[c.t .> 0] for c in values(d.curves)]...)
    ax = Axis(pos; xscale = log10, yscale = log10, title = latexstring("$(CASE_TITLE[d.case]), \$n = $(d.n)\$"),
              xlabel, ylabel, xticks = timeticks(minimum(tall), maximum(tall)), yticks = pow10ticks(1e-12, 1; step = 3))
    for m in METHODS[d.dim]
        c = d.curves[m]; keep = c.t .> 0                        # the k = 0 point (Y = 0) has t = 0
        scatterlines!(ax, c.t[keep], c.y[keep]; color = COLOR[m], marker = MARKER[m])
    end
    return ax
end

"""
Columns Laplacian | Random, one row per (dim, n) of `rows`; one legend per dimension (spanning
its rows).  Rows of both dimensions: each row is labelled Matrix / Tensor.
"""
function error_vs_time_figure(rows, name, path...)
    fig = Figure(size = figsize(7, 5.4))
    dims = unique(first.(rows)); mixed = length(dims) > 1
    for (i, (dim, n)) in enumerate(rows), (j, case) in enumerate(("laplacian", "random"))
        d = load(dim, case, "error_vs_time_n$n.jls"); d === nothing && continue
        error_vs_time_axis!(fig[i, j], d; xlabel = i == length(rows) ? "Wall-clock time [s]" : "",
                            ylabel = j == 1 ? ERR_LABEL[dim] : "")
    end
    for dim in dims
        is = findall(r -> first(r) == dim, collect(rows))
        Legend(fig[first(is):last(is), 3], [legend_entry(m) for m in METHODS[dim]], collect(METHODS[dim]))
        mixed && Label(fig[first(is):last(is), 0], titlecase(dim); rotation = π / 2, font = :bold, tellheight = false)
    end
    savefig(fig, name, path...)
end
error_vs_time_figure((("matrix", 4096), ("matrix", 16384)), "fig_matrix_error_vs_time", "matrix")
error_vs_time_figure((("tensor", 512), ("tensor", 4096)), "fig_tensor_error_vs_time", "tensor")
error_vs_time_figure((("matrix", 16384), ("tensor", 4096)), "fig_error_vs_time")      # → results/figures

for (dim, case, n) in (("matrix", "laplacian", 16384), ("matrix", "convection_diffusion", 4096), ("matrix", "random", 16384),
                       ("tensor", "laplacian", 4096), ("tensor", "convection_diffusion", 4096), ("tensor", "random", 4096))
    d = load(dim, case, "error_vs_time_n$n.jls"); d === nothing && continue
    fig = Figure(size = figsize(4, 3))
    ax = error_vs_time_axis!(fig[1, 1], d)
    axislegend(ax, [legend_entry(m) for m in METHODS[dim]], collect(METHODS[dim]); position = :rt)
    savefig(fig, "fig_$(dim)_$(case)_error_vs_time_n$n", dim, case)
end

# -- time, iterations and rank vs n -------------------------------------------
"""A sweep CSV (time_vs_n.jl) as columns; x = its first column (n)."""
function sweep(file)
    isfile(file) || (println("missing ", file); return nothing)
    M, h = readdlm(file, ','; header = true); col(name) = Float64.(M[:, findfirst(==(name), vec(h))])
    ok = M[:, 2] .∈ Ref(("converged", "stagnated"))
    i = sortperm(Float64.(M[ok, 1]))
    return (; x = Float64.(M[ok, 1][i]), (Symbol(c) => col(c)[ok][i] for c in ("time", "iters", "rank", "workdim", "residual"))...)
end

"""Integer ticks on [lo, hi]: step 1, 2, 5, 10, ... so that at most 6 ticks appear."""
function intticks(lo, hi)
    step = first(s for s in (1, 2, 5, 10, 20, 50, 100) if (hi - lo) / s ≤ 5)
    return collect(step * fld(lo, step):step:step * cld(hi, step))
end

"""
One panel per (column, label) of `panels` vs x, one curve per method; `files[m]` = that
method's sweep CSV.  Time and residual on log axes, the rest linear with integer ticks.
"""
function sweep_figure(files, methods, (xscale, xlabel, xt), panels, name, path...)
    data = Dict(m => sweep(files[m]) for m in methods)
    s = filter(!isnothing, collect(values(data))); isempty(s) && return
    fig = Figure(size = figsize(7 * length(panels) / 3, 3.2))
    for (j, (field, label)) in enumerate(panels)
        y = vcat((x[field] for x in s)...)
        logy = field in (:time, :residual)
        ax = Axis(fig[1, j]; xscale, xlabel, ylabel = label, xticks = xt, yscale = logy ? log10 : identity,
                  yticks = field == :time ? timeticks(extrema(y)...) : logy ? pow10ticks(extrema(y)...) : intticks(extrema(y)...))
        for m in methods
            data[m] === nothing || scatterlines!(ax, data[m].x, data[m][field]; color = COLOR[m], marker = MARKER[m])
        end
        j == 1 && axislegend(ax, [legend_entry(m) for m in methods], collect(methods); position = :rb)
    end
    savefig(fig, name, path...)
end

const N2D = (log2, L"n", pow2ticks(128, 2^17; step = 4))
const N3D = (log2, L"n", pow2ticks(64, 4096))
const TIME, ITERS = (:time, "Time [s]"), (:iters, "Iterations")
const WORKRANK, MAXRANK = (:workdim, "Working rank"), (:rank, "Max multilinear rank")
csvs(dim, case, file, methods) = Dict(m => datafile(dim, case, "$(file)_$m.csv") for m in methods)
sweep_figure(csvs("matrix", "laplacian", "time_vs_n", ("BUG", "RKSM")), ("BUG", "RKSM"), N2D,
             (TIME, ITERS, WORKRANK), "fig_comparison_2d", "matrix", "laplacian")
sweep_figure(csvs("tensor", "laplacian", "time_vs_n", ("BUG", "TBRK")), ("BUG", "TBRK"), N3D,
             (TIME, ITERS, MAXRANK), "fig_comparison_3d", "tensor", "laplacian")
sweep_figure(csvs("tensor", "random", "time_vs_n", ("BUG", "TBRK")), ("BUG", "TBRK"), N3D,
             (TIME, ITERS, MAXRANK), "fig_comparison_3d_random", "tensor", "random")
