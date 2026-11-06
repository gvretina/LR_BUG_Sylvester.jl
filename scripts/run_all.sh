#!/usr/bin/env bash
# Regenerate all data, then the figures.  Two lanes, each pinned to 4 physical cores of one NUMA
# node with its memory bound to that node and 4 BLAS threads; nothing else should run meanwhile
# (timings).  Run from the paper/ directory:  bash scripts/run_all.sh
set -eu
cd "$(dirname "$0")/.."
export BLAS_THREADS=4
J="julia --project=. --startup-file=no"
LOG=results/logs; mkdir -p $LOG
rm -f results/*/*/data/time_vs_n_*.csv               # the sweeps append

lane() {   # lane <cpus> <node> <commands...>   (each command: a script and its arguments)
    local cpus=$1 node=$2; shift 2
    for cmd in "$@"; do
        echo "[cores $cpus] $cmd"
        numactl --physcpubind=$cpus --membind=$node $J scripts/$cmd > "$LOG/${cmd// /_}.log" 2>&1
    done
}

# The exact references are cached in results/<dim>/<case>/data/reference_n*.h5 (27 MB) with a
# fingerprint of the problem (scripts/common.jl): only the first run computes them (≈ 45 min),
# later runs reuse them.  The 2-D random tasks share one lane so that per_case.jl writes each one
# before error_vs_time.jl reads it.  Convection-diffusion runs only at the sizes of the reference
# experiment family (n = 4096 in 2-D and 3-D); its time_vs_n rows give the timings, without a figure.
# The 3-D convection-diffusion error_vs_time (TBRK: 23 points of up to ≈ 50 s each) is the
# longest single task, so it shares the lane of the 2-D random jobs.
lane 0-3 0 \
    "per_case.jl matrix random 4096"        "per_case.jl matrix random 16384" \
    "error_vs_time.jl matrix random 4096"   "error_vs_time.jl matrix random 16384" \
    "error_vs_time.jl tensor convection_diffusion 4096" &
lane 14-17 1 \
    "per_case.jl matrix laplacian 4096"     "per_case.jl matrix laplacian 16384" \
    "error_vs_time.jl matrix laplacian 4096" "error_vs_time.jl matrix laplacian 16384" \
    "time_vs_n.jl matrix laplacian BUG 128,256,512,1024,2048,4096,8192,16384,32768,65536,131072" \
    "time_vs_n.jl matrix laplacian RKSM 128,256,512,1024,2048,4096,8192,16384,32768,65536,131072" \
    "time_vs_n.jl tensor laplacian BUG 64,128,256,512,1024,2048,4096" \
    "time_vs_n.jl tensor laplacian TBRK 64,128,256,512,1024,2048,4096" \
    "time_vs_n.jl tensor random BUG 64,128,256,512,1024,2048,4096" \
    "time_vs_n.jl tensor random TBRK 64,128,256,512,1024,2048,4096" \
    "error_vs_time.jl tensor laplacian 512" "error_vs_time.jl tensor laplacian 4096" \
    "error_vs_time.jl tensor random 512"    "error_vs_time.jl tensor random 4096" \
    "per_case.jl tensor laplacian 512"      "per_case.jl tensor laplacian 4096" \
    "per_case.jl tensor random 512"         "per_case.jl tensor random 4096" \
    "per_case.jl matrix convection_diffusion 4096"  "error_vs_time.jl matrix convection_diffusion 4096" \
    "time_vs_n.jl matrix convection_diffusion BUG 4096" "time_vs_n.jl matrix convection_diffusion RKSM 4096" \
    "per_case.jl tensor convection_diffusion 4096" \
    "time_vs_n.jl tensor convection_diffusion BUG 4096" "time_vs_n.jl tensor convection_diffusion TBRK 4096" &
wait
$J scripts/plot.jl
