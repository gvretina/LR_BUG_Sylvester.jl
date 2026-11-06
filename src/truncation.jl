# Truncation and stopping rules shared by every method (BUG, RKSM, Tucker BUG, TBRK).
#
#   truncation : keep the smallest rank r with  ‖σ_{r+1:end}‖₂ ≤ θ · ‖σ‖₂,
#                σ = singular values of the core (per mode for Tucker: ‖σ‖₂ = ‖core‖_F);
#   stopping   : the relative residual of the approximation the method RETURNS,
#                ‖𝓛(Y) − B‖_F / ‖B‖_F ≤ tol   (θ and tol are independent: the paper uses
#                θ = 1e-14, tol = 1e-10),
#                or stagnation of that residual: |res_k − res_{k−1}| ≤ STAGNATION · res_k.
#                (For RKSM / TBRK the stagnation must come from the truncation: their
#                untruncated residual must be ≤ 0.1 · res, see rksm.jl.)

const STAGNATION = 0.01

"""Smallest r ≥ 1 with ‖σ[r+1:end]‖₂ ≤ θ · ‖σ‖₂ (σ sorted in decreasing order)."""
function truncation_rank(σ, θ)
    ϑ = θ * norm(σ)
    tail = 0.0; r = length(σ)
    while r > 1 && sqrt(tail + σ[r]^2) ≤ ϑ
        tail += σ[r]^2
        r -= 1
    end
    return r
end

stagnated(res, prev) = abs(res - prev) ≤ STAGNATION * res
