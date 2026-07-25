# Theory: error analysis

```@meta
CurrentModule = SLCEMonteCarlo
```

## Log-binning

Markov-chain measurements are autocorrelated: the naive standard error
``σ/\sqrt{n}`` underestimates the true uncertainty by
``\sqrt{2τ_{\mathrm{int}} + 1}``. [`LogBinner`](@ref) tracks, per level ``k``, the
statistics of bin means of size ``2^{k-1}`` in a streaming cascade (O(levels)
memory — no time series is stored). As the bin size passes the correlation time
the binned error grows to a plateau; the reported error is the naive error at the
deepest level still holding ≥ 32 bins, and

```math
τ_{\mathrm{int}} = \tfrac12\left(\left(\frac{ε_{\mathrm{plateau}}}{ε_{\mathrm{naive}}}\right)^2 − 1\right).
```

With 32 bins the plateau estimate itself fluctuates by ≈ 13% (and ``τ``,
quadratic in it, by ≈ 26%) — fine for error bars; treat ``τ_{\mathrm{int}}`` as a
diagnostic, not a precision measurement.

A warning that applies to *any* within-chain error estimate: a chain trapped in a
metastable basin reports small ``τ_{\mathrm{int}}`` and confident error bars
about the **basin**, not the equilibrium ensemble. Cross-seed disagreement far
beyond the reported errors is the tell (and parallel tempering the fix).

## Jackknife for derived quantities

Specific heat, susceptibility, and the Binder cumulant are *nonlinear* functions
of means — plugging binned means into the formula gives no error bar and a
``O(1/n_b)`` bias. [`BinStore`](@ref) keeps `nbins` (default 32) equal-size bin
means per raw observable; [`jackknife`](@ref) then forms the leave-one-bin-out
estimates ``θ_i``, the bias-corrected value
``n_b\,θ_{\mathrm{full}} − (n_b−1)\,\bar θ``, and the error
``\sqrt{(n_b−1)/n_b\,Σ_i(θ_i−\bar θ)^2}``. For a linear function this reproduces
the plain mean and error exactly (a machine-precision test gate).
