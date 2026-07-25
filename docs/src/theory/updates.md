# Theory: the update schemes

```@meta
CurrentModule = SLCEMonteCarlo
```

## The exact single-spin ΔE

A fitted SCE energy is a sum of cluster terms
``E = Σ_k c_k (4π)^{N_k/2} Σ_μ F_k[μ] ∏_i Z_{l_iμ_i}(\boldsymbol e_{s_i})``,
each instance touching **distinct** sites (a constructor invariant). Freezing all
spins but site ``s`` therefore makes the energy *linear in the tesseral row* of
``\boldsymbol e_s``:

```math
E = c_s · Z(\boldsymbol e_s) + \text{const},\qquad
ΔE = c_s · \bigl(Z(\boldsymbol e_s') − Z(\boldsymbol e_s)\bigr),
```

where the leave-one-out coefficient vector ``c_s`` ([`site_coeffs!`](@ref))
contracts every adjacent instance against the *other* sites' concrete harmonics
and is independent of ``\boldsymbol e_s`` itself. The move energetics are exact
for any body order — no linearization, no small-angle assumption. β enters only
in the accept step.

## Metropolis kernel

Active sites are scanned in the Hamiltonian's color-class order (a proper
coloring of the "shares a cluster instance" conflict graph; inactive sites —
`TiledHamiltonian.site_active`, spin-independent energy — stay frozen): each
single-site kernel is π-reversible, and a composition of π-stationary kernels in
any fixed order is π-stationary. Sites within one class have exactly independent
kernels, so a class may be updated by several tasks concurrently with a
bit-identical result (each site owns its RNG stream; `sweep_tasks` — see the
parallelism guide and `docs/specs/updates-stationarity.md` U1). No RNG is
consumed for site selection, which keeps runs bit-reproducible. The proposal is a symmetric
two-component mixture — an
antipodal flip with probability 0.2 (ergodicity between the ± lobes of a bimodal
single-site potential) or a Rodrigues rotation by `step·randn` about a uniform
axis. Acceptance `ΔE ≤ 0 || rand < exp(−βΔE)`, with the uniform drawn **only**
when needed (the RNG-consumption contract).

The proposal `step` adapts toward a target acceptance during thermalization only
and freezes for measurement: a step that keeps responding to chain history makes
the kernel history-dependent — a finite-run bias source — and would break
bit-identical restart.

## Overrelaxation

The classical decorrelation move for continuous spins, generalized to any SCE:
reflect ``\boldsymbol e → 2(\boldsymbol e·\hat h)\hat h − \boldsymbol e`` about
the local ``l=1`` field axis ``\hat h`` (read off ``c_s``'s three ``l=1``
components), then Metropolis-accept on the **exact** ΔE.

Stationarity: the reflection is a deterministic involution (`S∘S = id`), an
isometry of the sphere, and its axis depends only on the other spins — for such
proposals ``π(e)\,A(e→Se) = \min(π(e), π(Se)) = π(Se)\,A(Se→e)`` holds pointwise,
so each site kernel is reversible. For a pure-``l=1`` site channel the reflection
conserves the site energy exactly (``ΔE ≡ 0``, always accepted — classical
microcanonical overrelaxation); the ``l ≥ 2`` / multi-body remainder is corrected
exactly by the accept step. Overrelaxation alone is not ergodic (it conserves
``\boldsymbol e·\boldsymbol h`` in the pure case), so it only ever runs inside
compound sweeps with Metropolis.

Full arguments and the metastability caveats: `docs/specs/updates-stationarity.md`.
