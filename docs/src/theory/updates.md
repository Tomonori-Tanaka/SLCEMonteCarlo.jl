# Theory: the update schemes

```@meta
CurrentModule = SLCEMonteCarlo
```

## The exact single-site ΔE

A fitted SLCE energy is a sum of cluster terms

```math
E = Σ_t c_t\,(4π)^{n^{(t)}_{\mathrm{spin}}/2} Σ_μ F_t[μ] ∏_i f_i(μ_i),
```

where axis ``i`` of term ``t`` sits on site ``a(i)`` and contributes that site's
own factor — ``Z_{l_i μ_i}(\boldsymbol e_{a(i)})`` on a spin axis,
``|\boldsymbol u_{a(i)}|^{2p_i} R_{l_i μ_i}(\boldsymbol u_{a(i)})`` on a
displacement one — and ``n^{(t)}_{\mathrm{spin}}`` counts the term's spin slots. Each instance touches **distinct sites** (a constructor invariant), though
one site may carry both a spin and a displacement axis of the same term.

Freezing everything but **one channel of one site** therefore makes the energy
*linear in that channel's row block*:

```math
E = c_s · f(x_s) + \text{const},\qquad
ΔE = c_s · \bigl(f(x_s') − f(x_s)\bigr),
```

where the leave-one-out coefficient vector ``c_s`` ([`site_coeffs!`](@ref))
contracts every adjacent instance against the *other* axes' concrete factors —
including the site's own **other** channel, which the move holds fixed — and is
independent of ``x_s`` itself. The move energetics are exact for any body order —
no linearization, no small-angle assumption. β enters only in the accept step.

Every move below is built on this one identity; they differ only in what ``x_s``
is and how a candidate is proposed. The strain move is the exception — it changes
every site at once and is treated separately at the end.

## Metropolis kernel

Spin-active sites are scanned in the Hamiltonian's color-class order (a proper
coloring of the "shares a cluster instance" conflict graph; sites the spin
channel does not touch — `TiledHamiltonian.site_has_spin`, and more broadly the
energy-inactive `site_active` — stay frozen): each
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

The classical decorrelation move for continuous spins, generalized to any SLCE:
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

## The displacement move

On a joint model the same kernel runs on the second channel: one isotropic
Gaussian shift ``\boldsymbol u_s → \boldsymbol u_s + \mathrm{step\_u}·\boldsymbol ξ``
per displacement-active site, accepted on the exact ΔE above. The proposal is
symmetric, so the accept ratio is again ``\min(1, e^{−βΔE})``, and the site scan,
colouring, and RNG-consumption discipline are the spin move's — a displacement
sweep is the spin sweep with a different row block.

Two things are specific to this channel:

- **The frame.** Where a component's rigid shift is a symmetry, the chain
  random-walks along it. That flatness is **measured**, per component and per
  Cartesian direction (`H.comp_free`) — the acoustic sum rule alone does not imply
  it, and on a joint model the measurement is taken at one probe spin
  configuration, which makes it a screen rather than a proof. Along a flat
  direction the gauge coordinate is a decoupled passenger: the chain is a skew
  product whose quotient marginal is Markov and reversible *whether or not* it is
  re-centred. So `_recenter!` (every `renorm_interval` sweeps) buys the reporting
  convention and the numerical conditioning, not stationarity — and a measurement
  taken between two renormalizations sees a drifted frame, which is why a
  displacement observable subtracts the component mean itself (spec U7).
- **Boundedness is not free.** ``E(u)`` is a truncated polynomial, so ``e^{−βE}``
  need not be normalizable. That is a property of the fitted model, not of the
  sampler; it is detected (recurrence measurement + `escaped`) rather than
  constrained (spec U8), because constraining it would silently change the model
  being sampled. See [joint spin–lattice models](../guide/joint.md).

## The strain move (NPT)

The isothermal–isobaric move is **outer**: it changes the linear cell scale
``s``, hence every site at once. A symmetric step in the proposal variable
(``\ln V`` by default, or ``s``) gives a candidate ``s'``; the displacements
follow affinely (``\boldsymbol u → (s'/s)\,\boldsymbol u``, fixed scaled
coordinates) and the Hamiltonian's coefficients are replaced by the volume grid's
interpolant at ``s'``.

The accept ratio carries the Jacobian of that map on top of the Boltzmann factor:

```math
\ln A = (3N_{\mathrm{mob}} + j)\,\ln\frac{s'}{s} − \frac{ΔE}{k_BT},
\qquad j = 3\ (\ln V\ \text{proposal}),\quad j = 2\ (s\ \text{proposal}),
```

with ``N_{\mathrm{mob}}`` the number of displacement-active sites and ``ΔE``
the full state-energy difference — the configurational energy at the new
coefficients, the elastic ``n_{\mathrm{cells}}\,j_0(s)``, and ``P\,V(s)``. The
volume power is ``V^{N_{\mathrm{mob}}}`` (one factor per *mobile site*, the
scaled-coordinate Jacobian), and the re-centred COM directions are **not**
quotiented out of it: their range is the cell, so removing them would hide a
measure factor rather than eliminate it. Two consequences worth keeping in mind:

- A proposal outside the grid's domain is **rejected**, never clamped — a
  truncating clamp is an asymmetric proposal and biases the chain toward the
  boundary. The sampled ensemble is therefore the volume-constrained one, exactly.
- Without this move the chain samples the **constant-strain** ensemble
  ``F(T, ε)``: no Jacobian, no ``P\,V``. That is a different ensemble, not a
  cheaper approximation of the same one.

Under replica exchange the same weight replaces the energy in the swap rule
(``W = E + n_{\mathrm{cells}} j_0 + PV``); the volume power cancels between the
two lanes exactly and never enters it. Full derivation and the measured exponent:
`docs/specs/strain-move.md`.

Full arguments and the metastability caveats: `docs/specs/updates-stationarity.md`.
