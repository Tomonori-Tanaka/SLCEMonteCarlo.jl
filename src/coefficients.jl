# Coefficient hot-swap: rewrite a built `TiledHamiltonian`'s coefficients without
# rebuilding its tiling, adjacency, coloring or program index arrays.
#
# WHY THIS IS POSSIBLE AT ALL. The program entry SET is coefficient-independent: the
# site-program skip is on `folded[idx]`, not on `coef · folded[idx]`, so a term's
# entries neither appear nor vanish when its coefficient changes, and the weight
# stream factors as `sent_w[i] == term_coef[sent_term[i]] · sent_base[i]`. A rewrite is
# then a fused multiply over that stream plus one scalar per template.
#
# WHY IT NEEDS `keep_zero_terms`. Everything structural — `site_active`,
# `site_has_spin` / `site_has_disp` and their counts, the coloring, the displacement-
# coupling components — is derived from the term list the Hamiltonian was BUILT with.
# A term dropped by the default zero prune has no program to rewrite, so a coefficient
# set that makes it nonzero is not representable. `keep_zero_terms = true` at
# construction is what makes those structures a property of the support rather than of
# the values.
#
# Measured cost (Nd₂Fe₁₄B, 4692 terms): a full rebuild is ~3× one sweep at both 4³ and
# 8³, so a rebuild-per-strain-move would make a run ~4× slower — which is what this
# path exists to avoid.

"""
    set_coefficients!(H::TiledHamiltonian, coefs; recheck_translation = true) -> H

Rewrite `H`'s coefficients in place from `coefs`, a vector of **raw** (unscaled)
coefficients indexed by the term list `H` was built from — `spin_multipole_terms(model)`
or `decorated_terms(model)`, in that order, including any term the zero prune dropped.
That is **one entry per term**, not per SALC: `coef(model)` is the per-SALC vector and is
the wrong granularity here (it throws, but it is the natural thing to reach for).
The `(4π)^(n_spin_slots/2)` scale each term was ingested with is re-applied here, so
`coefs` is in the same units `coef(model)` reports; the scale is never re-derived from
the cluster shape.

Everything else about `H` is untouched: the instance enumeration, the per-site
adjacency, the coloring, the displacement-coupling components and every program index
array are shared, and only the weight streams and the per-template coefficients are
written. That is what makes this affordable at the cadence a strain move needs
(a full rebuild costs ~3× a sweep).

# The support is fixed at construction

A term whose coefficient was exactly `0.0` when `H` was built is **not present** —
the default prune drops it so that "no adjacent instance" keeps meaning
"state-independent site". Passing a nonzero coefficient for such a term is an error,
not a silent no-op. Build with `keep_zero_terms = true` when you intend to hot-swap
across a family whose support varies:

```julia
H = TiledHamiltonian(model; dims = (4, 4, 4), keep_zero_terms = true)
set_coefficients!(H, [t.coef for t in decorated_terms(next_model; keep_zero = true)])
```

The cost of `keep_zero_terms` is that a site coupled only through currently-zero terms
counts as active. That is the honest accounting: such a site is *sampled*, with every
move accepted at `ΔE = 0`, so it is a free spin that belongs in `⟨m⟩` — unlike a
structurally absent site, which is frozen at its initial direction and must be excluded.

# What is verified, and what goes stale

`recheck_translation = true` (the default) re-measures the rigid-shift flatness and
**refuses** a coefficient set that changes the verdict `comp_free` — a joint model whose
sampler re-centres along a direction that is no longer a symmetry would bias silently,
and no other gate would notice. The measurement costs several full-energy passes, so a
caller rewriting once per sweep should verify once and then pass
`recheck_translation = false`, having established that its family preserves the ASR (a
grid of ASR-constrained fits does by construction). The stored `comp_residual` /
`translation_residual` numbers are the ones `H` was **built** with and are not
rewritten; the verdict they summarize is what gets re-checked.

Two things become stale elsewhere and are the caller's responsibility:

- a `GPUTiledHamiltonian` uploaded from `H` keeps the old device tables — re-upload;
- the checkpoint fingerprint changes with the coefficients (by design), so a file
  written before the swap will correctly refuse to resume after it.

See also [`TiledHamiltonian`](@ref).
"""
function set_coefficients!(H::TiledHamiltonian, coefs::AbstractVector{<:Real};
                           recheck_translation::Bool = true)::TiledHamiltonian
    length(coefs) == H.n_input_terms || throw(DimensionMismatch(
        "got $(length(coefs)) coefficients for a Hamiltonian built from " *
        "$(H.n_input_terms) terms"))
    all(isfinite, coefs) ||
        throw(ArgumentError("the coefficient vector has non-finite entries"))
    # Every input index the build dropped must still be zero. Report the count and a
    # sample rather than the first offender alone: a caller who hit this once has
    # usually hit it for a whole group of terms, and the actionable fact is the kwarg.
    present = falses(H.n_input_terms)
    @inbounds for src in H.term_source
        present[src] = true
    end
    missing_nz = Int[]
    @inbounds for i = 1:H.n_input_terms
        present[i] || coefs[i] == 0.0 || push!(missing_nz, i)
    end
    isempty(missing_nz) || throw(ArgumentError(
        "$(length(missing_nz)) coefficient(s) are nonzero for terms this Hamiltonian " *
        "does not carry (input index $(first(missing_nz, 8))$(length(missing_nz) > 8 ?
        ", …" : "")): they were dropped as exactly zero when it was built, so there is " *
        "no program to rewrite. Rebuild with " *
        "`TiledHamiltonian(...; keep_zero_terms = true)` to hot-swap across a family " *
        "whose support varies."))

    old_free = copy(H.comp_free)
    pr = H.progs
    @inbounds for k in eachindex(H.terms)
        t = H.terms[k]
        c = Float64(coefs[H.term_source[k]]) * H.term_scale[k]
        H.terms[k] = ScaledTerm(c, t.atoms, t.shifts, t.slots, t.folded)
        pr.term_coef[k] = c
    end
    # The program invariant, re-established: `sent_w == term_coef[sent_term] · sent_base`.
    # The energy program carries raw `folded` and multiplies by `term_coef` per instance,
    # so it needs no rewrite at all.
    @inbounds for i in eachindex(pr.sent_w)
        pr.sent_w[i] = pr.term_coef[pr.sent_term[i]] * pr.sent_base[i]
    end

    if recheck_translation && H.n_disp_comps > 0
        new_free = _translation_residuals(H) .<= _TRANSLATION_TOL
        new_free == old_free || throw(ArgumentError(
            "these coefficients change the rigid-shift flatness verdict of the " *
            "displacement channel (was $(count(old_free))/$(length(old_free)) " *
            "free (direction, component) pairs, now $(count(new_free))): the " *
            "sampler re-centres along exactly the free directions, so continuing " *
            "would project along a direction that is no longer a symmetry. The " *
            "Hamiltonian now holds the NEW coefficients — rebuild it rather than " *
            "reusing this one. If the model is meant to be translation-invariant, " *
            "refit under the acoustic sum rule (`fit(...; asr = true)`, the default)."))
    end
    return H
end
