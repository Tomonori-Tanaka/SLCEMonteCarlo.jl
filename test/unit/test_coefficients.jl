# Coefficient hot-swap (`set_coefficients!`) — the machinery the strain outer move and
# the active-learning hook need: rewrite a built Hamiltonian's coefficients without
# touching its tiling, adjacency, coloring or program index arrays.
#
# Three things are pinned here. (1) The no-op round trip is BYTE-identical to a fresh
# build — that is what says the factored weight stream (`sent_w == term_coef[sent_term]
# · sent_base`) really is the same arithmetic. (2) The support is fixed at construction:
# a term dropped by the zero prune has no program, and a nonzero value for it is a loud
# error naming `keep_zero_terms`, never a silent no-op. (3) A swap reproduces a fresh
# build of the swapped coefficients — the actual claim the strain move rests on.

using Test
using SLCEMonteCarlo
using SLCE
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :_chain_terms) || include("fixtures.jl")

# Every program array of a Hamiltonian, so a comparison cannot silently miss a field
# added later: the weight streams AND the index arrays.
function _prog_arrays(H)
    pr = H.progs
    return (pr.site_prog, pr.sprog_ptr, pr.sent_w, pr.sent_base, pr.sent_term,
            pr.sent_tgt, pr.sfac_ptr, pr.sfac_row, pr.sfac_slot, pr.site_col,
            pr.site_col2, pr.pent_row, pr.pent_row2, pr.eprog_ptr, pr.eent_w,
            pr.efac_ptr, pr.efac_row, pr.efac_site, pr.term_coef)
end

_rand_cfg(n, seed) = MC.SpinConfig([SVector{3,Float64}(normalize(randn(MersenneTwister(
    seed + s), 3))) for s = 1:n])

@testset "set_coefficients!: no-op round trip is byte-identical" begin
    terms = _chain_terms(0.7)
    H = TiledHamiltonian(1, terms; dims = (4, 1, 1))
    fresh = TiledHamiltonian(1, terms; dims = (4, 1, 1))
    config = _rand_cfg(H.n_sites, 3)
    E0 = total_energy(H, config)

    @test H.n_input_terms == length(terms)
    @test H.term_source == 1:length(terms)
    @test set_coefficients!(H, [t.coef for t in terms]) === H
    for (a, b) in zip(_prog_arrays(H), _prog_arrays(fresh))
        @test a == b                       # byte-identical, not approximately equal
    end
    @test [t.coef for t in H.terms] == [t.coef for t in fresh.terms]
    @test total_energy(H, config) === E0      # and the energy did not move by an ulp

    # the factored stream is an invariant, not an implementation detail
    pr = H.progs
    @test all(i -> pr.sent_w[i] == pr.term_coef[pr.sent_term[i]] * pr.sent_base[i],
              eachindex(pr.sent_w))
end

@testset "set_coefficients!: a swap reproduces a fresh build" begin
    terms = _threebody_terms(0.9)                       # triplet fast path
    H = TiledHamiltonian(1, terms; dims = (4, 1, 1))
    config = _rand_cfg(H.n_sites, 11)
    new_coefs = [-0.35]
    swapped = [SpinMultipoleTerm(new_coefs[k], t.body, t.atoms, t.shifts, t.ls, t.folded)
               for (k, t) in enumerate(terms)]
    fresh = TiledHamiltonian(1, swapped; dims = (4, 1, 1))
    set_coefficients!(H, new_coefs)
    for (a, b) in zip(_prog_arrays(H), _prog_arrays(fresh))
        @test a == b
    end
    @test total_energy(H, config) === total_energy(fresh, config)

    # the energy is linear in the coefficients (j0 is excluded everywhere), so a
    # uniform rescale is an independent check that the SCALE was re-applied once and
    # not re-derived or dropped
    E1 = total_energy(H, config)
    set_coefficients!(H, 2 .* new_coefs)
    @test total_energy(H, config) ≈ 2 * E1 rtol = 1e-13
end

@testset "set_coefficients!: the support is fixed at construction" begin
    terms = [_chain_terms(0.7); _threebody_terms(0.0)]  # the 3-body term is a zero
    H = TiledHamiltonian(1, terms; dims = (4, 1, 1))
    @test length(H.terms) == 2                          # the zero was pruned ...
    @test H.n_input_terms == 3                          # ... but is still indexable
    @test H.term_source == [1, 2]

    # a zero for the pruned term is fine; a nonzero one is a loud, named error
    @test set_coefficients!(H, [0.5, 0.5, 0.0]) === H
    err = try
        set_coefficients!(H, [0.5, 0.5, 0.3])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("keep_zero_terms", err.msg)
    @test occursin("3", err.msg)                        # names the offending index

    # ... and with the escape hatch it works, matching a fresh build of the same list
    Hk = TiledHamiltonian(1, terms; dims = (4, 1, 1), keep_zero_terms = true)
    @test length(Hk.terms) == 3
    @test Hk.term_source == 1:3
    live = [_chain_terms(0.5); _threebody_terms(0.3)]
    fresh = TiledHamiltonian(1, live; dims = (4, 1, 1))
    set_coefficients!(Hk, [t.coef for t in live])
    config = _rand_cfg(Hk.n_sites, 5)
    @test total_energy(Hk, config) ≈ total_energy(fresh, config) rtol = 1e-13
end

@testset "set_coefficients!: keep_zero_terms is byte-neutral without zeros" begin
    # Nothing to prune ⇒ the two arms must produce the same Hamiltonian, so the flag
    # cannot quietly change the physics of an ordinary build.
    terms = _chain_terms(0.7)
    a = TiledHamiltonian(1, terms; dims = (4, 1, 1))
    b = TiledHamiltonian(1, terms; dims = (4, 1, 1), keep_zero_terms = true)
    for (x, y) in zip(_prog_arrays(a), _prog_arrays(b))
        @test x == y
    end
    @test a.n_active == b.n_active && a.n_colors == b.n_colors
end

@testset "set_coefficients!: error surface" begin
    terms = _chain_terms(0.7)
    H = TiledHamiltonian(1, terms; dims = (4, 1, 1))
    @test_throws DimensionMismatch set_coefficients!(H, [1.0])
    @test_throws DimensionMismatch set_coefficients!(H, zeros(5))
    @test_throws ArgumentError set_coefficients!(H, [NaN, 1.0])
    @test_throws ArgumentError set_coefficients!(H, [Inf, 1.0])
end

@testset "set_coefficients!: the flatness verdict is re-checked, not inherited" begin
    model, _ = _joint_model(5)
    layout = SLCE.row_layout(model)
    dterms = SLCE.decorated_terms(model)
    H = TiledHamiltonian(SLCE.n_atoms(model), dterms, layout; dims = (2, 2, 2),
                         keep_zero_terms = true)
    @test H.n_disp_comps > 0 && H.translation_invariant
    coefs = [t.coef for t in dterms]

    # the identity swap keeps the verdict, and re-checking is the default
    @test set_coefficients!(H, coefs) === H

    # a coefficient set that is NOT in the ASR null space breaks the rigid-shift
    # symmetry the sampler re-centres along — and must be refused rather than
    # silently accepted
    bad = copy(coefs)
    bad[findfirst(t -> any(s -> s.factor.channel != SLCE.SPIN, t.slots), dterms)] += 0.5
    err = try
        set_coefficients!(H, bad)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("flatness verdict", err.msg)
    # the opt-out exists for the hot path and does not throw (the Hamiltonian is
    # already holding `bad` from the refused call above — the message says so)
    @test set_coefficients!(H, bad; recheck_translation = false) === H
    @test set_coefficients!(H, coefs) === H              # and the good set restores it
end
