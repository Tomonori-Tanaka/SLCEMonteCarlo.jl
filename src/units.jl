# Temperature control: the kelvin ↔ model-energy-unit boundary.
#
# `KB_EV` and `resolve_kt` are NOT defined here — they live in `SLCE` (`src/units.jl`),
# the one package every sampler in the family already depends on. They used to be
# duplicated here and in SLCETools' `MetropolisSampler`, character for character; two
# copies of a unit conversion are two things that can drift apart while both suites stay
# green, so the convention is owned upstream and re-exported from here.
#
# What this package keeps is the *policy*: every public entry point takes exactly one of
# two keywords — `temperature` in kelvin or `kT` in the model's energy units. The two
# live under distinct names deliberately: a single keyword serving both units would let
# `temperature = 300` (meant as kelvin) be read as 300 eV — a silent
# infinite-temperature run.

using SLCE: KB_EV, resolve_kt

"""
    GPA_PER_EV_A3

GPa per one eV/Å³: exactly `160.2176634` (from the exact SI elementary charge
`e = 1.602176634e-19` C, since 1 eV/Å³ = `e`·10³⁰ J/m³). The pressure analogue of
`KB_EV`, applied exactly once, at keyword resolution (`pressure_GPa` →
model units); nothing downstream of the resolver ever converts. Defined here for
now — if another family package grows a pressure keyword, move it upstream to
`SLCE.units` exactly as `KB_EV` was.
"""
const GPA_PER_EV_A3 = 160.2176634
