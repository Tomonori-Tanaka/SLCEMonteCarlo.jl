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
