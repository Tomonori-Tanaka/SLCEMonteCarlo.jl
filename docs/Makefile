# Documentation build for SLCEMonteCarlo.jl.
#
#   make build        build the HTML site into build/
#   make serve        build, then serve it locally with live reload (LiveServer.jl)
#   make open         open the built site in a browser
#   make clean        remove the build/ output
#
# The first run resolves docs/Project.toml (Documenter + LiveServer + SLCEMonteCarlo +
# SLCE, both by path). SLCE must sit at ../../SLCE.jl.

.PHONY: build serve open clean help

# -t 4: the executed PT examples in the guide sweep their lanes over threads
# (results are bit-identical regardless of the thread count).
build:
	julia -t 4 --project=. -e 'using Pkg; Pkg.instantiate(); include("make.jl")'

serve: build
	julia --project=. -e 'using LiveServer; serve(dir = "build")'

open:
	open build/index.html

clean:
	rm -rf build/

help:
	@echo "build  - build the HTML documentation into build/"
	@echo "serve  - build and serve locally (live reload)"
	@echo "open   - open the built site in a browser"
	@echo "clean  - remove build/"
