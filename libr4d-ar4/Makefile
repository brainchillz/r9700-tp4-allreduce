# R4D kernel library. build.sh is the build; this only wraps it.
#
#     make                              # build r4d.so here (needs hipcc + pybind11 on PATH)
#     make IMAGE=some/rocm-image        # build it inside that container instead
#     make verify IMAGE=...             # import the built r4d.so and list its entry points
#     make clean
GFX_ARCH ?= gfx1201
PYTHON   ?= python3

ifdef IMAGE
RUN = docker run --rm --entrypoint bash -v "$(CURDIR):/work" -w /work $(IMAGE) -c
else
RUN = bash -c
endif

.DEFAULT_GOAL := all
.PHONY: all verify clean

all:
	@$(RUN) 'GFX_ARCH=$(GFX_ARCH) PYTHON=$(PYTHON) ./build.sh'

verify:
	@$(RUN) 'PYTHONPATH=$$PWD $(PYTHON) -c "import torch, r4d; \
	  print(\"r4d\", r4d.__version__); \
	  [print(\"  \", k[\"family\"].ljust(5), k[\"name\"].ljust(34), k[\"shape\"]) for k in r4d.kernels()]"'

clean:
	rm -f r4d.so *.o
