SBR_ROOT ?= /Users/ramanathana/Work/StructBioReasoner
JNANA_ROOT ?= /Users/ramanathana/Work/Jnana
export PYTHONPATH := .:$(SBR_ROOT)
export BINDER_ADAPTER_SBR_ROOT := $(SBR_ROOT)
export BINDER_ADAPTER_JNANA_ROOT := $(JNANA_ROOT)

.PHONY: test
test:
	python -m pytest tests/

.PHONY: compile
compile:
	python -m compileall -q binder_adapter tests
