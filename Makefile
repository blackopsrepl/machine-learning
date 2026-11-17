# Machine Learning Laboratory — pure Ruby, zero dependencies.
RUBY ?= ruby
GROUPS := supervised neural_networks evaluation trees unsupervised recommenders reinforcement_learning
DEMOS := $(foreach group,$(GROUPS),$(sort $(wildcard $(group)/*.rb)))
NAMES := $(notdir $(basename $(DEMOS)))

.DEFAULT_GOAL := help
.PHONY: all help list check $(GROUPS) $(NAMES)
help: list
list:
	@printf '\n── machine learning · one file per algorithm ─────────\n'
	@$(foreach group,$(GROUPS),printf '\n  $(group)\n'; $(foreach file,$(filter $(group)/%,$(DEMOS)),printf '    make %-30s %s\n' '$(basename $(notdir $(file)))' '$(file)';)) true
	@printf '\n  make all       run every demo\n  make check     syntax checks and every executable assertion\n\n'
all: $(NAMES)
$(GROUPS):
	@$(foreach file,$(filter $@/%,$(DEMOS)),$(RUBY) $(file) &&) true

define demo_target
$(basename $(notdir $(1))):
	@$$(RUBY) $(1)
endef
$(foreach file,$(DEMOS),$(eval $(call demo_target,$(file))))

check:
	@for file in lib/*.rb $(DEMOS); do $(RUBY) -wc "$$file" >/dev/null || exit 1; done
	@$(MAKE) --no-print-directory all
