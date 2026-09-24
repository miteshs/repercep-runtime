PY := $(CURDIR)/.venv/bin/python
# Resolve uv from PATH (handles both ~/.local/bin/uv from the official
# installer and /usr/bin/uv from a system package). Falls back to the
# common installer path if `which` fails.
UV := $(shell command -v uv || echo $(HOME)/.local/bin/uv)
CARGO := cargo

.PHONY: help install lint format typecheck test check-gpu info \
        rust-build rust-check rust-fmt rust-fmt-check rust-clippy rust-test \
        rust-install kernels-cpu kernels-cpu-bf16 kernels-cpu-int8 \
        kernels-cpu-fp16 lint-all check-all

help:
	@echo "Repercep Runtime — make targets:"
	@echo ""
	@echo "  Python:"
	@echo "    install         install the package + dev/model/serving extras into .venv"
	@echo "    lint            ruff lint"
	@echo "    format          ruff format"
	@echo "    typecheck       mypy --strict"
	@echo "    test            pytest"
	@echo "    check-gpu       standalone MI300X/ROCm smoke test"
	@echo "    info            print detected backend + devices"
	@echo ""
	@echo "  Rust workspace (cache/scheduler/router crates — see Cargo.toml and ADR-0004):"
	@echo "    rust-build      cargo build --workspace"
	@echo "    rust-check      cargo check --workspace"
	@echo "    rust-fmt        cargo fmt --all"
	@echo "    rust-fmt-check  cargo fmt --all -- --check"
	@echo "    rust-clippy     cargo clippy --workspace --all-targets -- -D warnings"
	@echo "    rust-test       cargo test --workspace"
	@echo "    rust-install    maturin develop --release for every crate (into .venv)"
	@echo ""
	@echo "  Combined:"
	@echo "    lint-all        ruff + rust-fmt-check + rust-clippy"
	@echo "    check-all       lint-all + typecheck + test + rust-test"

install:
	$(UV) pip install --python .venv -e ".[models,serving,dev]"

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest -q

check-gpu:
	$(PY) scripts/check_gpu.py

info:
	$(PY) -m repercep.cli info

# Rust targets. Operate on the virtual workspace at the repo root. No-op cleanly
# while crates/ is empty; ready for the first crate when the fork is resolved
# (see docs/adr/0004-polyglot-build-tooling.md).
#
# Guard: cargo build/check/fmt/clippy/test all error on a zero-member workspace.
# The workspace has three crates as of ADR-0004, but this stays defensive
# (e.g. a shallow/partial checkout missing crates/) rather than assuming
# they're always present.
HAVE_CRATES := $(shell find crates -mindepth 2 -maxdepth 2 -name Cargo.toml -print -quit 2>/dev/null)

define rust-skip-msg
	@echo "$(1): no crates found under crates/ (see docs/adr/0004) — skipping"
endef

rust-build:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-build)
else
	$(CARGO) build --workspace
endif

rust-check:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-check)
else
	$(CARGO) check --workspace
endif

rust-fmt:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-fmt)
else
	$(CARGO) fmt --all
endif

rust-fmt-check:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-fmt-check)
else
	$(CARGO) fmt --all -- --check
endif

rust-clippy:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-clippy)
else
	$(CARGO) clippy --workspace --all-targets -- -D warnings
endif

rust-test:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-test)
else
	$(CARGO) test --workspace
endif

# Build each crate's PyO3 extension into a wheel, then install all wheels into
# .venv via uv. We deliberately do NOT use `maturin develop`: that command
# requires pip in the venv, but our venv is uv-managed (no pip), and
# `--uv` in maturin 1.13 doesn't reliably handle workspace-member crates.
# Build + install is two steps but more robust and CI-friendly.
rust-install:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-install)
else
	@rm -f target/wheels/repercep_*.whl
	@for crate in $$(find crates -mindepth 2 -maxdepth 2 -name Cargo.toml -printf '%h\n'); do \
	    echo "==> maturin build --release in $$crate"; \
	    $(PY) -m maturin build --release --manifest-path $$crate/Cargo.toml || exit 1; \
	done
	@echo "==> uv pip install --reinstall <built wheels>"
	@$(UV) pip install --python .venv --reinstall target/wheels/repercep_*.whl
endif

# Build the CPU AMX flash-attention kernels (Intel Sapphire Rapids+).
# Three siblings live under kernels/cpu/, each with its own /proc/cpuinfo
# gate so a host that lacks the relevant ISA flag skips cleanly rather than
# failing the build.  See docs/adr/0007-cpu-backend.md.
#
#   amx_attn        — AMX_BF16 (Sapphire/Emerald/Granite Rapids)
#   amx_int8_attn   — AMX_INT8 (Sapphire/Emerald/Granite Rapids)
#   amx_fp16_attn   — AMX_FP16 (Granite Rapids ONLY)
kernels-cpu: kernels-cpu-bf16 kernels-cpu-int8 kernels-cpu-fp16

kernels-cpu-bf16:
	@if [ ! -d kernels/cpu/amx_attn ]; then \
	    echo "kernels/cpu/amx_attn missing — skipping AMX BF16 build"; \
	elif [ "$$REPERCEP_AMX_FORCE_BUILD" = "1" ]; then \
	    echo "==> REPERCEP_AMX_FORCE_BUILD=1 — compile-only smoke for AMX BF16 kernel"; \
	    cd kernels/cpu/amx_attn && $(PY) setup.py build_ext --inplace; \
	elif ! grep -q amx_bf16 /proc/cpuinfo 2>/dev/null; then \
	    echo "==> CPU lacks amx_bf16; AMX BF16 kernel build skipped"; \
	else \
	    echo "==> building CPU AMX BF16 flash kernel in kernels/cpu/amx_attn/"; \
	    cd kernels/cpu/amx_attn && $(PY) setup.py build_ext --inplace; \
	fi

kernels-cpu-int8:
	@if [ ! -d kernels/cpu/amx_int8_attn ]; then \
	    echo "kernels/cpu/amx_int8_attn missing — skipping AMX INT8 build"; \
	elif [ "$$REPERCEP_AMX_FORCE_BUILD" = "1" ]; then \
	    echo "==> REPERCEP_AMX_FORCE_BUILD=1 — compile-only smoke for AMX INT8 kernel"; \
	    cd kernels/cpu/amx_int8_attn && $(PY) setup.py build_ext --inplace; \
	elif ! grep -q amx_int8 /proc/cpuinfo 2>/dev/null; then \
	    echo "==> CPU lacks amx_int8; AMX INT8 kernel build skipped"; \
	else \
	    echo "==> building CPU AMX INT8 flash kernel in kernels/cpu/amx_int8_attn/"; \
	    cd kernels/cpu/amx_int8_attn && $(PY) setup.py build_ext --inplace; \
	fi

kernels-cpu-fp16:
	@if [ ! -d kernels/cpu/amx_fp16_attn ]; then \
	    echo "kernels/cpu/amx_fp16_attn missing — skipping AMX FP16 build"; \
	elif [ "$$REPERCEP_AMX_FORCE_BUILD" = "1" ]; then \
	    echo "==> REPERCEP_AMX_FORCE_BUILD=1 — compile-only smoke for AMX FP16 kernel"; \
	    cd kernels/cpu/amx_fp16_attn && $(PY) setup.py build_ext --inplace; \
	elif ! grep -q amx_fp16 /proc/cpuinfo 2>/dev/null; then \
	    echo "==> CPU lacks amx_fp16 (Granite Rapids+); AMX FP16 kernel build skipped"; \
	else \
	    echo "==> building CPU AMX FP16 flash kernel in kernels/cpu/amx_fp16_attn/"; \
	    cd kernels/cpu/amx_fp16_attn && $(PY) setup.py build_ext --inplace; \
	fi

lint-all: lint rust-fmt-check rust-clippy

check-all: lint-all typecheck test rust-test
