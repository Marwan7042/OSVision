PYTHON ?= python3
VENV ?= .venv
ifeq ($(wildcard $(VENV)/bin/python),)
RUN := $(PYTHON)
PIP := $(PYTHON) -m pip
else
RUN := $(VENV)/bin/python
PIP := $(RUN) -m pip
endif

.PHONY: venv install-pi install-laptop check record reconstruct regression regression-score

venv:
	$(PYTHON) -m venv $(VENV)

install-pi: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/requirements-pi.txt

install-laptop: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/requirements-laptop.txt

check:
	$(PYTHON) -m py_compile src/load_config.py src/utils.py src/ekf.py src/record.py src/reconstruct.py

record:
	$(RUN) src/record.py

reconstruct:
	$(RUN) src/reconstruct.py

regression:
	$(RUN) src/regression_harness.py

regression-score:
	$(RUN) src/ regression_harness.py --no-run
