PYTHON ?= python3
VENV ?= .venv
ifeq ($(wildcard $(VENV)/bin/python),)
RUN := $(PYTHON)
PIP := $(PYTHON) -m pip
else
RUN := $(VENV)/bin/python
PIP := $(RUN) -m pip
endif

.PHONY: venv install-pi install-laptop check record reconstruct

venv:
	$(PYTHON) -m venv $(VENV)

install-pi: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-pi.txt

install-laptop: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-laptop.txt

check:
	$(PYTHON) -m py_compile load_config.py utils.py ekf.py record.py reconstruct.py

record:
	$(RUN) record.py

reconstruct:
	$(RUN) reconstruct.py
