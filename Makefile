PYTHON=python
PKG=bench

prepare:
	$(PYTHON) -m $(PKG).cli action=prepare dataset=$(DATASET)

train:
	$(PYTHON) -m $(PKG).cli action=train dataset=$(DATASET) model=$(MODEL) $(EXTRA)

eval:
	$(PYTHON) -m $(PKG).cli action=eval dataset=$(DATASET) model=$(MODEL) $(EXTRA)

export:
	$(PYTHON) -m $(PKG).cli action=export dataset=$(DATASET) model=$(MODEL) $(EXTRA)

profile:
	$(PYTHON) -m $(PKG).cli action=profile dataset=$(DATASET) model=$(MODEL) $(EXTRA)

smoke:
	$(PYTHON) -m $(PKG).cli action=prepare dataset=ims
	$(PYTHON) -m $(PKG).cli action=train dataset=ims model=elm model.hidden_width=128
	$(PYTHON) -m $(PKG).cli action=eval dataset=ims model=elm

mlflow:
	. .venv/bin/activate && mlflow ui

clean:
	rm -rf outputs/ mlruns/ data/processed/ __pycache__ */__pycache__ *.pyc

.PHONY: prepare train eval export profile smoke mlflow clean

