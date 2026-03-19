PYTHON := $(HOME)/anaconda3/bin/python3

.PHONY: test demo install

test:
	$(PYTHON) -m pytest tests/ -v

demo:
	$(PYTHON) examples/burgers_demo.py

install:
	$(PYTHON) -m pip install -e ".[dev]"
