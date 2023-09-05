.PHONY: \
	setup setup-requirements setup-contrib \

default: setup

all: setup

setup: setup-requirements

setup-requirements: 
	python -m venv .venv \
	&& ( \
		source .venv/bin/activate \
		&& \
		pip install --upgrade pip \
		&& \
		pip install -r requirements.txt \
	)


clean: 
	rm -rf .venv

