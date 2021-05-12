SHELL := /usr/bin/env bash
IMAGE := pynguin
VERSION=$(shell git rev-parse --short HEAD)

ifeq ($(STRICT), 1)
	POETRY_COMMAND_FLAG =
	PIP_COMMAND_FLAG =
	SAFETY_COMMAND_FLAG =
	BANDIT_COMMAND_FLAG =
	SECRETS_COMMAND_FLAG =
	BLACK_COMMAND_FLAG =
	DARGLINT_COMMAND_FLAG =
	ISORT_COMMAND_FLAG =
	MYPY_COMMAND_FLAG =
else
	POETRY_COMMAND_FLAG = -
	PIP_COMMAND_FLAG = -
	SAFETY_COMMAND_FLAG = -
	BANDIT_COMMAND_FLAG = -
	SECRETS_COMMAND_FLAG = -
	BLACK_COMMAND_FLAG = -
	DARGLINT_COMMAND_FLAG = -
	ISORT_COMMAND_FLAG = -
	MYPY_COMMAND_FLAG = -
endif

ifeq ($(POETRY_STRICT), 1)
	POETRY_COMMAND_FLAG =
else ifeq ($(POETRY_STRICT), 0)
	POETRY_COMMAND_FLAG = -
endif

ifeq ($(PIP_STRICT), 1)
	PIP_COMMAND_FLAG =
else ifeq ($(PIP_STRICT), 0)
	PIP_COMMAND_FLAG = -
endif

ifeq ($(SAFETY_STRICT), 1)
	SAFETY_COMMAND_FLAG =
else ifeq ($(SAFETY_STRICT), 0)
	SAFETY_COMMAND_FLAG = -
endif

ifeq ($(BANDIT_STRICT), 1)
	BANDIT_COMMAND_FLAG =
else ifeq ($(BANDIT_STRICT), 0)
	BANDIT_COMMAND_FLAG = -
endif

ifeq ($(SECRETS_STRICT), 1)
	SECRETS_COMMAND_FLAG =
else ifeq ($(SECRETS_STRICT), 0)
	SECRETS_COMMAND_FLAG = -
endif

ifeq ($(BLACK_STRICT), 1)
	BLACK_COMMAND_FLAG =
else ifeq ($(BLACK_STRICT), 0)
	BLACK_COMMAND_FLAG = -
endif

ifeq ($(DARGLINT_STRICT), 1)
	DARGLINT_COMMAND_FLAG =
else ifeq ($(DARGLINT_STRICT), 0)
	DARGLINT_COMMAND_FLAG = -
endif

ifeq ($(ISORT_STRICT), 1)
	ISORT_COMMAND_FLAG =
else ifeq ($(ISORT_STRICT), 0)
	ISORT_COMMAND_FLAG = -
endif

ifeq ($(MYPY_STRICT), 1)
	MYPY_COMMAND_FLAG =
else ifeq ($(MYPY_STRICT), 0)
	MYPY_COMMAND_FLAG = -
endif


.PHONY: download-poetry
download-poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

.PHONY: install
install:
	poetry lock -n
	poetry install -n
ifneq ($(NO_PRE_COMMIT), 1)
	poetry run pre-commit install
endif

.PHONY: check-safety
check-safety:
	$(POETRY_COMMAND_FLAG)poetry check
	$(PIP_COMMAND_FLAG)pip check
	$(SAFETY_COMMAND_FLAG)poetry run safety check --full-report
	$(BANDIT_COMMAND_FLAG)poetry run bandit -ll -r pynguin

.PHONY: check-style
check-style:
	$(BLACK_COMMAND_FLAG)poetry run black --diff --check ./
	$(ISORT_COMMAND_FLAG)poetry run isort --check-only . --profile black
	$(MYPY_COMMAND_FLAG)poetry run mypy software_analysis_examples

.PHONY: codestyle
codestyle:
	poetry run pre-commit run --all-files

.PHONY: test
test:
	poetry run pytest --cov=software_analysis_examples --cov=tests --cov-branch --cov-report=term-missing --cov-report html:cov_html tests/

.PHONY: mypy
mypy:
	poetry run mypy software_analysis_examples

.PHONY: flake8
flake8:
	poetry run flake8 .

.PHONY: isort
isort:
	poetry run isort . --profile black

.PHONY: black
black:
	poetry run black .

.PHONY: check
check: isort black mypy flake8 test

.PHONY: lint
lint: test check-safety check-style

.PHONY: documentation
documentation:
	poetry run sphinx-build docs docs/_build

.PHONY: clean_build
clean_build:
	rm -rf build/
	rm -rf .hypothesis
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf cov_html
	rm -rf dist
	rm -rf docs/_build
	find . -name pynguin-report -type d | xargs rm -rf {};
	find . -name ".coverage*" -type f | xargs rm -rf {};

.PHONY: clean
clean: clean_build
