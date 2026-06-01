.DEFAULT_GOAL := help

PRESENTATION_DIR := presentation
RUNNER := $(PRESENTATION_DIR)/scripts/run-presentation.mjs

VENV_DIR ?= .venv
ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
SETUP_PYTHON ?= python
else
VENV_PYTHON := $(VENV_DIR)/bin/python
SETUP_PYTHON ?= python3
endif

FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173
BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
REMOTE_HOST ?= 0.0.0.0
REMOTE_PORT ?= 4174
VITE_REMOTE_WS_URL ?=

RUNNER_ARGS := --venv-dir "$(VENV_DIR)" --python "$(SETUP_PYTHON)" --frontend-host "$(FRONTEND_HOST)" --frontend-port "$(FRONTEND_PORT)" --backend-host "$(BACKEND_HOST)" --backend-port "$(BACKEND_PORT)" --remote-host "$(REMOTE_HOST)" --remote-port "$(REMOTE_PORT)" --vite-remote-ws-url "$(VITE_REMOTE_WS_URL)"

.PHONY: help setup examples frontend backend laser remote run full presentation

help:
	@node "$(RUNNER)" help $(RUNNER_ARGS)

setup:
	@node "$(RUNNER)" setup $(RUNNER_ARGS)

examples:
	@node "$(RUNNER)" examples $(RUNNER_ARGS)

backend:
	@node "$(RUNNER)" backend $(RUNNER_ARGS)

frontend:
	@node "$(RUNNER)" frontend $(RUNNER_ARGS)

laser:
	@node "$(RUNNER)" laser $(RUNNER_ARGS)

remote:
	@node "$(RUNNER)" remote $(RUNNER_ARGS)

run:
	@node "$(RUNNER)" run $(RUNNER_ARGS)

full:
	@node "$(RUNNER)" full $(RUNNER_ARGS)

presentation:
	@node "$(RUNNER)" presentation $(RUNNER_ARGS)
