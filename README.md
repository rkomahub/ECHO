# ECHO

**Avoid the Noise, Embrace the ECHO: Eliminating Crosstalk in Highly-Interacting On-Chip Spins**

ECHO is a Python library for simulating the dynamics and control of interacting quantum spin systems.

The library is designed to support progressively more complex physical models, from simple driven-spin systems to interacting NV-center registers.

## Goals

ECHO aims to provide reusable tools to:

- simulate quantum spin systems with configurable local spin dimension;
- model systems containing an arbitrary number of interacting spins;
- construct and combine different Hamiltonian models;
- implement quantum-control and dynamical-decoupling protocols;
- study spin-spin interactions, crosstalk, and noise;
- perform reproducible simulations through a Python API and a command-line interface.

## Installation

ECHO requires Python 3.9 or newer.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the library in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Development philosophy

The project is developed progressively. Simple physical models are used first to validate the numerical and software framework before more complex models are introduced.

The same library is intended to support both educational simulations and thesis-related research on interacting NV-center systems.

## Releases

Stable versions of the library are identified using Git tags and releases.

The version submitted for the Software and Computing exam will be explicitly marked as the exam release. Later versions may introduce additional thesis-related models and functionality.

## Status

Early development.