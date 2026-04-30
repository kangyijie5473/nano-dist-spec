#!/usr/bin/env python3
"""Backward-compatible install entry point.

Project metadata and dependencies live in ``pyproject.toml`` (PEP 621). Prefer:

    pip install .
    pip install -e .

Legacy editable installs (setuptools)::

    python setup.py develop
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
