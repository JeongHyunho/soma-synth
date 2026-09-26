"""Marks the test tree as a regular package.

A namespace package loses to any regular package of the same name found anywhere on
``sys.path``; being a regular package, plus the repository root on ``pythonpath`` (see
``pyproject.toml``), makes this tree win by path order instead of an unrelated installed
``tests`` package.
"""
