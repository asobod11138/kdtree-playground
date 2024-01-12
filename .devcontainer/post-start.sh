#!/bin/bash

if [ -f pyproject.toml ]; then
    poetry config virtualenvs.in-project true
    poetry install
fi
