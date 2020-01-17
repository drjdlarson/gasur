#!/bin/bash

LINE="================================================="


# install tox to default python environment to run unit tests
printf '%s\n' "$LINE"
printf "\tInstalling tox for running unit tests\n"
printf '%s\n' "$LINE"
pip install --user tox

#  create virtual environment with workaround for pip failure
printf '%s\n' "$LINE"
printf "\tSetting up python virtual environment for using package\n"
printf '%s\n' "$LINE"

# may need to change to python3 if python doesn't resolve to python 3.7
python -m venv .env --without-pip
source .env/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
deactivate
printf "\n\n"

# install package in development mode so changes are automatically tracked
printf '%s\n' "$LINE"
printf "\t Installing GASUR in development mode\n"
printf '%s\n' "$LINE"
source .env/bin/activate
pip install -e .
deactivate
printf "\n\n"

printf '%s\n' "$LINE"
printf '%s\n' "$LINE"
printf "SUCCESS\n"
printf "use source .env/bin/activate to start virtual environment an run python scripts\n"
printf "use deactivate to close virtual environment\n"
printf "\n\n"
