# gasur
A python package for Guidance, navigation, and control (GNC) of Autonomous Swarms Using Random finite sets (RFS) developed by the Laboratory for Autonomy GNC and Estimation Research (LAGER) at the University of Alabama (UA).

## Setup
This package requires python 3 and tox for running tests and building documentation. Once python 3 is installed tox can be installed by running `pip3 install tox`

### Using git with submodules
To clone a repo with sub modules use
`git clone --recursive [URL to Git repo]`

To pull new changes for all submodules and new changes in the base repo use
`git pull --recurse-submodules`

To just pull changes from all submodules use
`git submodule update --remote`

You can `cd` into individual submodules and use git commands as if you were inside that repo. More information can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules) .

### Importing into Spyder
The Spyder/Anaconda can be setup to install the local versions of the package so standalone scripts can be written as if the package were installed in the typical fashion (via pip install). Details on this process are included in the documentation.

## Testing
Unit and validation tests make use **pytest** for the test runner, and tox for automation. The test scripts are located within the **test/** sub-directory.
The tests can be run through a command line with python 3 and tox installed. There are 3 different environments for running tests. One for unit tests, another for validation tests, and a general purpose one  that accepts any arguments to pytest.
The general purpose environment is executed by running

`tox -e test -- PY_TEST_ARGS`

where PY_TEST_ARGS are any arguments to be passed directly to the pytest command (Note: if none are passed the `--` is not needed).
For example to run any test cases containing a keyword, run the following,

`tox -e test -- -k guidance`

To run tests marked as slow, pass the `--runslow` option.

The unit test envrionment runs all tests within the **test/unit/** sub-directory. These tests are designed to confirm basic functionality.
Many of them do not ensure algorithm performance but may do some basic checking of a few key parameters. This environment is run by

`tox - e unit_test`

The validation test envrionment runs all tests within the **test/validation/** sub-directory. These are designed to verify algorithm performance and include more extensive checking of the output arguments against known values. They often run slower than unit tests.
These can be run with

`tox - e validation_test`

## Building documentation
The documentation uses sphinx and autodoc to pull docstrings from the code. This process is run through a command line that has python 3 and tox installed. The built documentation is in the **docs/build/** sub-directory.
The HTML version of the docs can be built using the following command:
`tox -e docs -- html`

Then they can be viewed by opening **docs/build/html/index.html**


# Cite
Please cite the framework as follows

```
@Misc{gasur,
  author       = {Jordan D. Larson and Ryan W. Thomas and Vaughn Weirens},
  howpublished = {Web page},
  title        = {{GASUR}: A {P}ython library for {G}uidance, navigation, and control of {A}utonomous {S}warms {U}sing {R}andom finite sets},
  year         = {2019},
  url          = {https://github.com/drjdlarson/gasur},
}
```
