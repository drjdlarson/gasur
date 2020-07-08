# gasur
A python package for Guidance, navigation, and control (GNC) of Autonomous Swarms Using Random finite sets (RFS) developed by the Laboratory for Autonomy GNC and Estimation Research (LAGER) at the University of Alabama (UA).

## Using git with submodules
To clone a repo with sub modules use
`git clone --recursive [URL to Git repo]`

To pull new changes for all submodules and new changes in the base repo use
`git pull --recurse-submodules`

To just pull changes from all submodules use
`git submodule update --remote`

You can `cd` into individual submodules and use git commands as if you were inside that repo.

## Unit testing
Unit tests make use **pytest** for the test runner, and tox for automation. The test scripts are located within the **test/** sub-directory.
The tests can be run through a command line with python 3 and tox installed by running
`tox -e test`

Specific tests can be run by passing keywords such as
`tox -e test -- -k guidance`

To run tests marked as slow, pass the `--runslow` option,
`tox -e test -- --runslow`

## Building documentation
The documentation uses sphinx and autodoc to pull docstrings from the code. This process is run through a command line that has python 3 and tox installed. The built documentation is in the **docs/build/** sub-directory.
The HTML version of the docs can be built using the following command:
`tox -e docs -- html`

Then they can be viewed by opening **docs/build/html/index.html**
