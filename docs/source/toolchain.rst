Toolchain Setup
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
This assumes you have installed Anaconda Navigator and will be using the Spyder IDE for development. This guide 
will walk you through setting up a virtual environment within Anaconda, installing GASUR and its dependencies, 
and starting Spyder so scripts can use the GASUR package and run from within Spyder's IPython console with its 
debugger.

..
    BEGIN README INCLUDE

.. _GASUR: https://github.com/drjdlarson/gasur
.. _GNCPY: https://github.com/drjdlarson/gncpy
.. _STACKOVERFLOW: https://stackoverflow.com/questions/69704561/cannot-update-spyder-5-1-5-on-new-anaconda-install
.. _SUBMODULE: https://git-scm.com/book/en/v2/Git-Tools-Submodules

Setup
-----
Currently this package is not available via pip, the following subsections outline methods for using this package for writing code that uses gasur and for developing/extending gasur. If you want to do both then follow the instructions for developing/extending gasur. Note that only python 3 is supported. It is recommended to use Anaconda and the Spyder IDE, but this is not necessary.

Installation for only using gasur in other programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This can be run with or without using Anaconda. If using Anaconda, the :code:`--no-dependencies` flag may want to be used to prevent conflicts between pip and conda. The gasur/gncpy dependencies will then need to be manually installed with :code:`conda install`.

1) Download/clone the `gasur`_ repository somewhere on your system (remember the location).
2) Open a terminal that you can run python 3 from and navigate to the repo directory.
3) Run :code:`pip install -e ./gncpy`
4) Run :code:`pip install -e .`

Installation for developing gasur
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The recommended method for development is to use Anaconda and the Spyder IDE, also tox is used for running the built-in tests and generating the documentation. The tox setup assumes the `rm` command works, and that gasur has the gncpy submodule cloned. For recommended setup for git with submodules see the following subsection.

1) Download and install `Anaconda <https://www.anaconda.com/>`_. This should also install Spyder, numpy, matplotlib, and some other common libraries.
2) Open Spyder, if it is not version >= 5.1.5 and you want to update to the latest version close Spyder and run the following commands from a terminal (Anaconda terminal on windows) the second and third commands may give errors but they can be ignored. See here on `stackoverflow`_.

.. code-block:: bash

    conda remove spyder
    conda remove python-language-server
    conda update anaconda
    conda install spyder=5.1.5


3) Download/clone the `gncpy`_ repository and save it somewhere on your system (remember the location).
4) Download/clone the `gasur`_ repository somewhere on your system (remember the location).
5) Open the terminal that has the base Anaconda environment activated (normal terminal for linux, Anaconda terminal on windows).
6) Navigate to the directory where you saved the gncpy repo and run :code:`pip install -e ./gncpy`. Note, Spyder will use this version of gncpy when running code, but the tox tests will use the version contained within the gasur repository. This makes it easier to develop for gncpy in addition to gasur. If this is not needed then there is no need to clone the gncpy repo seperately and the version inside gasur can be pip installed instead.
7) Navigate to the directory containing the repo and run :code:`pip install -e ./gasur`
8) Install tox by running :code:`conda install -c conda-forge tox`

Now you should be able to run tests with tox and from within Spyder, also the Spyder IDE variable explorer should recognize gasur and gncpy data types for debugging. It is also possible to create conda environments and tell Spyder to use that as the interpreter for running code. Note that the proper version of :code:`spyder-kernels` must be installed in the environment but Spyder will tell you the command to run when it fails to start the interpreter. This can be useful if you need additional libraries for certain projects. Information on conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ and setting up Spyder with a custom interpreter can be done through Syder's settings. 

Using git with Submodules
^^^^^^^^^^^^^^^^^^^^^^^^^
To clone a repo with submodules use

.. code-block:: bash

    git clone --recursive [URL to Git repo]

To pull new changes for all submodules and new changes in the base repo use

.. code-block:: bash

    git pull --recurse-submodules

To just pull changes from all submodules use

.. code-block:: bash

    git submodule update --remote

You can :code:`cd` into individual submodules and use git commands as if you were inside that repo. See git's `submodule`_ page for more information.

It is recommended to setup git to handle some submodule commands automatically by running the following commands once.

.. code-block:: bash

    git config --global diff.submodule log
    git config --global status.submodulesummary 1
    git config --global submodule.recurse true



Testing
-------
Unit and validation tests make use of **pytest** for the test runner, and tox for automation. The test scripts are located within the **test/** sub-directory.
The tests can be run through a command line with python 3 and tox installed. If the Spyder setup instructions were followed then the tests can also be run as standalone scripts from within Spyder by uncommenting the appropriate line under the :code:`__main__` section.

There are 3 different environments for running tests. One for unit tests, another for validation tests, and a general purpose one  that accepts any arguments to pytest.
The general purpose environment is executed by running

.. code-block:: bash

    tox -e test -- PY_TEST_ARGS

where `PY_TEST_ARGS` are any arguments to be passed directly to the pytest command (Note: if none are passed the `--` is not needed).
For example to run any test cases containing a keyword, run the following,

.. code-block:: bash

    tox -e test -- -k guidance

To run tests marked as slow, pass the `--runslow` option.

The unit test environment runs all tests within the **test/unit/** sub-directory. These tests are designed to confirm basic functionality.
Many of them do not ensure algorithm performance but may do some basic checking of a few key parameters. This environment is run by

.. code-block:: bash

    tox - e unit_test

The validation test environment runs all tests within the **test/validation/** sub-directory. These are designed to verify algorithm performance and include more extensive checking of the output arguments against known values. They often run slower than unit tests.
These can be run with

.. code-block::

    tox - e validation_test

Building Documentation
----------------------
The documentation uses sphinx and autodoc to pull docstrings from the code. This process is run through a command line that has python 3 and tox installed. The built documentation is in the **docs/build/** sub-directory.
The HTML version of the docs can be built using the following command :code:`tox -e docs -- html`. Then they can be viewed by opening **docs/build/html/index.html** with a web browser.


Notes about tox
---------------
If tox is failing to install the dependencies due to an error in distutils, then it may be required to instal distutils seperately by

.. code-block:: bash

    sudo apt install python3.7-distutils

for a debian based system.
