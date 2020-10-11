Toolchain Setup
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
This assumes you have installed Anaconda Navigator and will be using the Spyder IDE for development. This guide 
will walk you through setting up a virtual environment within Anaconda, installing GASUR and its dependencies, 
and starting Spyder so scripts can use the GASUR package and run from within Spyder's IPython console with its 
debugger.


Inital setup
------------

First, open Anaconda Navigator and go to the environments tab. Then create a new environment using python version 3.7.
Name the environment something appropriate, and without spaces. Once this is done setting up you can close Navigator.
Now open Anaconda Prompt, a command line version of the Navigator. This is required for the initial setup and is faster 
for switching between environments than starting Navigator each time you want to launch spyder.
Once Anaconda Prompt is open you can switch to your new environment with the following command

.. code-block:: shell  
   
   conda activate ENV_NAME

or with

.. code-block:: shell  
   
   activate ENV_NAME

where :code:`ENV_NAME` is the name of the environment you created. Next :code:`cd` to the root directory of the GASUR 
repository. Now install the local GNCPy repo with the following command,

.. code-block:: shell  
   
   conda install conda-build
   pip install -e gncpy

This will use whatever files are currently on your system, meaning if you pull changes from the remote repo or switch 
branches your installation will  automatically update. Next install GASUR with

.. code-block:: shell  
   
   pip install -e .

Finally, spyder needs to be installed in this environment. Spyders documentation recommends a different approach 
(`here <https://github.com/spyder-ide/spyder/wiki/Working-with-packages-and-environments-in-Spyder>`_), but that requires extra setup once Spyder starts and its variable explorer will not be able to properly display 
class information for classs within GASUR or GNCPy. Spyder can be installed to this environment with 

.. code-block:: shell  
   
   conda install spyder

Once this is complete, you can start spyder by running

.. code-block:: shell  
   
   spyder

from the Anaconda Prompt terminal. Note, that the terminal must remain open while Spyder is running, if the terminal is 
closed Spyder will also close.


Returning
---------
Once the environment is setup, you can start Spyder in that environment by opening Anaconda Prompt and running 

.. code-block:: shell  
   
   activate ENV_NAME
   spyder

where :code:`ENV_NAME` is the name of the environment you created. If you forget the name, you can see all the environments 
in the Anaconda Navigator.
