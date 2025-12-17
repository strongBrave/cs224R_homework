## Install dependencies

There are two options:

A. (Recommended) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

2. Create a conda environment that will contain python 3:
	```
	conda create -n cs224r python=3.11
	```
3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	conda activate cs224r
	```
4. Install the requirements into this conda environment
	```
	pip install -r requirements.txt
	```
5. Allow your code to be able to see 'cs224r'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python:
	```
	pip install -r requirements.txt
	```

## Troubleshooting 

If you are encountering errors in `pip install -r requirements.txt` involving `Error building wheels...`, you may need to manually install conda dependencies. For example, if there is a problem with swig install, you might want to do `conda install -c conda-forge swig` to resolve this. If you can't find the version number, make sure you are using the correct version of Python (3.11, as above).

You may encounter the following GLFW errors if running on machine without a display:

GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
GLFWError: (65537) b'The GLFW library is not initialized'

These can be resolved with:
```
export MUJOCO_GL=egl
```
