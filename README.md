# mc3g
Monte Carlo simulation of 3G rules

This project contains the Python code to do simulations of events according to the 3G rule (in German: "Geimpft, Genesen, Getestet").
It also contains the code to run an interactive session using [streamlit](https://streamlit.io/). The Monte Carlo code makes use of [numpy](https://numpy.org/) and [numba](https://numba.pydata.org/).

## Setup
It is advised to setup a virtualenv to install all dependencies. To create a virtualenv, type

    python3 -m venv env

To enable the virtualenv, type

    . env/bin/activate

Now you can install all dependencies:

    pip install -r requirements.txt

You can deactivate the virtualenv by typing

    deactivate


## Usage
In the virtualenv, you can either edit and run 

    main.py

to create plots, or you can run 

    streamlit run app.py

to create a small webserver and run an interactive session.
