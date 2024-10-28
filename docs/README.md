# WallGo documentation

To make the html version of the docs, one first needs to install the
requirements. To install Sphinx, see the following.

Linux:

    sudo apt-get install python3-sphinx

MacOS:

    brew install python-sphinx


Next, to install the necessary Python libraries, from the root directory of the
repository, run

    pip install -e '.[docs]'

Then, make the html version of the docs with::

    cd docs
    make html

These are then installed under

    docs/_build/html

Look for the index.html file to start with.
