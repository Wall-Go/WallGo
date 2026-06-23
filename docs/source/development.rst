======================================
Development version
======================================

WallGo is under active development, with plans for future versions with
additional functionality. The latest developments are available from our
`git repository`_.

.. _git repository: https://github.com/Wall-Go/WallGo

To install the development version, together with the requirements for building the docs, running the tests and linting, run the following::

    git clone https://github.com/Wall-Go/WallGo.git
    cd WallGo
    pip install -e ".[docs,tests,lint]"


Tests can then be run with::

    pytest -v

The `.hdf5` collision files packaged with WallGo are tracked with `git-lfs <https://git-lfs.com/>`_, so that to download them, you also need to run::

    git lfs install
    git lfs pull

This only needs to be done once.

|
