======================================
WallGo documentation
======================================

WallGo is an open source code for the computation of the bubble wall velocity and bubble wall width in first-order cosmological phase transitions. If you use WallGo, please cite the WallGo paper `JHEP 04 (2025) 101 <https://doi.org/10.1007/JHEP04(2025)101>`_. :footcite:p:`Ekstedt:2024fyq`

As the universe cooled after the Hot Big Bang, it may have gone through any number of cosmological first-order phase transitions. Such transitions proceed via the nucleation and growth of bubbles, as shown in the image below. :footcite:p:`Weir_2016` The collisions of these bubbles may lead to an observable gravitational wave signal today, depending on the speed of the bubble walls as they collide.

.. image:: figures/weir-bubbles1.jpeg
    :width: 400
    :align: center
    :alt: Bubbles growing in a cosmological phase transition.

|

How fast does the bubble wall go? WallGo is a Python package for answering this question. Our methods largely follow previous work by Laurent and Cline. :footcite:p:`Laurent:2022jrs`

**********
References
**********

.. footbibliography::


.. toctree::
    :caption: Getting Started:
    :hidden:

    installation
    firstExample
    examples
    faqs


.. toctree::
    :caption: Development:
    :hidden:

    contact
    changelog
    development
    license

.. toctree::
    :caption: API reference:
    :hidden:

    WallGo API <_autosummary/WallGo>

