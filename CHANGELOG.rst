==================
Changelog
==================

1.1.0 (2025-08-20)
==================

* Phase tracing updates to improve algorithm stability.
* User may now pass the free energy of a phase as arrays.
* Spectral truncation options added to avoid aliasing, with an automatic method used by default.
* Error estimate for the wall speed updated to make it more theoretically sound.
* Linearization criterion updated so that `linearizationCriterion2` estimates the second order correction.
* Error estimate due to the Tanh ansatz now computed and added to `WallGoResults`.
* Accuracy of energy-momentum conservation computed, and printed for logging level `DEBUG`.
* New tests added for the Standard Model with light Higgs.
* `PTTools <https://github.com/CFT-HY/pttools>`_ example file added.


1.0.0 (2024-11-07)
==================

* WallGo released.

0.1.0 (2024-10-08)
==================

* WallGoCollision code migrated to a separate repository.
* Alpha version made public.

0.0.1 (2023-05-18)
==================

* Project initiated at `How fast does the bubble grow? <https://indico.desy.de/event/37126/>`_
