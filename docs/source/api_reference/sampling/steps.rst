.. _api_sampling_steps:

Steps
=====

.. automodule:: stix.sampling.steps
    :no-members:

Generator-type tables
---------------------

.. autodata:: stix.sampling.steps.STEP_FNS

.. autodata:: stix.sampling.steps.DRIFT_FNS

.. autodata:: stix.sampling.steps.DIFFUSION_FNS

Step updates
------------

.. autofunction:: stix.sampling.steps.velocity_euler_step

.. autofunction:: stix.sampling.steps.velocity_and_score_euler_maruyama_step

.. autofunction:: stix.sampling.steps.ctmc_euler_step

Drift and diffusion
-------------------

.. autofunction:: stix.sampling.steps.velocity_drift

.. autofunction:: stix.sampling.steps.velocity_diffusion

.. autofunction:: stix.sampling.steps.velocity_and_score_drift

.. autofunction:: stix.sampling.steps.velocity_and_score_diffusion
