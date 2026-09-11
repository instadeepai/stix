.. _tutorials:

📚 Tutorials
==================

We provide a set of tutorials that cover the basics of the library so that one
can quickly get started with a broad overview of its capabilities.

Each tutorial is a Jupyter notebook hosted in the repository; follow the
*Open the notebook* link to view the rendered version on GitHub.

New to ``stix``? We recommend reading the :doc:`Introduction </introduction>`
first — it maps the core mathematical objects onto the library's API and will
make the tutorials below easier to follow.

Training & sampling
~~~~~~~~~~~~~~~~~~~~~

Build, train, and sample from a stochastic-interpolant model end-to-end. A good
first read to understand how the pieces of ``stix`` fit together.

**What you'll learn**

- How to define an interpolant and wrap it in a generative model.
- How to configure and run the training loop.
- How to draw samples with a solver and visualise the results.

`Open the notebook → <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/1.training_and_sampling.ipynb>`_

Multimodal data loading
~~~~~~~~~~~~~~~~~~~~~~~~~

Feed your own data into ``stix`` with `grain <https://google-grain.readthedocs.io/en/latest/index.html>`_,
a JAX-friendly, checkpointable data-loading library.

**What you'll learn**

- The minimum elements required to wire a data source into the ``stix`` pipeline.
- How to plug in an external dataset and custom synthetic sources.
- How to checkpoint and restore the data stream so an interrupted run resumes
  exactly where it stopped.

`Open the notebook → <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/2.grain_multimodal_dataloading.ipynb>`_

Building generative models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A detailed look at the :class:`~stix.core.gen_model.GenerativeModel`: how the
choice of prediction target defines the loss, and how that same choice determines
the sample-time conversions.

**What you'll learn**

- How to subclass ``GenerativeModel`` to build your own model, worked through
  three examples: a denoiser (predicts the clean data), a noise-prediction model,
  and an advanced hybrid model mixing a velocity target (continuous modalities)
  with a cross-entropy logits target (discrete modalities).
- How the loss fixes what the network predicts, and how to convert that
  prediction into the sample-time generator via ``get_generator``.
- How to reuse the pre-baked one-sided-linear interpolant converters.

`Open the notebook → <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/3.generative_model.ipynb>`_

Conditioning and guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn how to use ``stix``'s conditioning and guidance mechanisms to steer
the generative model at sampling time. Understand both the code and theory
behind the two conditioning channels and how to write your own guidance recipe.


**What you'll learn**

- The difference between the two conditioning channels : **context** (condition the network output
  based on provided context, requires training with context) and **intrinsic** (no specific training required,
  use standard data modality to build surrogate conitional score).
- How to leverage intrinsic conditioning methods to condition on standard learned data modalities, including inpainting.
- How to restore a context-conditional model and amplify it with classifier-free guidance.
- How to write a custom ``GuidanceFn``. In the notebook we go through the process of creating a novel recipe,
  in this case a combination of both intrinsic (inpainting) and classifier-free context guidance.

`Open the notebook → <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_

Coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement :class:`~stix.core.coupling.Coupling` s in ``stix``. A coupling pairs
source and target samples together; choosing one that minimises the energy
required to move between the two distributions can substantially reduce the
number of solver steps needed at sampling time.

**What you'll learn**

- How the default (no coupling: independent pairing) compares to a mini-batch
  optimal-transport coupling.
- How to implement your own mini-batch optimal-transport coupling, worked through
  two examples: on a discrete ``index`` modality and on a continuous ``coordinates`` modality.
- How to couple at the dataset level instead of the batch level, reproducing
  rectified flow.
- Observe the effect these different coupling approaches have on the training time and sampling performance.

`Open the notebook → <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/5.coupling.ipynb>`_

Discrete models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model **discrete** modalities (tokens, categories) as a continuous-time Markov
chain (CTMC), and sample them jointly with the continuous O/SDEs. Covers masked
and uniform diffusion — the two share a single convex *mixture path* and differ
only in their source distribution.

**What you'll learn**

- How masked and uniform diffusion arise as
  :class:`~stix.core.interpolant.MaskDiscreteInterpolant`
  and
  :class:`~stix.core.interpolant.UniformDiscreteInterpolant`.
- How :class:`~stix.core.gen_model.factory.PosteriorMixtureGenerativeModel`
  turns a predicted denoising posterior into a denoising cross-entropy loss and a
  ``TransitionRates`` generator sampled as a CTMC by the
  :class:`~stix.sampling.solver_manual.ManualSolver`.
- How :class:`~stix.core.gen_model.factory.VelocityAndPosteriorGenerativeModel`
  advances a continuous O/SDE and a discrete CTMC jointly in one sampling pass.

`Open the notebook → <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/6.discrete_models.ipynb>`_
