.. raw:: html

   <div style="text-align:center; margin-top: 1.5rem; margin-bottom: 2rem;">
     <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
       Stochastic Interpolants
     </h1>
   </div>

================================================================================

*Stix* is a JAX library for building, training, and sampling from
generative models based on **stochastic interpolants**.

It implements the framework introduced in
*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*
by Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden
(`JMLR 2025 <https://www.jmlr.org/papers/v26/23-1605.html>`_), and shows how a single,
minimal setup can express and unify a wide class of generative models —
including **Flow Matching**, **EDM-style diffusion**, Uniform- and Masked-diffusion, and
**Bayesian Flow Networks (BFNs)**.

🤔 Why stix?
----------------------------------------

**A framework for nearly every interpolation scheme:**
Flow matching, diffusion (e.g. VE and VP), Bayesian Flow Networks,
and masked and uniform discrete diffusion all under a single framework.
Adding your own is incredibly straightforward.

**Discrete and continuous modalities in one framework, by abstracting over the**
:class:`~stix.core.generator.Generator`\ **:** Build multimodal models with both
discrete and continuous data easily. Truly discrete diffusion runs as a
continuous-time Markov chain and continuous data as an ODE or SDE, yet both
are sampled simultaneously from the same network call.

**Sampling decoupled from training:** On one-sided paths, target, noise,
velocity and score convert in closed form, so a velocity-trained model samples
as either an ODE or an SDE with no second head and no retraining.

**Different options for handling discrete data:**
We offer mask and uniform diffusion, as well as methods to learn continuous embeddings of
discrete data to be used with continuous interpolants.

🛠️ Installation
----------------------------------------

.. code-block:: bash

    pip install stix-ml

See :doc:`Installation <installation/index>` for GPU/TPU extras and the
development setup.

📐 Introduction
----------------------------------------

The :doc:`Introduction <introduction>` maps the core mathematical objects of
stochastic interpolants and discrete flow matching onto the corresponding
classes in ``stix``.

📚 Tutorials
----------------------------------------

A set of Jupyter notebooks that cover the basics of the library so you can get
started quickly with a broad overview of its capabilities.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Training & sampling
      :link: https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/1.training_and_sampling.ipynb
      :link-type: url

      Build, train, and sample from a model end-to-end.

   .. grid-item-card:: Multimodal data loading
      :link: https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/2.grain_multimodal_dataloading.ipynb
      :link-type: url

      Feed your own data into ``stix`` with ``grain``.

   .. grid-item-card:: Generative model: loss and prediction
      :link: https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/3.generative_model.ipynb
      :link-type: url

      Define custom losses and the sample-time prediction functions.

   .. grid-item-card:: Conditioning and guidance
      :link: https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb
      :link-type: url

      Condition your samples: context, intrinsic guidance, and custom guidance recipes.

   .. grid-item-card:: Coupling
      :link: https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/5.coupling.ipynb
      :link-type: url

      Pair source and target samples to cut the number of sampling steps.

   .. grid-item-card:: Discrete models
      :link: https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/6.discrete_models.ipynb
      :link-type: url

      Masked and uniform diffusion, and joint continuous/discrete models.

See the :doc:`Tutorials <tutorials/index>` page for the full list.

🔗 API reference
----------------------------------------

The API reference documents the public functions, modules, and objects available
in ``stix``, along with descriptions of their purpose and operation.

See the :doc:`API reference <api_reference/index>` for the full documentation.

.. toctree::
   :caption: Documentation
   :hidden:

   Introduction <introduction>
   Installation <installation/index>
   Tutorials <tutorials/index>

.. toctree::
   :caption: API reference
   :hidden:

   Overview <api_reference/index>
