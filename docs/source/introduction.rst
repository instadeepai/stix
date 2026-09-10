.. _introduction:

Introduction
============

``stix`` is a JAX / Flax NNX library for generative models based on the
**Stochastic Interpolants** (Albergo, Boffi, Vanden-Eijnden,
`JMLR 2025 <https://www.jmlr.org/papers/v26/23-1605.html>`_) and the
**Discrete Flow Matching** (Gat et al.,
`NeurIPS 2024 <https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html>`_)
frameworks. Designed to be flexible and explicitly tailored for multi-modal
applications, it enables the implementation of a wide variety of models,
including flow matching, diffusion, Bayesian flow networks, and discrete flow
matching.

This page introduces the core mathematical concepts on which the library is
built, and maps each of them onto the corresponding objects in ``stix``. We
recommend reviewing this foundation before diving into the
:doc:`tutorials <tutorials/index>`, as it will make navigating the API much
more intuitive.

.. _intro-framework:

The stochastic interpolants framework
-------------------------------------

Generative modeling can be described as a **transport** problem, where one
seeks to map samples from a source distribution :math:`p_{\mathrm{src}}` to
samples of a target distribution :math:`p_{\mathrm{tgt}}`.

Modern generative models achieve this by evolving samples from the source
according to a Markovian dynamical process, whose generators are fitted so that
the distribution of the evolved samples matches the target. Stochastic
interpolants provide a generic framework for constructing such a process. Here
we use a slightly extended version of the original construction, so that the
same picture also encompasses discrete flow matching, and interpolants whose
endpoints might not match the source and target exactly.

Given a source and a target distribution :math:`p_{\mathrm{src}}` and
:math:`p_{\mathrm{tgt}}`, an interpolant is a process :math:`z_t` on the time
interval :math:`t \in [0, 1]`, defined as

.. math::

   z_t = I_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}, \epsilon)

where :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})` is drawn from a joint
distribution :math:`\pi` whose marginals match :math:`p_{\mathrm{src}}` and
:math:`p_{\mathrm{tgt}}` respectively, and :math:`\epsilon` is a random noise
variable *independent* of :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.

Typically the interpolant is built so that the path endpoints match the pair
variables,

.. math::

   z_0 = z_{\mathrm{src}}, \qquad z_1 = z_{\mathrm{tgt}}.

The marginal laws of this process then form a continuous path of probability
distributions :math:`(p_t)_{0 \leq t \leq 1}` with :math:`p_0 = p_{\mathrm{src}}`
and :math:`p_1 = p_{\mathrm{tgt}}`.

Having defined an interpolant, one then fits the generator of the considered
dynamical process so as to reproduce its path of marginals, and uses this
learned generator to evolve samples of the source distribution into samples of
the target one.

In ``stix``, interpolants are represented by
:class:`~stix.core.interpolant.Interpolant` objects, whose
:meth:`~stix.core.interpolant.Interpolant.interpolate` method implements the
map :math:`I_t`. The generators of the process are encapsulated in subclasses
of a :class:`~stix.core.generator.Generator` data class. Two types of Markovian
processes are implemented: continuous processes, defined by a stochastic or an
ordinary differential equation (S/ODE), and discrete processes defined by a
continuous-time Markov Chain (CTMC). Each type of Markovian process is
associated with a given ``Generator`` subclass, which is a property of the
corresponding ``Interpolant``.

A generative model is completely defined by a
:class:`~stix.core.gen_model.GenerativeModel` object. Importantly, this object
is responsible for providing the ``Generator``\s that are required to evolve
the dynamical process at sampling time. It encapsulates the learnable
components that allow to compute the ``Generator``\s, as well as everything
that defines the strategy for training these components.

.. _intro-multimodal:

Multi-modal setups and embeddings
---------------------------------

``stix`` is designed to support multi-modal generative models. We distinguish
two data spaces: a *raw* data space, which describes data in the form in which
they are observed or stored, and an *embedded* space in which the interpolation
takes place. These spaces may coincide in practice, but keeping them separate
allows the model to choose a representation that is more convenient for the
generative task, for example, by carrying out the dynamics in a latent space
rather than directly on the observed data. We generally denote variables in the
raw data space by :math:`x` and their counterparts in the embedded space by
:math:`z`. The maps between these spaces are encapsulated in
:class:`~stix.core.embedder.Embedder` objects.

In the embedded space, variables are decomposed into a collection of
*modalities* of different shapes and nature (discrete or continuous), each one
associated with a specific interpolant. Modalities are described by
:class:`~stix.core.modality.Modality` objects containing, among other things,
the corresponding ``Embedder`` and ``Interpolant``. The collection of
modalities is managed by a single
:class:`~stix.core.modality.ModalityRegistry`, which contains a generic pytree
of ``Modality`` objects. The registry itself is owned by the
``GenerativeModel``, tying the representation of the data, the interpolation
strategy, and the corresponding generators together into a single object.

.. _intro-one-sided:

One- and two-sided interpolants
-------------------------------

Interpolants can be partitioned into two categories: the one- and two-sided
interpolants.

**One-sided** interpolants, implemented by
:class:`~stix.core.interpolant.OneSidedInterpolant`, do not depend on any
source variable :math:`z_{\mathrm{src}}`, so their interpolated variables read

.. math::

   z_t = I_t(z_{\mathrm{tgt}}, \epsilon).

The source variable has been forgotten, so the joint :math:`\pi` reduces to the
target distribution, and in principle :math:`z_0` depends only on
:math:`\epsilon`.

Interpolants that explicitly depend on both a source and a target variable are
said to be **two-sided**.

.. _intro-continuous:

Continuous interpolants
-----------------------

We call **continuous interpolant** an interpolant of the form

.. math::

   z_t = J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}) + \gamma_t\,\epsilon,
   \qquad \epsilon \sim \mathcal{N}(0, I).

These are the generic interpolants considered in (Albergo, Boffi, Vanden-Eijnden,
`JMLR 2025 <https://www.jmlr.org/papers/v26/23-1605.html>`_). They are
implemented by the :class:`~stix.core.interpolant.ContinuousInterpolant` class.

Sampling: ODE, SDE, and the Fokker–Planck equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The marginals of an interpolant of the form above satisfy the Fokker–Planck
equation

.. math::
   :label: eq-fokker-planck

   \partial_t p_{t}(z) = -\nabla_{z}\cdot(p_{t}b_{t}),

where :math:`b_t(z)` is the *marginal* velocity of the interpolant, defined as

.. math::

   b_t(z) = \mathbb{E}\bigl[\partial_t J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}})
   + \dot\gamma_t\,\epsilon \mid z_t = z\bigr].

Define the score of the time-:math:`t` marginal :math:`s_t(z)` and the velocity
field :math:`v_t(z)` as

.. math::

   \begin{aligned}
   s_t(z) &= \nabla_z \log p_t(z)\\
   v_t(z) &= \mathbb{E}\bigl[\partial_{t} J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}})
   \mid z_t = z\bigr]
   \end{aligned}.

We have :math:`s_t(z) = - \mathbb{E}\bigl[ \epsilon \mid z_t = z\bigr] / \gamma_t`,
so that the velocity can be written

.. math::

   b_t(z) = v_t(z) - \gamma_t\dot\gamma_t\,s_t(z).

One can show that, for any non-negative stochasticity schedule :math:`\lambda_t`,
the process defined by the **stochastic differential equation** (SDE)

.. math::

   \mathrm{d}z_t = \bigl(b_t(z_t) + \lambda_t\, s_t(z_t)\bigr)\,\mathrm{d}t
   + \sqrt{2\lambda_t}\,\mathrm{d}W_t

yields the same Fokker–Planck equation as the previous continuous interpolant.
As a result, assuming it starts from the same initial distribution :math:`p_{0}`,
this process reproduces the *same* marginals :math:`p_t` as the continuous
interpolant.

To produce samples from the target distribution, one can thus fit :math:`b_t`
and :math:`s_t` and evolve samples from :math:`p_{0}` using the above SDE.
Taking :math:`\lambda \equiv 0` turns the SDE into an ordinary differential
equation (ODE) that only involves :math:`b_t`, so one may choose not to fit
:math:`s_t` and sample with the ODE alone.

Training: conditional velocity and score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The marginal fields :math:`b_t` and :math:`s_t` are not available in closed
form, but their *conditional* counterparts given a pair
:math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})` and a noise sample :math:`\epsilon`
are. In fact, we have

.. math::

   \begin{aligned}
   b_t(z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})
   &= \partial_t J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}) + \dot\gamma_t\,\epsilon, \\
   s_t(z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})
   &= -\epsilon / \gamma_t
   \end{aligned}.

One can rely on these conditional quantities to fit the associated
unconditional ones. For instance, to fit the velocity, one can minimize

.. math::

   \mathcal{L}
   = \mathbb{E}
   \bigl\| \hat{b}_t(z_t) - b_t(z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}}) \bigr\|^2,

where the expectation is taken over
:math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})\sim\pi`,
:math:`\epsilon\sim\mathcal{N}(0,I)` and :math:`t\sim\mathcal{U}(0,1)`.
Expanding and using the property of conditional expectation shows that

.. math::

   \mathcal{L} = \mathbb{E}
   \bigl\| \hat{b}_t(z_t) - b_t(z_t) \bigr\|^2,

which is the desired unconditional objective. This method is also valid for
:math:`s_t` and :math:`v_t`. Note that this approach is not unique, and other
objectives might exist that allow one to estimate the quantities required for
sampling, depending on the exact continuous interpolant considered.

Linear continuous interpolants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An important subclass of continuous interpolants are the **linear continuous
interpolants** (:class:`~stix.core.interpolant.LinearInterpolant`) of the form

.. math::

   z_t = \alpha_t z_{\mathrm{src}} + \beta_t z_{\mathrm{tgt}} + \gamma_t \epsilon,

or, for the one-sided case
(:class:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant`),

.. math::

   z_t = \beta_t z_{\mathrm{tgt}} + \gamma_t \epsilon.

This class of interpolants encompasses many of the modern generative models,
including Flow Matching
(:class:`~stix.core.interpolant.FlowMatchingOneSidedInterpolant`,
:class:`~stix.core.interpolant.FlowMatchingTwoSidedInterpolant`,
:class:`~stix.core.interpolant.StochasticFlowMatchingTwoSidedInterpolant`),
diffusion models in the EDM framework
(:class:`~stix.core.interpolant.VarianceExplodingDiffusionOneSidedInterpolant`),
and Bayesian Flow Networks
(:class:`~stix.core.interpolant.ContinuousBFNOneSidedInterpolant`,
:class:`~stix.core.interpolant.DiscreteBFNOneSidedInterpolant`). The one-sided
linear path identifies :math:`z_0` with :math:`\epsilon`, which will matter when
choosing a coupling.

.. _intro-discrete:

Discrete interpolants
---------------------

The interpolants above yield continuous interpolated states. We also implement
a class of interpolants (:class:`~stix.core.interpolant.DiscreteInterpolant`)
taking values in a discrete space, following the
Discrete Flow Matching framework of (Gat et al.,
`NeurIPS 2024 <https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html>`_).
These interpolants produce interpolated variables in a discrete state space
:math:`\Sigma` with :math:`K` elements, with conditional marginal probability
paths of the form

.. math::

   p_t(z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})
   = \sum_{j=1}^{M} \kappa^j_t\, w^j(z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}}),

where :math:`w^j(\cdot \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})` are
time-independent mixture conditional distributions on :math:`\Sigma` and
:math:`\kappa_t = (\kappa^j_t)_{j=1}^{M}` is a *schedule* on the simplex
:math:`\Delta^{M-1}`, namely :math:`\kappa^j_t\ge 0`,
:math:`\sum_j\kappa^j_t=1`.

To obtain these marginals, we define the interpolant map as

.. math::

   I_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}, \epsilon) = s_J,
   \qquad
   s_J\sim w_J(\,\cdot\mid z_{\mathrm{src}}, z_{\mathrm{tgt}}).

where :math:`J\sim\mathrm{Cat}(\kappa_t)`. Here :math:`\epsilon` is used as a
random seed to first sample :math:`J`, and then sample :math:`s_J` from
:math:`w_J`. This is achieved by splitting :math:`\epsilon` into
:math:`\epsilon = (\epsilon_m, \epsilon_s)` with
:math:`\epsilon_m,\epsilon_s\sim\mathcal{U}[0,1)`, and using :math:`\epsilon_m`,
:math:`\epsilon_s` to draw :math:`J` and :math:`s_J` by inverse CDF sampling.

Conditionally on :math:`\epsilon`, :math:`z_t` is a point mass at :math:`I_t`.
Integrating :math:`\epsilon` recovers the marginal paths above.

Sampling: CTMC and probability velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Akin to the marginals of an SDE, which satisfy the Fokker–Planck equation
:eq:`eq-fokker-planck`, the marginals :math:`p_t(z)` associated with a CTMC
satisfy a continuity equation that reads

.. math::
   :label: eq-ctmc-continuity

   \partial_t p_{t}(z) = -\mathrm{div}_{z}(p_{t}u_{t}),

where :math:`u_t` is the *probability velocity*. It is defined by

.. math::

   u_{t} (z\mid y) = \left(\partial_h p_{t+h\mid t}(z\mid y)\right)_{h=0}

for all :math:`z, y \in \Sigma`, where

.. math::

   p_{t+h\mid t}(z\mid y) = \mathbb{P}\left[z_{t+h} = z\mid z_t = y\right]

are the *transition probabilities*, and :math:`\mathrm{div}_{z}` is the *discrete divergence operator* that reads

.. math::

   \mathrm{div}_{z}(p_t u_t) = \sum_{y\in\Sigma}
   (u_t(y\mid z) p_t(z) - u_t(z\mid y) p_t(y)).

Using the fact that the transition probabilities sum to 1, we have
:math:`\sum_{y\in\Sigma} p_{t+h\mid t}(y\mid z) = 1` for all :math:`z \in \Sigma`,
so that the continuity equation reduces to

.. math::

   \partial_t p_{t}(z) = \sum_{y\in\Sigma}u_t(z\mid y)p_t(y)\,.

Similar to the continuous case above, assuming the initial distributions :math:`p_0`
are the same, we can thus build a CTMC process reproducing the desired
marginals by learning an approximation to the probability velocity :math:`u_t`.

Then, at sampling time we can simulate the process by taking stochastic jumps between times
:math:`t` and :math:`t+\delta t` according to the Euler discretisation

.. math::

   z_{t+\delta t} \sim \delta_{z_t}(\cdot) + \delta t\, u_t(\cdot \mid z_t),

i.e.

.. math::

   z_{t + \delta t} =
   \begin{cases}
      z_t & \text{w.p. } 1 - \delta t \sum_{y \neq z_t} u_t(y \mid z_t), \\
      y \neq z_t & \text{w.p. } \delta t\, u_t(y \mid z_t).
   \end{cases}

Note that for the right-hand side to be a proper probability distribution
for a small enough :math:`\delta t`, the probability velocity must satisfy
the conditions

.. math::
   :label: eq-ctmc-positivity

   \sum_{z\in\Sigma} u_t(z\mid y) = 0
   \quad \text{and} \quad
   u_t(z\mid y) \geq 0 \quad \forall z\neq y.

The first condition ensures that the probabilities sum to 1, and the
second that they are non-negative.

Training: conditional probability velocity and posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As previously, the probability velocity is not available in closed form, but
one can use its *conditional* counterpart given
:math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`. Following (Gat et al.,
`NeurIPS 2024 <https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html>`_), for the previous interpolant
one can write the conditional probability velocity as

.. math::

   u_t(z \mid y, (z_{\mathrm{src}}, z_{\mathrm{tgt}}))
   = \sum_{j=1}^{M} a^j_t\, w^j(z \mid  (z_{\mathrm{src}}, z_{\mathrm{tgt}}))
   + b_t\,\delta_y(z),

with :math:`a^j_t = \dot\kappa^j_t - \kappa^j_t\,\dot\kappa^\ell_t / \kappa^\ell_t`,
:math:`b_t = \dot\kappa^\ell_t / \kappa^\ell_t`, and
:math:`\ell = \arg\min_j \dot\kappa^j_t / \kappa^j_t`.

In fact, injecting this conditional probability velocity into the
continuity equation :eq:`eq-ctmc-continuity` yields the correct time
derivative for the conditional marginals
:math:`p_t(z\mid z_{\mathrm{src}}, z_{\mathrm{tgt}})`. Note that the
terms involving the :math:`\ell` index in the coefficients
:math:`a_t` and :math:`b_t` are chosen so as to ensure the constraints
in :eq:`eq-ctmc-positivity`.

One can show that

.. math::

   u_t(z \mid y) = \sum_{(z_{\mathrm{src}}, z_{\mathrm{tgt}})\in\Sigma^2}
   u_t(z \mid y, (z_{\mathrm{src}}, z_{\mathrm{tgt}}))
   p_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}\mid y)\,.

For the considered interpolants, this yields

.. math::

   u_t(z \mid y) = \sum_{j=1}^{M} a^j_t\, \hat w^j_t(z \mid  y) + b_t\,\delta_y(z)

where

.. math::

   \hat w^j_t = \sum_{(z_{\mathrm{src}}, z_{\mathrm{tgt}})\in\Sigma^2}
   w^j(z \mid  (z_{\mathrm{src}}, z_{\mathrm{tgt}}))
   p_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}\mid y)

are the posteriors of the :math:`w^j` given :math:`z_t = y`.

As in the continuous case, one can thus rely on the conditional velocity to
learn :math:`u_t`. In practice, since :math:`\Sigma` is finite, one can also
directly fit the posterior distribution
:math:`p_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}\mid z)` and build the
:math:`\hat w^j_t` to derive an estimator of :math:`u_t(z \mid y)`.

Time reversal and corrector sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuity equation :eq:`eq-ctmc-continuity` constrains the divergence
:math:`\mathrm{div}_{z}(p_t u_t)`, not the probability velocity itself. Just as
the continuous case admits a whole family of processes sharing the same
marginals (indexed by the stochasticity schedule :math:`\lambda_t`), for any :math:`\eta_t` such that
:math:`\mathrm{div}_{z}(p_t\eta_t) = 0`, :math:`u_t + \eta_t` also reproduces
the marginals :math:`p_t`.

A canonical example is given by the **time-reversed** velocity
:math:`\check u_t`, which walks the same path of marginals with *decreasing*
:math:`t`. For the mixture paths above it is obtained by substituting
:math:`\dot\kappa_t \to -\dot\kappa_t` in the coefficients of the previous
section,

.. math::

   \check u_t(z \mid y) = \sum_{j=1}^{M} \tilde a^j_t\, \hat w^j_t(z \mid y)
   + \tilde b_t\,\delta_y(z),

where :math:`(\tilde a_t, \tilde b_t)` are the coefficients
:math:`(a_t, b_t)` evaluated with :math:`\dot\kappa_t` replaced by
:math:`-\dot\kappa_t`. The substitution generally moves the index
:math:`\ell = \arg\min_j \dot\kappa^j_t / \kappa^j_t`, which is precisely what
keeps the constraints :eq:`eq-ctmc-positivity` satisfied in the reversed
direction. By construction we have

.. math::

   \mathrm{div}_{z}(p_t \check u_t) = -\mathrm{div}_{z}(p_t \hat u_t)
   = \partial_t p_{t},

so :math:`\hat u_t + \check u_t` is divergence-free against :math:`p_t`, and
the whole family

.. math::
   :label: eq-ctmc-corrector

   \bar u_t = (1 + \lambda_t)\,\hat u_t + \lambda_t\,\check u_t,
   \qquad \lambda_t \geq 0,

reproduces the marginals :math:`p_t`. Here :math:`\lambda_t` plays exactly the same role it
plays for the SDE: :math:`\lambda_t = 0` is the plain
generative chain, while :math:`\lambda_t > 0` adds a **corrector** that undoes
part of the transport and lets the forward term decide again. For a masked
diffusion path, for instance, :math:`\check u_t` sends already-revealed states
back onto the mask symbol, so :math:`\lambda_t > 0` is the familiar re-masking
corrector. Swapping the two weights, i.e.
:math:`\bar u_t = \lambda_t\,\hat u_t + (1 + \lambda_t)\,\check u_t`, runs the
chain backwards instead, transporting the target back onto the source.

In ``stix``, the reversed velocity is produced by the ``backward=True``
argument of
:meth:`~stix.core.interpolant.DiscreteInterpolant.rates_from_mixture_distributions`,
and of the ``rates_from_target_posterior`` methods of the standard discrete
interpolants. Either direction comes out as valid transition rates, and both
are carried by the :class:`~stix.core.generator.TransitionRates` generator: its
``forward_rates`` hold :math:`\hat u_t` and its ``backward_rates`` hold
:math:`\check u_t`, so a solver assembles :eq:`eq-ctmc-corrector` as written.
The weight is the ``stochasticity_scale`` field of
:class:`~stix.sampling.solver_manual.ManualSolverConfig` — the very field that
carries :math:`\lambda_t` for a continuous SDE — which defaults to
:math:`\lambda_t \equiv 0`.

Finally, both the continuous and the discrete processes can be integrated in
either direction, as selected by the :class:`~stix.sampling.utils.Direction`
field of the solver configurations. On
:attr:`~stix.sampling.utils.Direction.REVERSE`, an S/ODE is integrated from
:math:`t=1` down to :math:`t=0` with the sign of its score term flipped, and a
CTMC swaps the two weights of :eq:`eq-ctmc-corrector`, so that the default
:math:`\lambda_t \equiv 0` becomes the pure reversed chain
:math:`\bar u_t = \check u_t`.

.. _intro-endpoints:

Approximate endpoints and sampling time
---------------------------------------

For two-sided interpolants, the identification :math:`z_0 = z_{\mathrm{src}}`,
:math:`z_1 = z_{\mathrm{tgt}}` is the intended design, but in a generic
implementation it may hold only approximately. For instance, one can consider a
linear two-sided interpolant of the form

.. math::

   z_t = t z_{\mathrm{tgt}} + (1-t) z_{\mathrm{src}} + \gamma_t \epsilon,
   \quad \epsilon\sim\mathcal{N}(0,1)

with :math:`\gamma_0` and :math:`\gamma_1` very small but non-zero, leaving a
small residual noise at both endpoints. In order to avoid confusion, we keep
distinct notations in the library, using :math:`z_{\mathrm{src}}` and
:math:`z_{\mathrm{tgt}}` for the *source* and *target* variables, and
:math:`z_0` and :math:`z_1` for the *initial* and *final* variables.

One-sided interpolants use no source distribution, so their initial state
:math:`z_0` should typically be determined by the noise :math:`\epsilon`. Again,
this might not always be true in practice. As an example, the one-sided
interpolant corresponding to a variance exploding (VE) diffusion model
(:class:`~stix.core.interpolant.VarianceExplodingDiffusionOneSidedInterpolant`)
reads

.. math::

   z_t = z_{\mathrm{tgt}} + \sigma_t\epsilon, \quad \epsilon\sim\mathcal{N}(0,1),
   \quad \sigma_1=0,\quad \sigma_0 = \sigma_{\mathrm{max}}\gg 1,

so that the initial state :math:`z_{0}` depends on :math:`z_{\mathrm{tgt}}`,
although it should be largely dominated by the noise term
:math:`\sigma_{\mathrm{max}}\epsilon`.

This kind of residual dependency on the target variable is problematic at
sampling time. Because :math:`z_{\mathrm{tgt}}` is unavailable, evaluating
:math:`z_{0}` using :math:`I_{t=0}` would require passing an arbitrary value for
:math:`z_{\mathrm{tgt}}`.

To avoid this arbitrary choice, every
:class:`~stix.core.interpolant.Interpolant` implements a
:meth:`~stix.core.interpolant.Interpolant.sample_initial_state` method that
maps the quantities available then — :math:`z_{\mathrm{src}}` (or none, for a
one-sided path) and :math:`\epsilon` — onto the sampling-time law of
:math:`z_0`, without using :math:`z_{\mathrm{tgt}}`.

.. _intro-coupling:

Source and target distributions, coupling
-----------------------------------------

Since the target (and sometimes the source) distribution is not available in
closed form, training relies on sampling :math:`(x_{\mathrm{src}}, x_{\mathrm{tgt}})`
pairs from a dataset and embedding them as
:math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.

In the interpolant framework one is free to choose the joint distribution of
the pair, :math:`\pi`, subject only to the correct marginals. Different
couplings change the geometry of the learned trajectories and can substantially
reduce the number of solver steps needed at sampling time, without changing
:math:`p_{\mathrm{src}}` or :math:`p_{\mathrm{tgt}}`.

In practice, the sampling from a joint distribution :math:`\pi` is simulated by
coupling the source and target variables to introduce specific correlations.
This can be done at the level of the dataset itself, in which case the pairs
drawn from the dataset are already correlated. We call this approach an
**offline coupling**.

Another approach is to introduce the correlations on the fly at the level of
each batch, a method we call **online coupling**. In ``stix``, online coupling
can be implemented using a :class:`~stix.core.coupling.Coupling` object. At
training time, the ``Coupling`` receives a batch of raw and embedded pairs and
returns correlated re-coupled pairs.

Couplings act on **source and target only**. In particular, *they cannot be
used with one-sided interpolants*, for which no source variable is provided.
Following the discussion of the previous paragraph, one could consider using
couplings acting on the target variable :math:`z_\mathrm{tgt}` and the noise
:math:`\epsilon`. However, this would break the central assumption that
:math:`\epsilon` is independent of :math:`z_\mathrm{tgt}`. This assumption is
required to derive the formula

.. math::

   s_t = -\epsilon / \gamma_t

that relates the continuous conditional score and the noise. Correlating
:math:`\epsilon` and :math:`z_\mathrm{tgt}` would thus break SDE sampling
(although ODE sampling would be fine).

Instead of coupling the target variable with the noise, one must thus turn the
one-sided interpolant into an equivalent two-sided one. For instance, a linear
one-sided interpolant of the form

.. math::

   z_t = \beta_t z_{\mathrm{tgt}} + \gamma_t \epsilon,
   \quad \epsilon\sim\mathcal{N}(0,1)

can be replaced by a two-sided one of the form

.. math::

   z_t = \beta_t z_{\mathrm{tgt}} + \alpha_t  z_{\mathrm{src}} + \sigma_t \epsilon,
   \quad \epsilon\sim\mathcal{N}(0,1),

with :math:`\gamma_t^2 = \alpha_t^2+\sigma_t^2` and
:math:`z_{\mathrm{src}}\sim\mathcal{N}(0,1)`. A specific choice of the
:math:`\alpha_t` and :math:`\sigma_t` schedules then corresponds to a specific
decomposition of the initial variable :math:`z_0` into a part
:math:`\alpha_0  z_{\mathrm{src}}` that can be correlated to
:math:`z_{\mathrm{tgt}}` and an independent part :math:`\sigma_0 \epsilon` that
is used to build the score :math:`s_t = -\epsilon/\sigma_t`.

Implementing this requires being able to specify the marginal distribution of
:math:`z_{\mathrm{src}}` directly in the embedded space. When the embedding is
non-trivial, defining this distribution as the image of a distribution of the
raw source :math:`x_{\mathrm{src}}` can be difficult. To avoid this, we allow
generating source variables in embedded space on the fly. This is done by
attaching an ``embedded_source_prior`` method to the
:class:`~stix.core.modality.Modality` object. After embedding of the available
raw source variables :math:`x_{\mathrm{src}}`, missing embedded source
:math:`z_{\mathrm{src}}` are generated using the provided
``embedded_source_prior``. At training time, this occurs *before* any coupling,
so the generated source can be used for online coupling.
