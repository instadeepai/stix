# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Protocol

import jax
from jaxtyping import PyTree

from stix.core.gen_model.gen_model import GenerativeModel
from stix.core.generator import Generator, VelocityAndScore
from stix.typing import Mask, Scalar, Time, Var


class GuidanceFn(Protocol):
    """Protocol for guidance recipes.

    A guidance recipe builds per-modality :class:`~stix.core.generator.Generator` objects
    from a given :class:`~stix.core.gen_model.GenerativeModel`, embedded state :math:`z_t`,
    time :math:`t` and conditioning data.

    The solvers (:class:`~stix.sampling.solver.Solver` and its sibling
    :class:`~stix.sampling.solver_manual.ManualSolver`) call the configured recipe
    once per integration step to get the generators that drive the step. Every
    recipe shares this one signature so recipes slot interchangeably into
    the ``guidance_fn`` field of :class:`~stix.sampling.solver.SolverConfig`.

    Conditioning data are passed through two independent channels:

    - **context** (``context_data`` / ``context_mask``): a pytree fed to
      the generative model's network forward. Typically used for classifier-free guidance.

    - **intrinsic** (``intrinsic_data`` / ``intrinsic_mask``): conditioning data that are not fed to the network forward.
      Typically, these are data encoding conditions on the desired target state (e.g. a desired part of an image for image generation).

    This is a structural (callback) protocol, *not* a base class to subclass.
    Any plain function whose positional parameters align with ``__call__``
    satisfies it::

        def __call__(
            self,
            gen_model: GenerativeModel,
            z_t: PyTree[Var],
            t: Time,
            context_data: PyTree[Var] | None,
            context_mask: PyTree[Mask | None] | None,
            intrinsic_data: PyTree[Var] | None,
            intrinsic_mask: PyTree[Mask | None] | None,
            attention_mask: PyTree[Mask | None] | None,
            guidance_scale: PyTree[Callable[[Time], Scalar]],
        ) -> PyTree[Generator]:

    To add a recipe you merely write a function such as::

        def context_generator(gen_model, z_t, t, context_data, context_mask, *,
                               attention_mask, guidance_scale):
            net_out = gen_model.get_network_output(
                z_t, t, context_data, context_mask, attention_mask,
            )
            return gen_model.get_generator(net_out, z_t, t)

    and select it via ``SolverConfig(guidance_fn=context_generator)``.

    Help on defining custom implementations of this protocol can be found in
    the `conditioning and guidance tutorial
    <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_.
    """

    def __call__(
        self,
        gen_model: GenerativeModel,
        z_t: PyTree[Var],
        t: Time,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
        intrinsic_data: PyTree[Var] | None = None,
        intrinsic_mask: PyTree[Mask | None] | None = None,
        *,
        attention_mask: PyTree[Mask | None] | None = None,
        guidance_scale: PyTree[Callable[[Time], Scalar]],
    ) -> PyTree[Generator]:
        """Return the per-modality generator at time ``t``.

        Args:
            gen_model: Generative model computing the generators.
            z_t: Per-modality embedded state.
            t: Current time.
            context_data: Context conditioning data for the recipe.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Intrinsic-guidance conditioning data.
            intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks.
            guidance_scale: Per-modality time-dependent guidance scale.

        Returns:
            Per-modality :class:`~stix.core.generator.Generator`.
        """
        ...


def _require_score(generator_k: Generator) -> Var:
    """Extract the score from the :class:`~stix.core.generator.VelocityAndScore` :class:`~stix.core.generator.Generator`.

    The guided recipes operate on the score. A modality that declares a
    score-free generator type (e.g. a plain :class:`~stix.core.generator.Velocity`
    or a :class:`~stix.core.generator.TransitionRates`) cannot be guided through
    the score, so this raises a clear error rather than an obscure
    ``AttributeError``.

    Args:
        generator_k: :class:`~stix.core.generator.Generator` produced for that modality.

    Returns:
        The score leaf of ``generator_k``.

    Raises:
        TypeError: If ``generator_k`` is not a
            :class:`~stix.core.generator.VelocityAndScore`.
    """
    if not isinstance(generator_k, VelocityAndScore):
        raise TypeError(
            "Score-based guidance requires a VelocityAndScore generator, but a "
            f"modality returned {type(generator_k).__name__}. Use a stochastic "
            "continuous interpolant (which infers VelocityAndScore), or write a "
            "custom recipe that builds the generator directly."
        )
    return generator_k.score


def _with_conditional_score(
    modality, conditional_score: Var, z_t_k: Var, t: Time
) -> VelocityAndScore:
    r"""Build a :class:`~stix.core.generator.VelocityAndScore` from a score.

    Re-derives a velocity consistent with ``conditional_score`` via the
    interpolant's
    :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`
    (the score-to-velocity bridge the guided recipes call after computing the
    conditional score), then packages both into the generator the solver steps.

    Args:
        modality: Modality whose interpolant provides
            :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`.
        conditional_score: Guided score to package.
        z_t_k: Noisy state for this modality at time ``t``.
        t: Current time.

    Returns:
        A :class:`~stix.core.generator.VelocityAndScore` with a
        velocity consistent with ``conditional_score``.

    Raises:
        TypeError: If the modality's interpolant does not implement
            :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`
            (closed-form for one-sided linear SI). For such families, write a
            custom recipe that builds the generator directly rather than going
            through the score.
    """
    interpolant = modality.interpolant
    if not hasattr(interpolant, "velocity_from_score"):
        raise TypeError(
            f"{type(interpolant).__name__} does not implement 'velocity_from_score'; "
            "score-based guidance re-derives the velocity from the conditional "
            "score via it. Closed-form exists for one-sided linear SI. For another "
            "interpolant family, write a custom recipe that builds the generator "
            "directly."
        )
    velocity = interpolant.velocity_from_score(conditional_score, z_t_k, t)
    return VelocityAndScore(velocity=velocity, score=conditional_score)


def get_context_conditional_generator(
    gen_model: GenerativeModel,
    z_t: PyTree[Var],
    t: Time,
    context_data: PyTree[Var] | None = None,
    context_mask: PyTree[Mask | None] | None = None,
    intrinsic_data: PyTree[Var] | None = None,
    intrinsic_mask: PyTree[Mask | None] | None = None,
    *,
    attention_mask: PyTree[Mask | None] | None = None,
    guidance_scale: PyTree[Callable[[Time], Scalar]],
) -> PyTree[Generator]:
    r"""Produce the generator from a single network forward, passing the context data to the network.

    The intrinsic channel data is ignored.

    Used as the default recipe in :class:`~stix.sampling.solver.SolverConfig`.

    Args:
        gen_model: Generative model computing the generators.
        z_t: Per-modality embedded state.
        t: Current time.
        context_data: Context conditioning data for the recipe.
        context_mask: Context masks paired with ``context_data``.
        intrinsic_data: Intrinsic-guidance conditioning data.
        intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
        attention_mask: Per-modality attention masks.
        guidance_scale: Per-modality time-dependent guidance scale. Ignored by
            this recipe.

    Returns:
        Per-modality :class:`~stix.core.generator.Generator`.
    """
    del intrinsic_data, intrinsic_mask, guidance_scale
    net_out = gen_model.get_network_output(
        z_t,
        t,
        context_data,
        context_mask,
        attention_mask,
    )
    return gen_model.get_generator(net_out, z_t, t)


def get_intrinsic_guidance_generator(
    gen_model: GenerativeModel,
    z_t: PyTree[Var],
    t: Time,
    context_data: PyTree[Var] | None = None,
    context_mask: PyTree[Mask | None] | None = None,
    intrinsic_data: PyTree[Var] | None = None,
    intrinsic_mask: PyTree[Mask | None] | None = None,
    *,
    attention_mask: PyTree[Mask | None] | None = None,
    guidance_scale: PyTree[Callable[[Time], Scalar]],
) -> PyTree[Generator]:
    r"""Score-based **classifier guidance** recipe.

    This recipe is only valid for generative models whose modalities' generators
    are all of type :class:`~stix.core.generator.VelocityAndScore`.

    Up to terms constant in :math:`z_t`, Bayes' rule gives

    .. math::
        s_{\text{cond}}(z_t, t)
        = \underbrace{\nabla_{z_t} \log p_t(z_t)}_{\text{unconditional score}}
        + \omega(t) \underbrace{\nabla_{z_t} \log p_t(c | z_t)}_{\text{guidance gradient}},

    where :math:`c` is the intrinsic conditioning data.

    The unconditional score is computed from the network output's generator
    (:meth:`~stix.core.gen_model.GenerativeModel.get_generator`); the
    guidance gradient is obtained by automatic differentiation of the model's
    :meth:`~stix.core.gen_model.GenerativeModel.get_guidance_loss`,
    which is interpreted as an approximation to :math:`-\log p(c | z_t)`.
    (See that method for precise details and caveats!).

    The conditional score :math:`s_{\text{cond}}(z_t, t)` is
    then used to re-derive a consistent velocity via the interpolant's
    :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`
    method, and packaged into a :class:`~stix.core.generator.VelocityAndScore`
    generator that is returned by the recipe.

    The context data are **not** ignored, they are passed to the network forward.

    For more information see the `conditioning and guidance tutorial
    <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_.

    Args:
        gen_model: Generative model computing the generators.
        z_t: Per-modality embedded state.
        t: Current time.
        context_data: Context conditioning data for the recipe.
        context_mask: Context masks paired with ``context_data``.
        intrinsic_data: Intrinsic-guidance conditioning data.
        intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
        attention_mask: Per-modality attention masks.
        guidance_scale: Per-modality time-dependent guidance scale.

    Returns:
        Per-modality intrinsic-guided generator (a
        :class:`~stix.core.generator.VelocityAndScore`).

    Raises:
        NotImplementedError: If ``gen_model`` does not override
            :meth:`~stix.core.gen_model.GenerativeModel.get_guidance_loss`:
            the recipe differentiates that model-side hook, so a model without
            it cannot drive intrinsic guidance.
        TypeError: If a modality's generator is not a
            :class:`~stix.core.generator.VelocityAndScore`, or if its interpolant
            does not implement
            :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`.
    """

    def _intrinsic_loss_with_aux(
        z_t: PyTree[Var],
    ) -> tuple[Scalar, PyTree[Var]]:
        """Return ``(intrinsic_loss, net_out)`` for ``value_and_grad``."""
        # Only the context channel reaches the network forward; the intrinsic
        # targets steer purely through the autodiff path below.
        net_out = gen_model.get_network_output(
            z_t,
            t,
            context_data,
            context_mask,
            attention_mask,
        )
        loss = gen_model.get_guidance_loss(
            net_out,
            z_t,
            t,
            intrinsic_data,
            intrinsic_mask,
        )
        return loss, net_out

    (_, net_out), grad_intrinsic_loss = jax.value_and_grad(
        _intrinsic_loss_with_aux, has_aux=True
    )(z_t)

    generator_unconditional = gen_model.get_generator(net_out, z_t, t)
    return gen_model.modality_registry.map(
        lambda modality, generator_k, guidance_scale_k, grad_k, z_t_k: (
            _with_conditional_score(
                modality,
                _require_score(generator_k) - guidance_scale_k(t) * grad_k,
                z_t_k,
                t,
            )
        ),
        generator_unconditional,
        guidance_scale,
        grad_intrinsic_loss,
        z_t,
    )


def get_classifier_free_guidance_generator(
    gen_model: GenerativeModel,
    z_t: PyTree[Var],
    t: Time,
    context_data: PyTree[Var] | None = None,
    context_mask: PyTree[Mask | None] | None = None,
    intrinsic_data: PyTree[Var] | None = None,
    intrinsic_mask: PyTree[Mask | None] | None = None,
    *,
    attention_mask: PyTree[Mask | None] | None = None,
    guidance_scale: PyTree[Callable[[Time], Scalar]],
) -> PyTree[Generator]:
    r"""Score-based **classifier-free guidance** recipe.

    This recipe is only valid for generative models whose modalities' generators
    are all of type :class:`~stix.core.generator.VelocityAndScore`.

    Up to terms constant in :math:`z_t`, Bayes' rule gives

    .. math::
        s_{\text{cond}}(z_t, t)
        \approx \omega(t)\, s(z_t, t, \text{context}=c)
        + (1 - \omega(t))\, s(z_t, t, \text{context}=\text{null}).

    The two scores are computed from the network output's generators, the
    first one using the context data, the second one using no context data
    (i.e. ``context_data=None, context_mask=None``).

    The combined score is then used to re-derive a consistent velocity via the interpolant's
    :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`
    method, and packaged into a :class:`~stix.core.generator.VelocityAndScore`
    generator that is returned by the recipe.

    The intrinsic channel data is ignored.

    For more information see the `conditioning and guidance tutorial
    <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_.

    Args:
        gen_model: Generative model computing the generators.
        z_t: Per-modality embedded state.
        t: Current time.
        context_data: Context conditioning data for the recipe.
        context_mask: Context masks paired with ``context_data``.
        intrinsic_data: Intrinsic-guidance conditioning data.
        intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
        attention_mask: Per-modality attention masks.
        guidance_scale: Per-modality time-dependent guidance scale.

    Returns:
        Per-modality classifier-free-guided generator (a
        :class:`~stix.core.generator.VelocityAndScore`).

    Raises:
        TypeError: If a modality's generator is not a
            :class:`~stix.core.generator.VelocityAndScore`, or if its interpolant
            does not implement
            :meth:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant.velocity_from_score`.
    """
    del intrinsic_data, intrinsic_mask
    net_out_uncond = gen_model.get_network_output(z_t, t, None, None, attention_mask)
    net_out_cond = gen_model.get_network_output(
        z_t,
        t,
        context_data,
        context_mask,
        attention_mask,
    )
    generator_unconditional = gen_model.get_generator(net_out_uncond, z_t, t)
    generator_conditional = gen_model.get_generator(net_out_cond, z_t, t)
    return gen_model.modality_registry.map(
        lambda modality,
        generator_uncond_k,
        generator_cond_k,
        guidance_scale_k,
        z_t_k: _with_conditional_score(
            modality,
            guidance_scale_k(t) * _require_score(generator_cond_k)
            + (1 - guidance_scale_k(t)) * _require_score(generator_uncond_k),
            z_t_k,
            t,
        ),
        generator_unconditional,
        generator_conditional,
        guidance_scale,
        z_t,
    )


def get_discrete_classifier_free_guidance_generator(
    gen_model: GenerativeModel,
    z_t: PyTree[Var],
    t: Time,
    context_data: PyTree[Var] | None = None,
    context_mask: PyTree[Mask | None] | None = None,
    intrinsic_data: PyTree[Var] | None = None,
    intrinsic_mask: PyTree[Mask | None] | None = None,
    *,
    attention_mask: PyTree[Mask | None] | None = None,
    guidance_scale: PyTree[Callable[[Time], Scalar]],
) -> PyTree[Generator]:
    r"""Classifier-free guidance recipe for discrete modalities, in logit space.

    This recipe is only valid for generative models whose modalities' generators
    are all of type :class:`~stix.core.generator.TransitionRates`.
    It also assumes that the network outputs the logits of the embedded target posterior distribution.

    The unconditional and conditional posterior distributions are combined
    through a tempered geometric mean

    .. math::
        \tilde{p}_{\mathrm{tgt}|t}(\cdot \mid z_t, c) \propto
        p_{\mathrm{tgt}|t}(\cdot \mid z_t, c)^{\omega(t)}\,
        p_{\mathrm{tgt}|t}(\cdot \mid z_t, \varnothing)^{1-\omega(t)},

    which is obtained from a linear combination of the pre-softmax logits

    .. math::
        \tilde{\hat{\ell}}(\mathrm{tgt}\mid z_t) = \omega(t)\,\hat{\ell}(\mathrm{tgt}\mid z_t, c) + (1 - \omega(t))\,\hat{\ell}(\mathrm{tgt}\mid z_t, \varnothing),

    where :math:`\hat{\ell}` is the network output approximating the logits.

    (The two logits are computed from the network outputs, the
    first one using the context data, the second one using no context data
    (i.e. ``context_data=None, context_mask=None``).)

    The intrinsic channel data is ignored.

    For more information see the `conditioning and guidance tutorial
    <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_.

    Args:
        gen_model: Generative model computing the generators.
        z_t: Per-modality embedded state.
        t: Current time.
        context_data: Context conditioning data for the recipe.
        context_mask: Context masks paired with ``context_data``.
        intrinsic_data: Intrinsic-guidance conditioning data.
        intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
        attention_mask: Per-modality attention masks.
        guidance_scale: Per-modality time-dependent guidance scale.

    Returns:
        Per-modality generator built from the guidance-combined network output
        (a :class:`~stix.core.generator.TransitionRates`).
    """
    del intrinsic_data, intrinsic_mask
    net_out_uncond = gen_model.get_network_output(z_t, t, None, None, attention_mask)
    net_out_cond = gen_model.get_network_output(
        z_t,
        t,
        context_data,
        context_mask,
        attention_mask,
    )
    guided_net_out = gen_model.modality_registry.map(
        lambda modality, out_uncond_k, out_cond_k, guidance_scale_k: (
            guidance_scale_k(t) * out_cond_k + (1 - guidance_scale_k(t)) * out_uncond_k
        ),
        net_out_uncond,
        net_out_cond,
        guidance_scale,
    )
    return gen_model.get_generator(guided_net_out, z_t, t)
