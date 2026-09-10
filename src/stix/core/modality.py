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

import copy
from typing import Any, Callable

import jax
import jax.random as jr
from flax import nnx
from jaxtyping import PyTree

from stix.core.embedded_source_prior import EmbeddedSourcePrior
from stix.core.embedder import Embedder
from stix.core.generator import Generator
from stix.core.interpolant import Interpolant
from stix.core.interpolant.interpolant import OneSidedInterpolant
from stix.core.utils import infer_num_samples_from_raw_source, is_one_hot_vector
from stix.typing import (
    Batch,
    PRNGKeyArray,
    RawSourceTargetPair,
    Shape,
    Var,
)


def _is_field_value(field: str, value: Any) -> bool:
    """Whether ``value`` is a broadcastable leaf for ``field`` (vs a pytree to recurse into).

    This function must be updated for each new field.

    Args:
        field: Name of the modality field being set.
        value: Candidate leaf value.

    Returns:
        ``True`` if ``value`` is a leaf to hold intact, ``False`` if it is a pytree to recurse into.

    Raises:
        ValueError: If ``field`` is not a known modality field.
    """
    match field:
        case "interpolant":
            return isinstance(value, Interpolant)
        case "embedder":
            return isinstance(value, Embedder)
        case "num_categories":
            return isinstance(value, int)
        case "embedded_source_prior":
            return isinstance(value, EmbeddedSourcePrior)
        case _:
            raise ValueError(f"Unknown modality field: {field!r}")


class Modality[InterpolantType: Interpolant](nnx.Module):
    """Per-modality configuration.

    All constructor fields default to ``None`` so a registry can be built
    incrementally — e.g. an empty skeleton from ``ModalityRegistry.from_batch``,
    populated later via ``ModalityRegistry.set``.
    *However*, ``None`` is a *construction-time* state, *not* a claim that a field
    is optional at runtime: the type-level ``| None`` only reflects what is
    permitted during construction, and each consumer validates (and raises for)
    the fields it actually needs. E.g.

    - ``shape``, ``is_discrete``: read off the raw target by ``from_batch``
    - ``interpolant``, ``embedder``: required by ``GenerativeModel.__init__``
    - ``num_categories``: required for index-valued (categorical) discrete
        modalities when the raw ``shape`` does not carry the category count
        (namely, when the modality is not one-hot encoded)
    - ``embedded_source_prior``: optional; samples the embedded source for
        two-sided interpolants (training fill + sampling). Invalid with a
        one-sided interpolant.

    The sampling-time :attr:`generator_type` is inferred from the interpolant
    (see :attr:`~stix.core.interpolant.Interpolant.generator_type`).

    An ``nnx.Module``, so the modality is itself a JAX pytree and the parameters
    of a stored ``embedder`` are visible to NNX tracing.
    """

    def __init__(
        self,
        shape: Shape | None = None,
        is_discrete: bool | None = None,
        interpolant: InterpolantType | None = None,
        embedder: Embedder | None = None,
        num_categories: int | None = None,
        embedded_source_prior: EmbeddedSourcePrior | None = None,
    ):
        """Initialise a modality; any field may be ``None`` until populated.

        Args:
            shape: Shape of the modality's data (leading batch axis dropped).
            is_discrete: Whether the modality is discrete.
            interpolant: The :class:`~stix.core.interpolant.Interpolant` defining
                the interpolation path. It also determines the sampling-time :attr:`generator_type`.
            embedder: Maps the modality between raw and embedded space. Must be specified, even if identity.
            num_categories: Number of categories :math:`K` for a discrete
                modality. When discrete data is one-hot encoded the count is the
                last dimension of ``shape`` and this may be left ``None``; for
                index-valued (categorical) data ``shape`` is ``(L,)`` so the
                count must be given explicitly. A static Python ``int``.
            embedded_source_prior: Optional callable that samples an embedded
                source. Only valid with a two-sided interpolant.
        """
        self.shape = shape
        self.is_discrete = is_discrete
        self.num_categories = num_categories
        self.embedded_source_prior = embedded_source_prior

        # Attributes with learnable parameters but are (potentially) initialised with None
        # need to be wrapped in ``nnx.data`` so that when they are replaced they can be traced by autodiff.
        self.embedder = nnx.data(embedder)

        # If one ever wanted any ``Interpolant`` to have learnable parameters then this must
        # be wrapped with ``nnx.data`` to be traced. However, if we wrap this with ``nnx.data``
        # when no learnable parameters exist then ``nnx.split`` works incorrectly and JIT tracing
        # fails.
        self.interpolant = interpolant
        assert_embedded_source_prior_valid(self)

    @property
    def generator_type(self) -> type[Generator]:
        """Sampling-time generator class inferred from the interpolant.

        Returns:
            The :class:`~stix.core.generator.Generator` subclass declared on the
            interpolant (e.g. ``VelocityAndScore`` for a stochastic continuous
            path, ``TransitionRates`` for a discrete DFM interpolant).

        Raises:
            ValueError: If the interpolant has not been set yet.
        """
        if self.interpolant is None:
            raise ValueError(
                "generator_type is inferred from the interpolant, but interpolant "
                "is None. Set the modality's interpolant first."
            )
        return type(self.interpolant).generator_type


class ModalityRegistry(nnx.Module):
    """Container describing a pytree of :class:`Modality` objects.

    An ``nnx.Module``, so a model holding a registry (e.g. ``GenerativeModel``)
    exposes the registry-held :class:`Modality` params, such as a learnable
    ``embedder``, to NNX tracing and ``nnx.split``. The inner ``registry`` pytree
    is wrapped in ``nnx.data`` so NNX descends into it rather than treating it as
    an opaque static leaf; otherwise those params are invisible to ``nnx.split``
    and silently frozen at initialisation.
    """

    def __init__(self, registry: PyTree[Modality]):
        """Initialise the registry from a modality pytree.

        Args:
            registry: A pytree whose leaves are :class:`Modality` configs.
        """
        self.registry = nnx.data(registry)

    @property
    def treedef(self):
        """The registry's pytree structure with each ``Modality`` as a leaf."""
        return jax.tree.structure(
            self.registry, is_leaf=lambda m: isinstance(m, Modality)
        )

    def _modality_leaves(self) -> list[Modality]:
        """All :class:`Modality` leaves, for any pytree registry.

        Returns:
            The pytree leaves at which the modalities live.
        """
        return jax.tree.leaves(self.registry, is_leaf=lambda m: isinstance(m, Modality))

    @classmethod
    def from_batch(cls, batch: Batch) -> "ModalityRegistry":
        """Build a registry skeleton by inferring structure from a training ``Batch``.

        The attributes ``shape``, ``is_discrete`` and (when inferable)
        ``num_categories`` are read off the raw target array.

        - ``shape``: the per-example target shape (leading batch axis dropped).
        - ``is_discrete``: taken directly from ``batch.is_discrete``. Whatever
            builds the ``Batch`` must state discreteness explicitly.
        - ``num_categories``: inferred only for *one-hot* discrete data,
            where it is the last dimension of ``shape``. One-hot vectors are
            identified by checking they hold binary values and sum to 1
            (see :func:`~stix.core.utils.is_one_hot_vector`).
            An integer dtype is taken to be index-valued, never one-hot.
            For *index-valued* discrete data
            (integer targets, ``shape == (L,)``), the count cannot be inferred and
            is left ``None``; set it explicitly via
            ``registry.set("num_categories", ...)``.

        Args:
            batch: A training batch whose ``raw_batch`` and ``is_discrete``
                define the registry structure.

        Returns:
            A :class:`ModalityRegistry` with ``shape`` / ``is_discrete`` /
            ``num_categories`` filled where inferable.

        Raises:
            ValueError: If ``batch.is_discrete`` is ``None``.
        """
        if batch.is_discrete is None:
            raise ValueError(
                "ModalityRegistry.from_batch requires batch.is_discrete to be set "
                "(one bool per modality); it is None. Whatever builds the Batch "
                "must state each modality's discreteness explicitly."
            )

        def _modality_from_pair(
            pair: RawSourceTargetPair, is_discrete: bool
        ) -> "Modality":
            """Build a :class:`Modality` skeleton from one raw pair."""
            shape = pair.target.shape[1:]
            # One-hot discrete data carries the category count in its last axis;
            # index-valued (integer) discrete data does not, so leave it unset.
            num_categories = None
            if is_discrete and is_one_hot_vector(pair.target):
                num_categories = int(shape[-1])
            return Modality(
                shape=shape, is_discrete=is_discrete, num_categories=num_categories
            )

        return cls(
            jax.tree.map(
                _modality_from_pair,
                batch.raw_batch,
                batch.is_discrete,
                is_leaf=lambda x: isinstance(x, RawSourceTargetPair),
            )
        )

    def set(
        self,
        field: str,
        value: Any | PyTree[Any],
        filter_fn: Callable[[Modality], bool] | None = None,
        use_deepcopy: bool = True,
        is_factory: bool = False,
    ) -> None:
        """Set a per-modality field, broadcasting each argument over the registry.

        One can either pass a single value or a pytree. The value argument
        is broadcasted to every modality beneath it.

        When ``is_factory`` is ``True``, each leaf of ``value`` is treated as a
        factory ``Callable[[Modality], Any]`` and invoked with the target
        modality to produce that modality's value.

        Examples:
            Considering a registry shaped as
            ``{"mod_a": ..., "mod_b": {"mod_b1": ..., "mod_b2": ...}}``::

                # Broadcast one value to every modality.
                registry.set("interpolant", interpolant)

                # Prefix pytree: mod_a gets its own value; the single value
                # under "mod_b" fans out to both mod_b1 and mod_b2.
                registry.set(
                    "interpolant", {"mod_a": interpolant_a, "mod_b": interpolant_b}
                )

                # Fully-resolved pytree: one value per modality.
                registry.set(
                    "interpolant",
                    {"mod_a": itp_a, "mod_b": {"mod_b1": itp_b1, "mod_b2": itp_b2}},
                )

                # Factory: build each modality's value from its Modality.
                registry.set(
                    "embedder",
                    lambda m: IdentityEmbedder(m.shape),
                    is_factory=True,
                )

                # filter_fn: only touch matching modalities.
                registry.set(
                    "embedder", continuous_embedder, filter_fn=lambda m: not m.is_discrete
                )

        Args:
            field: The field to set.
            value: The value to set the field to; a leaf broadcasts to every
                modality beneath it. With ``is_factory``, each leaf is instead a
                factory ``Callable[[Modality], Any]``.
            filter_fn: Optional predicate; only modalities returning ``True`` are
                modified (``None`` affects all).
            use_deepcopy: Whether to deep-copy each broadcast value so modalities do not
                alias one shared object. Ignored under ``is_factory``. Default is ``True``.
                A plain function (e.g. a ``jr.normal`` prior) is copied atomically and so
                survives as the same object; a *stateful* callable is snapshotted per
                modality, so pass ``False`` to share one live object across them.
            is_factory: Treat every leaf of ``value`` as a factory called with the
                target modality to build its value.
        """

        def _resolve(modality: Modality, leaf: Any) -> Any:
            """Resolve a broadcast leaf into the value assigned to ``modality``."""
            if is_factory:
                if not callable(leaf):
                    raise TypeError(
                        f"is_factory=True requires callable leaves for field "
                        f"{field!r}, got {type(leaf).__name__}."
                    )
                # Fresh value per modality, so ``use_deepcopy`` is moot here.
                return leaf(modality)
            return copy.deepcopy(leaf) if use_deepcopy else leaf

        def _assign(modality: Modality, leaf: Any) -> None:
            """Assign the resolved leaf to ``modality.field`` when not filtered out."""
            if filter_fn is None or filter_fn(modality):
                setattr(modality, field, _resolve(modality, leaf))
                if field in ("interpolant", "embedded_source_prior"):
                    assert_embedded_source_prior_valid(modality)

        # Broadcast ``value`` (a single leaf, or a coarser prefix) into one leaf
        # per modality, then assign each modality its leaf. ``is_leaf`` marks this
        # field's broadcastable values so they are not recursed into; in factory
        # mode every callable is such a broadcastable leaf.
        is_leaf = callable if is_factory else (lambda x: _is_field_value(field, x))
        per_modality = self.broadcast(value, is_leaf=is_leaf)
        self.map(_assign, per_modality)

    def assert_compatible(self, *trees: PyTree) -> None:
        """Check each tree is structurally compatible with the registry.

        The registry's ``Modality``-leaf treedef must be a *prefix* of each tree.
        A tree may resolve finer below each modality (e.g. a
        ``RawSourceTargetPair`` splitting into ``source``/``target``, where a
        ``None`` source changes the leaf count), but its structure *above* the
        modalities must match the registry exactly.

        ``jax.tree.map`` enforces this implicitly when the registry is its first
        argument (via ``flatten_up_to``); this check runs it explicitly to raise a
        readable, registry-specific error instead.

        Args:
            *trees: The pytrees to validate against the registry.

        Raises:
            ValueError: If any tree is not compatible with the registry structure.
        """
        for index, tree in enumerate(trees):
            try:
                # Flatten the (higher-resolution) tree up to the registry's
                # modality skeleton; raises iff the registry is not a prefix.
                self.treedef.flatten_up_to(tree)
            except ValueError as error:
                raise ValueError(
                    f"Argument {index} is not compatible with the modality "
                    "registry: the registry's modality structure must be a prefix "
                    "of every tree mapped against it.\n"
                    f"Registry structure (modalities as leaves):\n  {self.treedef}\n"
                    f"Argument {index} structure:\n  {jax.tree.structure(tree)}"
                ) from error

    def assert_fields_set(self, *fields: str) -> None:
        """Raise if any modality leaf leaves any of ``fields`` unset (``None``).

        Args:
            *fields: Field names that must be non-``None`` on every modality.

        Raises:
            ValueError: If any modality has ``None`` for any requested field.
        """
        for modality in self._modality_leaves():
            for field in fields:
                if getattr(modality, field) is None:
                    raise ValueError(
                        f"Every modality must define '{field}', but it is None. "
                        f"Modality: {modality!r}."
                    )

    def assert_num_categories_set_on_discrete_modalities(self) -> None:
        """Raise if any discrete modality has ``num_categories`` not set."""
        for modality in self._modality_leaves():
            if modality.is_discrete and modality.num_categories is None:
                raise ValueError(
                    f"Every discrete modality must have 'num_categories' set, "
                    f"but it is None for modality {modality!r}."
                )

    def map(self, f: Callable, *trees: PyTree) -> PyTree:
        """``jax.tree.map`` over the registry, treating each ``Modality`` as a leaf.

        This is the canonical way to map over per-modality data. It bundles three
        operations together:

        1. Assert that all pytrees are compatible with the registry.
        2. Run ``jax.tree.map`` with the registry as the first pytree.
        3. Set ``is_leaf`` so that each ``Modality`` is treated as a leaf.

        Args:
            f: Callable receiving the modality followed by the corresponding leaf
                of each tree in ``trees``.
            *trees: Pytrees to map over, each compatible with the registry.

        Returns:
            A pytree with the registry's structure holding the results of ``f``.
        """
        self.assert_compatible(*trees)
        return jax.tree.map(
            f, self.registry, *trees, is_leaf=lambda m: isinstance(m, Modality)
        )

    def broadcast(
        self,
        prefix_tree: Any | PyTree[Any],
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        """Broadcast a prefix tree up to the registry's per-modality structure.

        The registry's modality-key structure is the target; each leaf of
        ``prefix_tree`` is copied to every modality beneath it. ``prefix_tree``
        may be coarser than the modality structure — a single value broadcast to
        all modalities, or one value per modality *group*.

        Use this for coarse->fine broadcasting (one value to many modalities);
        use :meth:`map` for per-modality computation. By default container values
        are descended as pytrees; pass ``is_leaf`` to hold a structured value
        intact at each modality (so a compound value broadcasts as a single leaf
        rather than being descended into). A broadcast leaf is *shared* (the same
        object) across modalities.

        Args:
            prefix_tree: A value or coarser pytree to broadcast onto the registry.
            is_leaf: Optional predicate marking values that should not be descended
                as pytrees during broadcasting.

        Returns:
            A pytree with the registry's modality structure, holding one
            ``prefix_tree`` leaf per modality.

        Raises:
            ValueError: If ``prefix_tree`` is not a prefix of the registry's modality
                structure.
        """
        # ``Modality`` is itself a pytree, so we cannot broadcast onto the registry
        # directly (jax would descend into each modality's fields). Broadcast onto a
        # placeholder skeleton that carries only the modality-key structure; the
        # placeholder leaf values are discarded and replaced by the prefix.
        skeleton = self.treedef.unflatten([0] * self.treedef.num_leaves)

        # Similar philosophy to the ``assert_compatible`` method above.
        # ``jax.tree.broadcast`` already raises when ``prefix_tree`` is not a valid
        # prefix of the skeleton.
        # Note: the direction is the *reverse* of ``assert_compatible``: here the prefix
        # tree must be coarser than (a prefix of) the registry, not finer.
        try:
            return jax.tree.broadcast(prefix_tree, skeleton, is_leaf=is_leaf)
        except ValueError as error:
            raise ValueError(
                "Prefix tree is not compatible with the modality registry: it must "
                "be a prefix of (coarser than, or equal to) the registry's modality "
                "structure, so each leaf broadcasts to one or more modalities.\n"
                f"Registry structure (modalities as leaves):\n  {self.treedef}\n"
                "Prefix-tree structure:\n  "
                f"{jax.tree.structure(prefix_tree, is_leaf=is_leaf)}"
            ) from error

    def split_and_project_key(self, key: PRNGKeyArray) -> PyTree[PRNGKeyArray]:
        """Split ``key`` into one independent key per modality, projected onto the registry structure.

        The PRNG-dual of :meth:`broadcast`: ``broadcast`` shares one value across
        modalities; this hands each its *own* key. Feed the result to :meth:`map`.

        Args:
            key: PRNG key to split across modalities.

        Returns:
            A pytree of independent keys with the registry's modality structure.
        """
        return jax.tree.unflatten(
            self.treedef, list(jr.split(key, self.treedef.num_leaves))
        )

    def sample_initial_state(
        self,
        key: PRNGKeyArray,
        raw_source: PyTree[Var | None] | None = None,
        num_samples: int | None = None,
    ) -> PyTree[Var]:
        r"""Sample the per-modality initial state :math:`z_0`.

        Resolves :math:`z_{\mathrm{src}}` and :math:`\epsilon` for every
        modality, then delegates to
        :meth:`~stix.core.interpolant.Interpolant.sample_initial_state`.

        Per modality:

        - a raw source array is embedded and used as :math:`z_{\mathrm{src}}`
          (wins over ``embedded_source_prior``);
        - ``None`` on a one-sided interpolant stays ``None``;
        - ``None`` on a two-sided interpolant with ``embedded_source_prior``
          is filled from that prior;
        - a raw source on a one-sided interpolant, or a two-sided interpolant
          with neither a raw source nor a prior, raises.

        ``raw_source=None`` means every leaf is missing. ``num_samples`` is
        inferred from any provided source; it is required when every leaf is
        missing. If both are given they must match.

        Args:
            key: PRNG key; split into one independent key per modality.
            raw_source: Optional pytree of raw source arrays aligned with the
                registry (``None`` leaves are missing). ``None`` itself is
                “all leaves missing”.
            num_samples: Leading batch axis. Required when no raw source is
                provided; must match the source batch size when both are given.

        Returns:
            Per-modality initial embedded state :math:`z_0`.

        Raises:
            ValueError: If ``raw_source`` and ``num_samples`` are both
                ``None``; if provided sources disagree on batch size or
                disagree with ``num_samples``; if a one-sided modality is
                given a raw source; or if a modality is missing its
                interpolant or embedder.
            TypeError: If a two-sided modality has neither a raw source nor
                ``embedded_source_prior``.
        """
        if raw_source is None:
            raw_source = self.treedef.unflatten([None] * self.treedef.num_leaves)
        self.assert_compatible(raw_source)

        inferred = infer_num_samples_from_raw_source(raw_source)
        if inferred is None and num_samples is None:
            raise ValueError(
                "sample_initial_state requires num_samples when every "
                "raw_source leaf is None."
            )
        if inferred is not None and num_samples is not None and inferred != num_samples:
            raise ValueError(
                f"num_samples={num_samples} does not match the raw_source "
                f"batch size {inferred}."
            )
        resolved_num_samples = inferred if inferred is not None else num_samples
        assert resolved_num_samples is not None

        init_keys = self.split_and_project_key(key)

        def _per_modality(
            modality: Modality, raw_src: Var | None, init_key: PRNGKeyArray
        ) -> Var:
            """Resolve z_src and epsilon, then assemble z_0."""
            interpolant = modality.interpolant
            embedder = modality.embedder
            if interpolant is None or embedder is None:
                raise ValueError(
                    "sample_initial_state requires every modality to have an "
                    "interpolant and an embedder."
                )
            assert_embedded_source_prior_valid(modality)
            shape = (resolved_num_samples, *embedder.embedding_shape)

            if raw_src is not None:
                if isinstance(interpolant, OneSidedInterpolant):
                    raise ValueError(
                        "raw_source is not valid with a one-sided interpolant."
                    )
                z_src = embedder.from_raw_to_embeddings(raw_src)
                epsilon = interpolant.sample_noise(init_key, shape)
            elif isinstance(interpolant, OneSidedInterpolant):
                z_src = None
                epsilon = interpolant.sample_noise(init_key, shape)
            elif modality.embedded_source_prior is not None:
                src_key, noise_key = jr.split(init_key)
                z_src = modality.embedded_source_prior(src_key, shape)
                epsilon = interpolant.sample_noise(noise_key, shape)
            else:
                raise TypeError(
                    "sample_initial_state requires a raw source or "
                    "embedded_source_prior for two-sided interpolant "
                    f"{type(interpolant).__name__}."
                )
            return interpolant.sample_initial_state(z_src, epsilon)

        return self.map(_per_modality, raw_source, init_keys)


def assert_embedded_source_prior_valid(modality: Modality) -> None:
    """Raise if ``embedded_source_prior`` is set on a one-sided interpolant.

    Args:
        modality: Modality whose prior / interpolant pair is checked.

    Raises:
        ValueError: If ``embedded_source_prior`` is set while the interpolant is
            one-sided.
    """
    if modality.embedded_source_prior is None:
        return
    if isinstance(modality.interpolant, OneSidedInterpolant):
        raise ValueError(
            "embedded_source_prior is only valid with a two-sided interpolant."
        )
