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

"""Tests for :class:`stix.nn.SumContextEncoder`.

Pins the masking + summing semantics: ``time_encoder`` always contributes,
each encoder in ``context_encoders`` is mapped over the matching
``context_data`` leaf and optionally gated by the matching ``context_mask``
leaf. Three "unconditional" expressions: whole-tree ``context_data=None``
(skips every conditioning encoder), a per-leaf ``context_mask`` of ``0``
(encoder runs, contribution zeroed), and a per-leaf ``context_mask`` of
``None`` ≡ fully conditional.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jaxtyping import PyTree

from stix.nn import SumContextEncoder
from stix.typing import Mask

CTX_DIM = 3


class _ConstantTimeEncoder(nnx.Module):
    """Context encoder that ignores ``t`` and returns a known constant vector."""

    def __init__(self, value: float, rngs: nnx.Rngs) -> None:
        del rngs
        self.value = nnx.Param(jnp.full((CTX_DIM,), value, dtype=jnp.float32))

    def __call__(self, t):
        del t
        return self.value[...]


class _IdentityOfTEncoder(nnx.Module):
    """Context encoder whose output broadcasts ``t`` to ``(CTX_DIM,)``.

    Lets a test observe that ``t`` flowed into the time encoder.
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        del rngs

    def __call__(self, t):
        return jnp.broadcast_to(jnp.asarray(t, jnp.float32), (CTX_DIM,))


class _EmbedContextEncoder(nnx.Module):
    """Tiny categorical context encoder mapping ``{0, 1}`` to per-class vectors."""

    def __init__(self, values: tuple[float, float], rngs: nnx.Rngs) -> None:
        del rngs
        v0 = jnp.full((CTX_DIM,), values[0], dtype=jnp.float32)
        v1 = jnp.full((CTX_DIM,), values[1], dtype=jnp.float32)
        self.table = nnx.Param(jnp.stack([v0, v1], axis=0))

    def __call__(self, idx):
        return self.table[...][idx]


def test_time_encoder_called_with_t():
    """The time encoder receives ``t`` verbatim — different ``t`` ⇒ different output."""
    ctx = SumContextEncoder(time_encoder=_IdentityOfTEncoder(rngs=nnx.Rngs(0)))
    assert jnp.allclose(ctx(jnp.float32(0.5)), 0.5)
    assert jnp.allclose(ctx(jnp.float32(0.7)), 0.7)


def test_context_data_none_returns_time_only():
    """``context_data=None`` skips every conditioning encoder; output is the time term alone."""
    TIME_VALUE = 1.0
    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=TIME_VALUE, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=(10.0, 20.0), rngs=rngs)
        },
    )
    out = ctx(jnp.float32(0.5), context_data=None, context_mask=None)
    assert jnp.allclose(out, TIME_VALUE)


def test_context_data_dict_sums_with_time():
    """A non-time encoder consumes ``context_data[key]`` and the result sums with time."""
    TIME_VALUE = 1.0
    LABEL_VALUES = (10.0, 20.0)

    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=TIME_VALUE, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=LABEL_VALUES, rngs=rngs)
        },
    )
    context_mask: PyTree[Mask | None] = {"label": jnp.array([True, True])}
    out = ctx(
        jnp.float32(0.5),
        context_data={"label": jnp.array([0, 1], dtype=jnp.int32)},
        context_mask=context_mask,
    )
    assert jnp.allclose(out[0], TIME_VALUE + LABEL_VALUES[0])
    assert jnp.allclose(out[1], TIME_VALUE + LABEL_VALUES[1])


def test_mask_zero_zeros_contribution():
    """``context_mask[key]=False`` makes the conditioning encoder behaviorally absent.

    Differential check: mask-off must match the time-only configuration; mask-on must differ.
    """
    TIME_VALUE = 1.0
    LABEL_VALUES = (10.0, 20.0)
    context_data = {"label": jnp.array([0, 1], dtype=jnp.int32)}

    ctx_with_label = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=TIME_VALUE, rngs=nnx.Rngs(0)),
        context_encoders={
            "label": _EmbedContextEncoder(values=LABEL_VALUES, rngs=nnx.Rngs(0))
        },
    )
    ctx_without_label = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=TIME_VALUE, rngs=nnx.Rngs(0)),
    )

    mask_off: PyTree[Mask | None] = {"label": jnp.zeros((2,), dtype=jnp.bool_)}
    mask_on: PyTree[Mask | None] = {"label": jnp.ones((2,), dtype=jnp.bool_)}

    out_masked = ctx_with_label(jnp.float32(0.5), context_data, mask_off)
    out_no_label = ctx_without_label(jnp.float32(0.5))
    out_unmasked = ctx_with_label(jnp.float32(0.5), context_data, mask_on)

    assert jnp.allclose(out_masked, out_no_label)
    assert not jnp.allclose(out_unmasked, out_masked)


def test_mask_none_is_fully_conditional():
    """``context_mask=None`` and a per-leaf ``mask=True`` produce identical output;
    both differ from ``mask=False`` (which suppresses the contribution).
    """
    TIME_VALUE = 1.0
    LABEL_VALUES = (10.0, 20.0)
    context_data = {"label": jnp.array([0, 1], dtype=jnp.int32)}

    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=TIME_VALUE, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=LABEL_VALUES, rngs=rngs)
        },
    )

    out_mask_none = ctx(jnp.float32(0.5), context_data, context_mask=None)
    out_mask_true = ctx(
        jnp.float32(0.5),
        context_data,
        context_mask={"label": jnp.ones((2,), dtype=jnp.bool_)},
    )
    out_mask_false = ctx(
        jnp.float32(0.5),
        context_data,
        context_mask={"label": jnp.zeros((2,), dtype=jnp.bool_)},
    )

    assert jnp.allclose(out_mask_none, out_mask_true)
    assert not jnp.allclose(out_mask_none, out_mask_false)


def test_context_data_without_encoders_raises():
    """Passing ``context_data`` when no ``context_encoders`` are configured is a hard error.

    The encoder has nothing to consume the data with — silently ignoring it
    would mask a likely user error (typo / leftover from a refactor).
    """
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=1.0, rngs=nnx.Rngs(0)),
    )
    with pytest.raises(ValueError, match="no context_encoders are"):
        ctx(
            jnp.float32(0.5),
            context_data={"label": jnp.array([0, 1], dtype=jnp.int32)},
        )


def test_context_mask_without_data_raises():
    """Passing ``context_mask`` with ``context_data=None`` is a hard error.

    A mask gates data; without data there is nothing to gate. Both must be
    provided together or both omitted.
    """
    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=1.0, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=(10.0, 20.0), rngs=rngs)
        },
    )
    with pytest.raises(ValueError, match="without context_data"):
        ctx(
            jnp.float32(0.5),
            context_data=None,
            context_mask={"label": jnp.array([True, True])},
        )


def test_partial_context_data_raises_clear_error():
    """A ``context_data`` whose structure disagrees with ``context_encoders`` is
    a hard error.

    Whole-tree ``None`` is lenient (unconditional everywhere). A structurally
    mismatched ``context_data`` (e.g. a missing/extra source) surfaces as a
    ``jax.tree.map`` structure ``ValueError`` — it is not silently ignored.
    """
    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=1.0, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=(10.0, 20.0), rngs=rngs)
        },
    )
    with pytest.raises(ValueError):
        ctx(
            jnp.float32(0.5),
            context_data={"wrong_key": jnp.array([0, 1], dtype=jnp.int32)},
            context_mask=None,
        )


def test_sum_combine_is_well_defined_for_multiple_context_encoders():
    """Time + two conditioning encoders sum cleanly."""
    TIME_VALUE = 1.0
    LABEL_A_VALUES = (1.0, 2.0)
    LABEL_B_VALUES = (10.0, 20.0)

    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=TIME_VALUE, rngs=rngs),
        context_encoders={
            "label_a": _EmbedContextEncoder(values=LABEL_A_VALUES, rngs=rngs),
            "label_b": _EmbedContextEncoder(values=LABEL_B_VALUES, rngs=rngs),
        },
    )
    context_mask: PyTree[Mask | None] = {
        "label_a": jnp.array([True]),
        "label_b": jnp.array([True]),
    }
    out = ctx(
        jnp.float32(0.5),
        context_data={
            "label_a": jnp.array([0], dtype=jnp.int32),
            "label_b": jnp.array([1], dtype=jnp.int32),
        },
        context_mask=context_mask,
    )
    expected = TIME_VALUE + LABEL_A_VALUES[0] + LABEL_B_VALUES[1]
    assert jnp.allclose(out, expected)


def test_gradients_flow_to_context_encoder_params():
    """Regression: gradients reach context-encoder params through ``nnx.data``
    storage.

    The pytree/``nnx.data`` design must keep the encoder module leaves visible
    to nnx tracing — otherwise training would silently stop updating them. Both
    the context encoder's table and the time encoder's value must receive a
    finite, nonzero gradient.
    """
    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=1.0, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=(10.0, 20.0), rngs=rngs)
        },
    )

    def loss_fn(module):
        out = module(
            jnp.float32(0.5),
            context_data={"label": jnp.array([0, 1], dtype=jnp.int32)},
            context_mask=None,
        )
        return jnp.sum(out**2)

    grads = nnx.grad(loss_fn)(ctx)
    grad_leaves = jax.tree.leaves(grads)
    assert len(grad_leaves) == 2  # time encoder value + context encoder table
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    assert all(jnp.any(g != 0.0) for g in grad_leaves)


def test_jit_traces_cleanly():
    """``SumContextEncoder`` traces under ``jax.jit`` via split/merge."""
    rngs = nnx.Rngs(0)
    ctx = SumContextEncoder(
        time_encoder=_ConstantTimeEncoder(value=0.5, rngs=rngs),
        context_encoders={
            "label": _EmbedContextEncoder(values=(1.0, 2.0), rngs=rngs),
        },
    )
    graphdef, state = nnx.split(ctx)

    @jax.jit
    def call(state, t, cond, mask):
        merged = nnx.merge(graphdef, state)
        return merged(t, cond, mask)

    context_mask: PyTree[Mask | None] = {"label": jnp.array([True, True])}
    out = call(
        state,
        jnp.float32(0.5),
        {"label": jnp.array([0, 1], dtype=jnp.int32)},
        context_mask,
    )
    assert out.shape == (2, CTX_DIM)
    assert jnp.all(jnp.isfinite(out))
