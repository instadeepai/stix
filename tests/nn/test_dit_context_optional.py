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

"""End-to-end integration tests for DiT handling ``context=None`` gracefully.

The ``EncoderBackboneDecoderNetwork`` forwards ``context=None`` to backbone +
decoders when ``context_encoder`` is ``None``. The DiT components
(``AdaptiveLayerNorm``, ``TransformerBlock``, ``RegressionHead``,
``DiTDecoder``, ``DiTBackbone``) all accept ``Var | None`` and degrade to plain
pre-norm transformer behaviour when context is missing. These tests build a full
``EncoderBackboneDecoderNetwork`` with DiT components
and verify the whole stack runs end-to-end (eager + JIT) — catching any
regression where any single layer crashes on ``None``.
"""

from typing import cast

import jax
import jax.numpy as jnp
from flax import nnx

from stix.nn import (
    DiTBackbone,
    DiTDecoder,
    DiTEncoder,
    EncoderBackboneDecoderNetwork,
    NetworkDimsConfig,
)

EMBED_DIM = 8
CONTEXT_DIM = 8
NUM_HEADS = 2
NUM_LAYERS = 2
FFN_HIDDEN_DIM = 16
SEQ_LEN = 1


def _make_dit_network() -> EncoderBackboneDecoderNetwork:
    """Build a tiny DiT-based network with ``context_encoder=None``."""
    rngs = nnx.Rngs(0)
    dm_shape = {"mod_a": 2}
    dims = NetworkDimsConfig(
        embedding_dim=EMBED_DIM,
        context_dim=CONTEXT_DIM,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
    )
    encoders = {
        k: DiTEncoder(dims, input_dim=v, num_tokens=SEQ_LEN, rngs=rngs)
        for k, v in dm_shape.items()
    }
    decoders = {
        k: DiTDecoder(dims, output_dim=v, squeeze_sequence=True, rngs=rngs)
        for k, v in dm_shape.items()
    }
    backbone = DiTBackbone(
        dims,
        modality_num_tokens={k: SEQ_LEN for k in dm_shape},
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        rngs=rngs,
    )
    return EncoderBackboneDecoderNetwork(
        encoders=encoders,
        backbone=backbone,
        decoders=decoders,
        context_encoder=None,
    )


def test_network_with_dit_and_no_context_encoder_runs():
    """``EncoderBackboneDecoderNetwork(context_encoder=None)`` + DiT backbone runs end-to-end.

    The full forward — encode → backbone (all transformer blocks see
    ``context=None``) → decode — must produce finite outputs of the right
    shape.
    """
    net = _make_dit_network()
    z_t = {"mod_a": jnp.zeros((SEQ_LEN, 2), dtype=jnp.float32)}
    t = jnp.float32(0.5)
    out = net(z_t, t, context_data=None, context_mask=None, attention_mask=None)
    out_arr = cast(jax.Array, out["mod_a"])
    assert out_arr.shape == (2,)  # squeeze_sequence=True → (output_dim,)
    assert jnp.all(jnp.isfinite(out_arr))


def test_network_with_dit_no_context_jits():
    """The ``context=None`` path traces under ``jax.jit`` via the standard
    ``nnx.split`` / ``nnx.merge`` pattern. No Python-side conditional inside
    JIT that depends on a traced value (the ``if context is None`` check is
    on a static input).
    """
    net = _make_dit_network()
    graphdef, state = nnx.split(net)

    @jax.jit
    def call(state, z_t, t):
        merged = nnx.merge(graphdef, state)
        return merged(z_t, t)

    z_t = {"mod_a": jnp.zeros((SEQ_LEN, 2), dtype=jnp.float32)}
    out = call(state, z_t, jnp.float32(0.5))
    out_arr = cast(jax.Array, out["mod_a"])
    assert out_arr.shape == (2,)
    assert jnp.all(jnp.isfinite(out_arr))
