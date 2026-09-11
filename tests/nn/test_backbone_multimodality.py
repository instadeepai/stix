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

"""Multi-modality tests for the fusion backbones (``DiTBackbone``, ``MLPBackbone``).

The fuse/unfuse round trip itself is covered by ``tests/nn/test_fusion.py``;
these pin what the backbones do with it. Shapes alone are not enough — an offset
that drifts keeps every output length — so the slice boundaries and the mask
alignment are compared against a reference run over the joint stream.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jaxtyping import PyTree

from stix.nn import DiTBackbone, NetworkDimsConfig
from stix.nn.mlp import MLPBackbone

EMBED_DIM = 8
CONTEXT_DIM = 8


def _dit_backbone(num_tokens: PyTree[int]) -> DiTBackbone:
    return DiTBackbone(
        NetworkDimsConfig(
            embedding_dim=EMBED_DIM, context_dim=CONTEXT_DIM, ffn_hidden_dim=16
        ),
        modality_num_tokens=num_tokens,
        num_heads=2,
        num_layers=2,
        rngs=nnx.Rngs(0),
    )


def _encoded_a3_b5() -> dict[str, jax.Array]:
    """Distinct per-modality content over 3 and 5 tokens."""
    key = jax.random.key(0)
    return {
        "mod_a": jax.random.normal(jax.random.fold_in(key, 1), (3, EMBED_DIM)),
        "mod_b": jax.random.normal(jax.random.fold_in(key, 2), (5, EMBED_DIM)),
    }


def test_dit_backbone_slices_match_joint_stream():
    """Each modality's slice is its own span of the joint attention stream."""
    encoded = _encoded_a3_b5()
    joint = jnp.concatenate([encoded["mod_a"], encoded["mod_b"]], axis=0)
    reference = _dit_backbone({"solo": 8})({"solo": joint}, jnp.float32(0.5), None)
    reference_out = cast(jax.Array, reference["solo"])

    out = _dit_backbone({"mod_a": 3, "mod_b": 5})(encoded, jnp.float32(0.5), None)
    assert jnp.allclose(cast(jax.Array, out["mod_a"]), reference_out[:3])
    assert jnp.allclose(cast(jax.Array, out["mod_b"]), reference_out[3:])
    assert not jnp.allclose(cast(jax.Array, out["mod_a"]), reference_out[1:4])


def test_dit_backbone_attention_mask_gates_its_own_modality():
    """Masking one modality out matches attending over the other alone."""
    encoded = _encoded_a3_b5()
    masked = _dit_backbone({"mod_a": 3, "mod_b": 5})(
        encoded,
        jnp.float32(0.5),
        None,
        attention_mask={
            "mod_a": jnp.ones((3,), dtype=jnp.bool_),
            "mod_b": jnp.zeros((5,), dtype=jnp.bool_),
        },
    )
    alone = _dit_backbone({"mod_a": 3})(
        {"mod_a": encoded["mod_a"]}, jnp.float32(0.5), None
    )
    assert jnp.allclose(
        cast(jax.Array, masked["mod_a"]), cast(jax.Array, alone["mod_a"])
    )
    # The masked-out modality is gated too, not just excluded from the other's view.
    unmasked = _dit_backbone({"mod_a": 3, "mod_b": 5})(encoded, jnp.float32(0.5), None)
    assert not jnp.allclose(
        cast(jax.Array, masked["mod_b"]), cast(jax.Array, unmasked["mod_b"])
    )


def test_dit_backbone_declared_token_mismatch_raises():
    """Encoded token counts that disagree with ``modality_num_tokens`` raise."""
    backbone = _dit_backbone({"mod_a": 3, "mod_b": 5})
    bad_encoded = {
        "mod_a": jnp.ones((3, EMBED_DIM), dtype=jnp.float32),
        "mod_b": jnp.ones((4, EMBED_DIM), dtype=jnp.float32),  # declared 5, got 4
    }
    with pytest.raises(ValueError, match="do not match the declared modality_num"):
        backbone(bad_encoded, jnp.float32(0.5), context=None)


def _mlp_backbone() -> MLPBackbone:
    # embedding_dim is the concatenated feature width (3 + 5).
    cfg = MLPBackbone.Config(embedding_dim=8, backbone_depth=2, time_embedding_dim=4)
    return MLPBackbone(cfg, rngs=nnx.Rngs(0))


def test_mlp_backbone_keeps_each_feature_width():
    """Modalities of different feature widths keep their own widths on output.

    This backbone fuses along the feature axis, so the differing widths make a
    wrong fusion axis or split boundary a shape error.
    """
    encoded = [jnp.ones((4, 3), dtype=jnp.float32), jnp.full((4, 5), 2.0)]
    out = _mlp_backbone()(encoded, jnp.float32(0.5))
    assert cast(jax.Array, out[0]).shape == (4, 3)
    assert cast(jax.Array, out[1]).shape == (4, 5)
