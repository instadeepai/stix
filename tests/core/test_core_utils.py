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

"""Tests for the core utility helpers."""

import jax
import jax.numpy as jnp

from stix.core.utils import is_one_hot_vector


def test_is_one_hot_vector_rejects_integer_dtype():
    """Index-valued data that *looks* one-hot must not be read as an encoding.

    ``[[0, 1, 0, 0], [1, 0, 0, 0]]`` as ``int32`` is a batch of four-token index
    sequences, but it is binary and sums to 1 along the last axis. Without the
    dtype test, ``ModalityRegistry.from_batch`` infers ``num_categories = 4`` for
    it — the wrong count, and the wrong embedder width under a factory.
    """
    index_valued = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=jnp.int32)
    assert not bool(is_one_hot_vector(index_valued))

    genuinely_one_hot = jax.nn.one_hot(jnp.array([1, 0]), 4)
    assert bool(is_one_hot_vector(genuinely_one_hot))
