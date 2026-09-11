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

"""Shared dimensions for the DiT network components.

Several DiT components need the *same* widths: the embedding width
(``embedding_dim``), the conditioning-vector width (``context_dim``), the
feed-forward hidden width (``ffn_hidden_dim``), plus a common ``dtype``.
Declaring those on each component invites them to drift out of sync.

:class:`NetworkDimsConfig` holds them once. Construct it and inject the same
instance into each component (``DiTEncoder``, ``DiTBackbone``, ``DiTDecoder``,
``TimeNoiseContextEncoder``) so the shared dimensions cannot disagree, while
each component still takes its own specific arguments directly.
"""

from typing import Any

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, field_validator


class NetworkDimsConfig(BaseModel):
    """Dimensions shared across the DiT encoder, backbone, decoder and context encoder.

    Construct once and pass the same instance to each component so the shared
    widths stay consistent by construction.

    Attributes:
        embedding_dim: Embedding width shared by encoders, backbone and decoders.
        context_dim: Conditioning-vector width shared by the time context
            encoder, backbone and decoders.
        ffn_hidden_dim: Feed-forward hidden width shared by backbone and decoders.
        dtype: Parameter/compute dtype used throughout.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_dim: int = 128
    context_dim: int = 64
    ffn_hidden_dim: int = 256
    dtype: jnp.dtype = jnp.dtype(jnp.float32)

    @field_validator("dtype", mode="before")
    @classmethod
    def _coerce_dtype(cls, value: Any) -> jnp.dtype:
        """Accept dtype-likes (e.g. ``jnp.float32``) and normalise to a dtype.

        Components validate ``dtype`` as a concrete ``jnp.dtype`` instance, so
        coerce here before it is forwarded on.
        """
        return jnp.dtype(value)
