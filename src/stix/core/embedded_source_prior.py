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

from typing import Protocol, runtime_checkable

from stix.typing import EmbeddedVar, PRNGKeyArray, Shape


@runtime_checkable
class EmbeddedSourcePrior(Protocol):
    r"""Protocol for sampling an embedded source :math:`z_{\mathrm{src}}` from a prior.

    For modalities with a two-sided interpolant, this is
    can be used to fill missing ``embedded_pairs.source`` values
    at training and sampling time.

    NOTE: ``runtime_checkable`` so a prior can be recognised by ``isinstance``
        alongside the other modality fields. The check is structural: it tests that
        ``__call__`` exists, never its signature.
    """

    def __call__(self, key: PRNGKeyArray, shape: Shape) -> EmbeddedVar:
        r"""Sample an embedded source :math:`z_{\mathrm{src}}` of the given shape.

        Args:
            key: PRNG key for the draw.
            shape: Shape of the embedded source to draw (typically
                ``(batch, *embedding_shape)``).

        Returns:
            An embedded source sample :math:`z_{\mathrm{src}}`.
        """
        ...
