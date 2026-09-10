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

from abc import ABC, abstractmethod

from jaxtyping import PyTree

from stix.typing import EmbeddedSourceTargetPair, RawSourceTargetPair


class Coupling(ABC):
    r"""Abstract base class for the coupling function that correlates the source and target variables of a single batch.

    Couplings allow to simulate a sampling of the source and target
    variables from a joint distribution
    :math:`\pi(x_{\mathrm{src}}, x_{\mathrm{tgt}})` (or, equivalently,
    :math:`\pi(z_{\mathrm{src}}, z_{\mathrm{tgt}})`).

    This can only be used for *online* coupling, that is, couplings
    that operate on each batch independently. *Offline* couplings that require to access
    a full dataset at once must be implemented using custom data loaders.
    """

    @abstractmethod
    def __call__(
        self,
        raw_pairs: PyTree[RawSourceTargetPair],
        embedded_pairs: PyTree[EmbeddedSourceTargetPair],
    ) -> tuple[
        PyTree[RawSourceTargetPair],
        PyTree[EmbeddedSourceTargetPair],
    ]:
        """Couple the raw and embedded pairs.

        The same permutation (or other coupling function) is applied to the raw and
        embedded pairs, so both stay aligned.

        Args:
            raw_pairs: Per-modality raw ``(source, target)`` pairs.
            embedded_pairs: Per-modality embedded ``(source, target)`` pairs.

        Returns:
            The re-coupled ``(raw_pairs, embedded_pairs)``.
        """
        pass
