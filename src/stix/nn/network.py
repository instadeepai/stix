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

"""The ``Network`` contract: the interface a ``GenerativeModel`` expects.

``GenerativeModel.get_network_output`` invokes the network positionally as
``network(z_t, t, context_data, context_mask, attention_mask)``. Any network
plugged into a generative model must accept exactly that call. This module defines
the abstract base that encodes the contract; see
:class:`stix.nn.EncoderBackboneDecoderNetwork` for the default
encoder -> backbone -> decoder implementation.
"""

from abc import ABC, abstractmethod

from flax import nnx
from jaxtyping import PyTree

from stix.typing import Mask, Time, Var


class Network(nnx.Module, ABC):
    """Abstract base defining the network call contract.

    A :class:`~stix.core.gen_model.GenerativeModel` treats its network as a black
    box invoked as ``network(z_t, t, context_data, context_mask,
    attention_mask)`` (see
    :meth:`~stix.core.gen_model.GenerativeModel.get_network_output`). Subclass this
    and implement :meth:`__call__` with a compatible signature to plug a network
    into a generative model.
    """

    @abstractmethod
    def __call__(
        self,
        z_t: PyTree[Var],
        t: Time,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> PyTree[Var | tuple[Var, Var]]:
        """Map per-modality noisy state ``z_t`` at time ``t`` to per-modality output.

        Args:
            z_t: Per-modality noisy state in embedded space.
            t: Time scalar.
            context_data: Context conditioning (a pytree) consumed by the
                network's context encoder.
            context_mask: Context masks gating ``context_data``.
            attention_mask: Per-modality attention masks.

        Returns:
            Per-modality network output; a bare ``Var`` for one-sided, a
            ``(forward, reverse)`` tuple for two-sided.
        """
        ...
