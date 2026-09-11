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

"""Bundles multiple decoder heads sharing one backbone output."""

from flax import nnx

from stix.typing import Var


class MultiHeadDecoder(nnx.Module):
    """Bundles multiple decoder heads sharing one backbone output.

    Each head is called with the same ``(x, context=context)`` and the results
    are returned as a tuple, in the order given. Use this when a modality
    needs more than one prediction (e.g. velocity + score for two-sided
    stochastic interpolants).
    """

    def __init__(self, *heads: nnx.Module) -> None:
        """Initialise with at least 2 decoder heads.

        Args:
            *heads: The decoder heads, in the order their outputs are returned.

        Raises:
            ValueError: If fewer than 2 heads are given.
        """
        if len(heads) < 2:
            msg = f"MultiHeadDecoder needs at least 2 heads, got {len(heads)}"
            raise ValueError(msg)
        self.heads = nnx.List(list(heads))

    def __call__(self, x: Var, context: Var | None = None) -> tuple[Var, ...]:
        """Call each head with the same (x, context) and return results as a tuple."""
        return tuple(h(x, context=context) for h in self.heads)
