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

from flax import struct
from jaxtyping import PyTree

from stix.typing.core import Mask, Var
from stix.typing.variables import RawSourceTargetPair


class Batch(struct.PyTreeNode):
    """A batch of multimodal data, consumed by the loss pipeline and solver.

    A frozen ``flax.struct`` dataclass registered as a JAX pytree, so it flows
    through ``jit`` / ``vmap`` / ``jax.tree.map``.

    Attributes:
        raw_batch: Per-modality source/target pairs in raw (pre-embedding)
            space. ``source`` is ``None`` for one-sided (noise-to-data) setups
            and a data array for two-sided (data-to-data) transport.
        context_data: Optional conditioning signal fed to the network's context
            encoder (e.g. class labels); ``None`` when unconditional.
        context_mask: Optional per-modality mask gating ``context_data`` (e.g.
            zeroing a modality's contribution); ``None`` means fully conditional.
        attn_mask: Optional per-modality attention mask, e.g. to avoid attending
            to pad positions in variable-length sequences.
        loss_mask: Optional per-modality mask zeroing the loss at given
            positions, e.g. pad locations in variable-length sequences.
        is_discrete: Optional per-modality discreteness flags, mirroring the
            modality keys of ``raw_batch`` (``True`` for discrete
            modalities, ``False`` for continuous ones). Marked as static pytree
            metadata (``pytree_node=False``) so it is not treated as a traced
            leaf under ``jit`` / ``vmap`` and can be used for Python-level
            control flow. ``None`` when not provided.
    """

    raw_batch: PyTree[RawSourceTargetPair]
    context_data: PyTree[Var] | None = None
    context_mask: PyTree[Mask | None] | None = None
    attn_mask: PyTree[Mask | None] | None = None
    loss_mask: PyTree[Mask | None] | None = None
    is_discrete: PyTree[bool] | None = struct.field(pytree_node=False, default=None)
