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

"""Evaluation step for the training loop."""

from typing import Callable, TypeAlias

import jax
from flax import nnx
from jaxtyping import PyTree

from stix.training.loss_pipeline import LossPipeline
from stix.typing import PRNGKeyArray, Scalar
from stix.typing.data import Batch

EvalStepFn: TypeAlias = Callable[
    [nnx.State, Batch, PRNGKeyArray],
    PyTree[Scalar],
]


def _make_eval_step(
    *,
    graphdef: nnx.GraphDef,
    loss_pipeline: LossPipeline,
) -> EvalStepFn:
    """Build a JIT-compiled forward-only evaluation step.

    Args:
        graphdef: Static nnx graph definition (closed over).
        loss_pipeline: The loss pipeline (called inside JIT).

    Returns:
        A JIT-compiled callable ``(params, batch, key) -> metrics`` where
        ``metrics`` is the dict produced by the loss pipeline (at least
        ``{"loss": ...}``).
    """

    @jax.jit
    def _step(params: nnx.State, batch: Batch, key: PRNGKeyArray) -> PyTree[Scalar]:
        """JIT evaluation step returning loss-pipeline metrics."""
        gen_model = nnx.merge(graphdef, params)
        _, metrics = loss_pipeline(gen_model, batch, key)
        return metrics

    return _step
