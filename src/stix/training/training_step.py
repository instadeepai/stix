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

"""Training step for the training loop."""

from typing import Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx
from jaxtyping import PyTree

from stix.training.ema import _ema_update
from stix.training.loss_pipeline import LossPipeline
from stix.training.training_state import TrainingState
from stix.typing import PRNGKeyArray, Scalar
from stix.typing.data import Batch

TrainStepFn: TypeAlias = Callable[
    [TrainingState, Batch, PRNGKeyArray],
    tuple[TrainingState, PyTree[Scalar]],
]


def _make_train_step(
    *,
    graphdef: nnx.GraphDef,
    loss_pipeline: LossPipeline,
    optimizer_tx: optax.GradientTransformation,
    ema_decay: float,
    num_gradient_accumulation_steps: int = 1,
) -> TrainStepFn:
    """Build a JIT-compiled training step.

    Handles both the single-batch and gradient-accumulation cases with the
    same returned callable. When ``num_gradient_accumulation_steps > 1``, the
    ``batch`` argument must be a stacked :class:`~stix.typing.data.Batch` whose leaves have a
    leading axis of length ``num_gradient_accumulation_steps``; accumulation
    is fused inside the JIT via ``lax.scan``.

    Args:
        graphdef: Static nnx graph definition (closed over).
        loss_pipeline: The loss pipeline (called inside JIT).
        optimizer_tx: An optax gradient transformation.
        ema_decay: EMA decay rate applied inside the step.
        num_gradient_accumulation_steps: Number of sub-batches to accumulate
            gradients over before applying an optimizer update.

    Returns:
        A JIT-compiled callable
        ``(state, batch, key) -> (new_state, metrics)`` where ``metrics`` is a
        dict containing at least ``"loss"``, ``"gradient_norm"`` and
        ``"param_update_norm"``.
    """
    n_accum = num_gradient_accumulation_steps

    def _loss_fn(
        params: nnx.State, batch: Batch, key: PRNGKeyArray
    ) -> tuple[Scalar, PyTree[Scalar]]:
        """Loss + metrics for ``value_and_grad``."""
        gen_model = nnx.merge(graphdef, params)
        return loss_pipeline(gen_model, batch, key)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    @jax.jit
    def _step(
        state: TrainingState,
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[TrainingState, PyTree[Scalar]]:
        """JIT training step with optional gradient accumulation."""
        params = state.params

        if n_accum == 1:
            (_, loss_metrics), grads = grad_fn(params, batch, key)
        else:
            keys = jr.split(key, n_accum)
            zero_grads = jax.tree.map(jnp.zeros_like, params)

            sample_batch = jax.tree.map(lambda x: x[0], batch)
            _, metrics_struct = jax.eval_shape(_loss_fn, params, sample_batch, keys[0])
            zero_metrics = jax.tree.map(
                lambda s: jnp.zeros(s.shape, s.dtype), metrics_struct
            )

            def _body(carry, scan_input):
                """Accumulate gradients and metrics over one sub-batch."""
                accum_grads, accum_metrics = carry
                sub_batch, sub_key = scan_input
                (_, m), g = grad_fn(params, sub_batch, sub_key)
                return (
                    jax.tree.map(jnp.add, accum_grads, g),
                    jax.tree.map(jnp.add, accum_metrics, m),
                ), None

            (grads, total_metrics), _ = jax.lax.scan(
                _body, (zero_grads, zero_metrics), (batch, keys)
            )
            grads = jax.tree.map(lambda g: g / n_accum, grads)
            loss_metrics = jax.tree.map(lambda m: m / n_accum, total_metrics)

        updates, new_opt_state = optimizer_tx.update(grads, state.opt_state, params)
        new_params = cast(nnx.State, optax.apply_updates(params, updates))
        new_ema_params = _ema_update(state.ema_params, new_params, ema_decay)

        metrics = {
            **loss_metrics,
            "gradient_norm": optax.global_norm(grads),
            "param_update_norm": optax.global_norm(updates),
        }

        new_state = state.replace(
            params=new_params,
            opt_state=new_opt_state,
            ema_params=new_ema_params,
            num_steps=state.num_steps + 1,
        )
        return new_state, metrics

    return _step
