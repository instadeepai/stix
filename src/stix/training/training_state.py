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

"""Training state for the training loop."""

import jax
import optax
from flax import nnx, struct

from stix.typing import PRNGKeyArray


class TrainingState(struct.PyTreeNode):
    """Holds mutable training state across steps.

    Attributes:
        params: Current trainable parameters (``nnx.Param`` only).
        opt_state: The optax optimizer state.
        ema_params: Exponentially averaged parameters for evaluation.
        num_steps: Total number of optimizer steps taken (``jax.Array`` so it
            lives on-device and can be used inside JIT'd code).
        ema_decay: The EMA decay rate used to update ``ema_params``. Stored on
            the state (set from ``TrainingLoopConfig.ema_decay`` at init) so it
            is checkpointed alongside ``ema_params`` and ``num_steps``; this lets
            offline restore debias the EMA without the training config.
        rng_key: The PRNG key carrying the noise/time sampling stream, so a
            resumed run continues the same stream rather than reseeding. Stored
            as a ``PRNGKeyArray`` (uint32[2]) so it is checkpointed alongside the
            other leaves; inert inside the JIT'd train step (the step takes its
            key as an explicit argument).
        graphdef: The nnx graph definition (static structure of the model).
    """

    params: nnx.State
    opt_state: optax.OptState
    ema_params: nnx.State
    num_steps: jax.Array
    ema_decay: jax.Array
    # NOTE: rng_key lives here ONLY so it is checkpointed with the rest of the
    # state and a resumed run continues the same noise/time stream.
    # run() drives the live PRNG stream and writes
    # the current key into this field before each checkpoint.
    rng_key: PRNGKeyArray
    graphdef: nnx.GraphDef = struct.field(pytree_node=False)
